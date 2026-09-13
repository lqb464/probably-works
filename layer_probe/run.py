"""One encoder per run; see README.md for protocol and limits."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layer_probe.core import normalize, pool_tokens, partition, targets, ridge, retrieval, metrics, paired_ci


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_clip(args):
    from model.clip_model import CLIP
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = blob.get("model", blob.get("state_dict", blob))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    if any(k.startswith("base_model.") for k in state):
        state = {k[len("base_model."):]: v for k, v in state.items() if k.startswith("base_model.")}
    if "visual.proj" not in state:
        raise ValueError("Only repository-compatible ViT CLIP checkpoints are supported")
    count = lambda prefix: len({k[len(prefix):].split('.')[0] for k in state if k.startswith(prefix)})
    width = state["ln_final.weight"].numel()
    model = CLIP(embed_dim=state["text_projection"].shape[1], image_resolution=tuple(args.image_size),
                 vision_layers=count("visual.transformer.resblocks."), vision_width=state["visual.conv1.weight"].shape[0],
                 vision_patch_size=state["visual.conv1.weight"].shape[-1], stride_size=args.stride,
                 context_length=state["positional_embedding"].shape[0], vocab_size=state["token_embedding.weight"].shape[0],
                 transformer_width=width, transformer_heads=width // 64, transformer_layers=count("transformer.resblocks."))
    # No fuzzy suffix matching and no silent randomly initialized encoder weights.
    for key in ("logit_scale", "input_resolution", "context_length", "vocab_size"):
        state.pop(key, None)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(args.device)


def dataset_and_splits(args):
    from datasets.rstpreid import RSTPReid
    from datasets.cuhkpedes import CUHKPEDES
    from datasets.icfgpedes import ICFGPEDES
    ds = {"rstp": RSTPReid, "cuhk": CUHKPEDES, "icfg": ICFGPEDES}[args.dataset](root=args.root, verbose=False)
    def subset(source, ids):
        return {k: [v for v, pid in zip(source[k], source["image_pids" if k in ("image_pids", "img_paths") else "caption_pids"]) if pid in set(ids)] for k in source}
    if args.dataset == "icfg":
        if not args.allow_test_holdout:
            raise ValueError("ICFG has no official val: explicitly pass --allow-test-holdout for a NON-official 40/20/40 identity split")
        groups = partition(ds.test["caption_pids"], [.4, .2, .4], args.split_seed)
        splits = {name: subset(ds.test, ids) for name, ids in zip(("fit", "select", "test"), groups)}
        protocol = "ICFG test identity holdout; NOT official full-test performance"
    else:
        groups = partition(ds.val["caption_pids"], [.5, .5], args.split_seed)
        splits = {"fit": subset(ds.val, groups[0]), "select": subset(ds.val, groups[1]), "test": ds.test}
        protocol = "official val split by identity into fit/select; official test unchanged"
    sets = [set(v["caption_pids"]) | set(v["image_pids"]) for v in splits.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Identity leakage across fit/select/test")
    return splits, protocol


def extract(model, data, args):
    from datasets.bases import ImageDataset, TextDataset
    from datasets.build import build_transforms
    output = {}
    for modality in ("text", "image"):
        selected = modality == args.modality
        ds = (TextDataset(data["caption_pids"], data["captions"], text_length=model.context_length) if modality == "text" else
              ImageDataset(data["image_pids"], data["img_paths"], build_transforms(tuple(args.image_size), is_train=False)))
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=str(args.device).startswith("cuda"))
        chunks, captured, handles = {}, {}, []
        tokens = None
        def hook(index):
            def capture(module, inputs, outputs):
                x = outputs[0].permute(1, 0, 2).float()
                # Same non-learned per-token LayerNorm at every layer, then pooling.
                ng, nh = pool_tokens(F.layer_norm(x, (x.shape[-1],)), modality, tokens)
                captured[f"global_{index}"] = normalize(ng).cpu().half()
                captured[f"hidden_{index}"] = normalize(nh).cpu().half()
                # Applying the final head at intermediate layers measures alignment only.
                ln = model.ln_final if modality == "text" else model.visual.ln_post
                proj = model.text_projection if modality == "text" else model.visual.proj
                pg, ph = pool_tokens(ln(x) @ proj.float(), modality, tokens)
                captured[f"direct_global_{index}"] = normalize(pg).cpu().half()
                captured[f"direct_hidden_{index}"] = normalize(ph).cpu().half()
            return capture
        if selected:
            blocks = model.transformer.resblocks if modality == "text" else model.visual.transformer.resblocks
            handles = [b.register_forward_hook(hook(i + 1)) for i, b in enumerate(blocks)]
        try:
            with torch.inference_mode():
                for batch_index, (_, batch) in enumerate(loader):
                    batch = batch.to(args.device)
                    tokens = batch if modality == "text" else None
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(args.device).startswith("cuda")):
                        seq, _ = model.encode_text(batch) if modality == "text" else model.encode_image(batch)
                    global_feature = seq[torch.arange(len(batch), device=batch.device), batch.argmax(-1)] if modality == "text" else seq[:, 0]
                    captured["baseline"] = normalize(global_feature).cpu().half()
                    for k, v in captured.items():
                        chunks.setdefault(k, []).append(v)
                    if batch_index % 50 == 0:
                        print(f"extract {modality}: {batch_index + 1}/{len(loader)}", flush=True)
        finally:
            for h in handles:
                h.remove()
        output[modality] = {k: torch.cat(v) for k, v in chunks.items()}
        output[modality]["ids"] = torch.tensor(data["caption_pids" if modality == "text" else "image_pids"])
        if selected:
            final = output[modality][f"direct_global_{len(blocks)}"]
            fidelity = (final.float() - output[modality]["baseline"].float()).abs().max().item()
            if fidelity > .005:
                raise RuntimeError(f"Final head fidelity failed: {fidelity}")
            print(f"final global fidelity max_abs={fidelity:.6g}", flush=True)
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=["text", "image"], required=True)
    p.add_argument("--dataset", choices=["rstp", "cuhk", "icfg"], default="rstp")
    p.add_argument("--root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--image-size", type=int, nargs=2, default=[384, 128])
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--mix-seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--penalties", type=float, nargs="+", default=[.0001, .001, .01])
    p.add_argument("--mix-steps", type=int, default=300)
    p.add_argument("--allow-test-holdout", action="store_true")
    args = p.parse_args()
    if any(x <= 0 for x in args.penalties):
        p.error("penalties must be positive")
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    splits, protocol = dataset_and_splits(args)
    manifest = {"args": vars(args), "protocol": protocol, "checkpoint_sha256": digest(args.checkpoint),
                "torch": torch.__version__, "numpy": np.__version__,
                "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "identities": {k: sorted(set(v["caption_pids"])) for k, v in splits.items()},
                "annotations_sha256": {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest() for k, v in splits.items()}}
    save_json(out / "manifest.json", manifest)
    model = load_clip(args)
    features = {}
    for split in ("fit", "select"):
        print(f"Extract {split}", flush=True)
        features[split] = extract(model, splits[split], args)
        torch.save(features[split], out / f"{split}_features.pt")
    own, other = args.modality, "image" if args.modality == "text" else "text"
    def evaluate(prediction, split):
        f = features[split]
        q, g = (prediction, f[other]["baseline"]) if own == "text" else (f[other]["baseline"], prediction)
        return retrieval(q, g, f["text"]["ids"], f["image"]["ids"], args.device)
    fit = features["fit"][own]
    selection = features["select"][own]
    y = targets(fit["ids"], features["fit"][other]["baseline"].float(), features["fit"][other]["ids"])
    grid, candidates, weights = [], {}, {}
    layers = sorted(int(k.split('_')[-1]) for k in fit if k.startswith("hidden_"))
    def record(name, pred, spec):
        row = {"name": name, **metrics(evaluate(pred, "select"))}
        grid.append(row)
        candidates[name] = spec
        print(row, flush=True)
    record("baseline", selection["baseline"], {"kind": "direct", "key": "baseline"})
    for pool in ("global", "hidden"):
        for layer in layers:
            key = f"{pool}_{layer}"
            direct = f"direct_{key}"
            record(direct, selection[direct], {"kind": "direct", "key": direct})
            for penalty in args.penalties:
                name = f"ridge_{key}_{penalty}"
                w = ridge(normalize(fit[key]), y, fit["ids"], penalty)
                weights[name] = w
                record(name, normalize(selection[key]) @ w, {"kind": "ridge", "keys": [key], "weight": name})
        # Equal-capacity average and learned scalar mixing of raw layer features.
        keys = [f"{pool}_{l}" for l in layers]
        xf = torch.stack([normalize(fit[k]) for k in keys], 1)
        xs = torch.stack([normalize(selection[k]) for k in keys], 1)
        for penalty in args.penalties:
            w = ridge(normalize(xf.mean(1)), y, fit["ids"], penalty)
            name = f"average_{pool}_{penalty}"
            weights[name] = w
            record(name, normalize(xs.mean(1)) @ w, {"kind": "ridge", "keys": keys, "weight": name})
        # Optimize alpha and a shared linear projection on FIT only.
        for seed in args.mix_seeds:
            torch.manual_seed(seed)
            x = xf.to(args.device)
            target = y.to(args.device)
            alpha = torch.nn.Parameter(torch.randn(len(keys), device=args.device) * .01)
            w = torch.nn.Parameter(ridge(normalize(xf.mean(1)), y, fit["ids"], .001).to(args.device))
            optimizer = torch.optim.Adam([alpha, w], lr=.003)
            _, inv, counts = fit["ids"].unique(return_inverse=True, return_counts=True)
            balance = (1. / counts[inv].float()).to(args.device)
            balance /= balance.sum()
            for step in range(args.mix_steps):
                pred = normalize((x * alpha.softmax(0)[None, :, None]).sum(1)) @ w
                loss = (((pred - target) ** 2).sum(1) * balance).sum() + .001 * w.square().sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            name = f"mix_{pool}_seed{seed}"
            a = alpha.detach().softmax(0).cpu()
            weights[name] = w.detach().cpu()
            record(name, normalize((xs * a[None, :, None]).sum(1)) @ weights[name],
                   {"kind": "ridge", "keys": keys, "weight": name, "alpha": a.tolist()})
            del x, target, optimizer, w, alpha
    best = lambda rows: max(rows, key=lambda r: (r["r1"], r["map"]))["name"]
    # Fixed families; no test-based choice. Baseline is always an eligible no-op.
    chosen = {"overall": best(grid), "baseline": "baseline"}
    for pool in ("global", "hidden"):
        for family in ("direct", "ridge", "average", "mix"):
            rows = [r for r in grid if r["name"].startswith(f"{family}_{pool}_")]
            chosen[f"{family}_{pool}"] = best(rows)
        chosen[f"ridge_final_{pool}"] = best([r for r in grid if r["name"].startswith(f"ridge_{pool}_{layers[-1]}_")])
    # Test whether hidden information complements the original global embedding.
    hidden_name = chosen["ridge_hidden"]
    hidden_spec = candidates[hidden_name]
    hidden_pred = normalize(normalize(selection[hidden_spec["keys"][0]]) @ weights[hidden_name])
    fusion_rows = []
    for beta in (0., .25, .5, .75, 1.):
        name = f"fusion_hidden_beta{beta}"
        pred = (1 - beta) * normalize(selection["baseline"]) + beta * hidden_pred
        record(name, pred, {"kind": "fusion", "source": hidden_name, "beta": beta})
        fusion_rows.append(grid[-1])
    chosen["fusion_hidden"] = best(fusion_rows)
    chosen["overall"] = best(grid)
    # Include fusion dependencies for deployment without rerunning selection.
    chosen["ridge_hidden"] = hidden_name
    with (out / "validation_grid.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid[0])); writer.writeheader(); writer.writerows(grid)
    save_json(out / "selection.json", {"chosen": chosen, "specs": {n: candidates[n] for n in set(chosen.values())}})
    torch.save(weights, out / "adapters.pt")
    # Test extraction/evaluation starts only after choices are serialized.
    features["test"] = extract(model, splits["test"], args)
    torch.save(features["test"], out / "test_features.pt")
    test = features["test"][own]
    baseline = evaluate(test["baseline"], "test")
    results = []
    for name in dict.fromkeys(chosen.values()):
        spec = candidates[name]
        if spec["kind"] == "fusion":
            source = candidates[spec["source"]]
            hidden = normalize(normalize(test[source["keys"][0]]) @ weights[source["weight"]])
            pred = (1 - spec["beta"]) * normalize(test["baseline"]) + spec["beta"] * hidden
        elif spec["kind"] == "direct":
            pred = test[spec["key"]]
        else:
            x = torch.stack([normalize(test[k]) for k in spec["keys"]], 1)
            a = torch.tensor(spec.get("alpha", [1 / len(spec["keys"])] * len(spec["keys"])))
            pred = normalize((x * a[None, :, None]).sum(1)) @ weights[spec["weight"]]
        rows = evaluate(pred, "test")
        np.save(out / f"queries_{name}.npy", rows.numpy())
        results.append({"name": name, **metrics(rows), "delta_vs_baseline_ci95": paired_ci(rows, baseline, features["test"]["text"]["ids"])})
    save_json(out / "test_results.json", results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
