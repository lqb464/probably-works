"""Additive, frozen full-gallery retrieval; no oracle positive injection.

Local MaxSim reuses the existing FILIP-inspired scorer, not a newly trained model.
Fusion is a convex score-level interpolation, with alpha selected off test.
Projected EOS/CLS are independently extracted as a global-equivalence control.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scorers import build_specs, score_pairs_all
from uncertainty import IdentityBootstrap, decision_intervals
from protocol_v2 import partition_ids
from validated import build_eval_split_loaders, write_csv, write_json


@torch.inference_mode()
def special_tokens(model, txt_loader, img_loader, device, official):
    """Read base encoder projected EOS/CLS, before any wrapper pooling/head."""
    text, image, qids, gids = [], [], [], []
    model.eval()
    for pid, tokens in txt_loader:
        tokens = tokens.to(device).long()
        seq, _ = model.base_model.encode_text(tokens)
        text.append(seq[torch.arange(len(tokens), device=device), tokens.argmax(-1)].float().cpu())
        qids.append(pid.reshape(-1).cpu())
    for pid, pixels in img_loader:
        seq, _ = model.base_model.encode_image(pixels.to(device))
        image.append(seq[:, 0].float().cpu())
        gids.append(pid.reshape(-1).cpu())
    if not torch.equal(torch.cat(qids), official.qids) or not torch.equal(torch.cat(gids), official.gids):
        raise ValueError("Special-token extraction order differs from official embeddings")
    q, g = F.normalize(torch.cat(text), dim=-1), F.normalize(torch.cat(image), dim=-1)
    oq, og = F.normalize(official.qfeats.float(), dim=-1), F.normalize(official.gfeats.float(), dim=-1)
    audit = {
        "definition": "cosine of final post-LayerNorm, post-projection text EOS and image CLS",
        "text_preprojection_dim": int(model.base_model.text_projection.shape[0]),
        "image_preprojection_dim": int(model.base_model.visual.proj.shape[0]),
        "shared_projected_dim": int(q.shape[1]),
        "raw_preprojection_cosine": "not computed: distinct modality spaces; dimensions may differ",
        "text_max_abs_difference_from_global": float((q-oq).abs().max()),
        "image_max_abs_difference_from_global": float((g-og).abs().max()),
        "equivalent_to_global_at_1e-5": bool(torch.allclose(q, oq, atol=1e-5, rtol=1e-5)
                                               and torch.allclose(g, og, atol=1e-5, rtol=1e-5)),
    }
    return q, g, audit


def split_indices(validation, test, cfg, seed):
    """Reuse v2's exact identity split seeds, retaining the complete source galleries."""
    if validation is None:
        if cfg.get("missing_validation") != "test_identity_holdout":
            raise ValueError("Missing validation: explicitly enable test_identity_holdout")
        fraction = float(cfg.get("holdout_development_fraction", .2))
        dev, evaluation = partition_ids(test.qids.numpy(), [fraction, 1-fraction], seed)
        selection_source = test
        source = "official_test_identity_holdout; fixed official test gallery"
    else:
        if np.intersect1d(validation.qids.numpy(), test.qids.numpy()).size:
            raise ValueError("Validation and test identities overlap")
        dev = np.unique(validation.qids.numpy())
        evaluation = np.unique(test.qids.numpy())
        selection_source = validation
        source = "official validation selection; official test; fixed source galleries"
    fit, selection, calibration = partition_ids(dev, cfg.get("development_fractions", [.5,.25,.25]), seed+10)
    rows = [{"split": name, "pid": int(pid)} for name, ids in
            [("fit_unused", fit), ("selection", selection), ("calibration_unused", calibration),
             ("test", evaluation)] for pid in ids]
    return (np.flatnonzero(np.isin(selection_source.qids.numpy(), selection)),
            np.flatnonzero(np.isin(test.qids.numpy(), evaluation)), rows, source)


def rank_scores(global_score, local_score, alpha, k=None):
    """Stable ties preserve global order; outside top-K is never changed."""
    base = np.argsort(-global_score, kind="stable")
    if alpha == 0:
        return base
    candidates = base if k is None else base[:k]
    fused = (1-alpha)*global_score[candidates] + alpha*local_score[candidates]
    order = candidates[np.argsort(-fused, kind="stable")]
    if k is None:
        return order
    result = base.copy()
    result[:len(order)] = order
    return result


def one_metrics(order, qid, gids):
    match = gids[order] == qid
    hits = np.flatnonzero(match)
    if not len(hits):
        raise ValueError(f"Query identity {qid} has no positive in original gallery")
    return [float(match[:k].any()) for k in [1,5,10]] + [
        float(np.mean(np.arange(1,len(hits)+1)/(hits+1))), float(len(hits)/(hits[-1]+1))]


def choices_for(names, alphas, ks):
    choices = [{"key": "global", "scorer": "global", "family": "global", "alpha": 0., "k": None},
               {"key": "eos_cls_projected", "scorer": "eos_cls_projected",
                "family": "special_token_control", "alpha": 1., "k": None}]
    for name in names:
        choices.append({"key": name+"__hidden_full", "scorer": name,
                        "family": "hidden_full", "alpha": 1., "k": None})
        for family, limits in [("fusion_full", [None]), ("rerank_topk", ks)]:
            for k in limits:
                for alpha in alphas:
                    choices.append({"key": f"{name}__{family}__k{k}__a{alpha:g}",
                                    "scorer": name, "family": family, "alpha": alpha, "k": k})
    return choices


def select_choices(choices, metrics):
    """Select within each predeclared scorer/family, using selection R1 only."""
    base = metrics["global"][:,0].astype(bool)
    chosen = [c for c in choices if c["family"] in {"global", "special_token_control", "hidden_full"}]
    groups = sorted({(c["scorer"], c["family"]) for c in choices
                     if c["family"] in {"fusion_full", "rerank_topk"}})
    for scorer, family in groups:
        group = [c for c in choices if c["scorer"] == scorer and c["family"] == family]
        def key(c):
            new = metrics[c["key"]][:,0].astype(bool)
            return (int(new.sum())-int(base.sum()), -int((base & ~new).sum()),
                    c["alpha"] == 0, -c["alpha"], -(c["k"] or 0))
        chosen.append(max(group, key=key))
    return chosen


@torch.inference_mode()
def score_split(gf, hf, special, indices, specs, choices, device, batch_size, output, split, save=False):
    """Stream one query through ALL gallery images. GPU memory bounded by pair batch."""
    n, ng = len(indices), len(gf.gids)
    if n == 0 or batch_size <= 0:
        raise ValueError("Empty split or invalid pair batch size")
    qglobal, gglobal = F.normalize(gf.qfeats.float(), dim=-1), F.normalize(gf.gfeats.float(), dim=-1)
    gids = gf.gids.numpy()
    metrics = {c["key"]: np.empty((n,5), dtype=np.float64) for c in choices}
    top1 = {c["key"]: np.empty(n, dtype=np.int64) for c in choices}
    maps = {}
    if save:
        for name in ["global", "eos_cls_projected"] + [s.name for s in specs]:
            maps[name] = np.lib.format.open_memmap(output/f"{split}__{name}__similarity.npy",
                                                  mode="w+", dtype="float32", shape=(n,ng))
    np.savez_compressed(output/f"{split}__index.npz", query_index=indices,
                        qids=gf.qids[indices].numpy(), gids=gids)
    try:
        for row, qi in enumerate(indices):
            scores = {"global": (qglobal[qi] @ gglobal.T).numpy(),
                      "eos_cls_projected": (special[0][qi] @ special[1].T).numpy()}
            local = score_pairs_all(specs, hf.text_features, hf.text_mask, hf.image_features,
                                   torch.full((ng,), int(qi), dtype=torch.long), torch.arange(ng),
                                   device, batch_size, gf.qfeats, gf.gfeats)
            scores.update({name: values.numpy() for name, values in local.items()})
            if any(not np.isfinite(s).all() for s in scores.values()):
                raise ValueError(f"Nonfinite similarity in {split} query {qi}")
            for name, mm in maps.items():
                mm[row] = scores[name]
            for c in choices:
                order = rank_scores(scores["global"], scores[c["scorer"]], c["alpha"], c["k"])
                metrics[c["key"]][row] = one_metrics(order, int(gf.qids[qi]), gids)
                top1[c["key"]][row] = order[0]
            if row % 25 == 0 or row+1 == n:
                logging.info("Full gallery %s: %d/%d queries, %d images/query", split, row+1, n, ng)
    finally:
        for mm in maps.values():
            mm.flush()
    return metrics, top1


def report(choices, metrics, pids, cfg):
    rows = []
    base = metrics["global"]
    seed = int(cfg.get("seed",42))
    reps = int(cfg.get("bootstrap_repetitions",2000))
    confidence = float(cfg.get("bootstrap_confidence",.95))
    boot = IdentityBootstrap(pids, reps, confidence, seed)
    for c in choices:
        values = metrics[c["key"]]
        row = {**c, **decision_intervals(pids, base[:,0], values[:,0], reps, confidence, seed)}
        for i, name in enumerate(["r1","r5","r10","map","minp"]):
            row.update(boot.report(name, values[:,i], scale=100))
            row.update(boot.report("delta_"+name, values[:,i]-base[:,i], scale=100))
        rows.append(row)
    return rows


def run_addon(config, run_dir, model, device, repo_args, gf, hf, txt_loader, img_loader, cache_path, cache_meta,
              prepared_validation=None):
    from run_experiments import load_or_extract_features, git_commit, sha256_file
    output = run_dir/"full_gallery"
    output.mkdir()
    cfg = dict(config.get("validated") or {})
    cfg["seed"] = int(config.get("seed",42))
    alphas = sorted(set([0.,1.] + [float(x) for x in config.get("fusion_weights", [.1,.2,.3,.5])]))
    if any(not 0 <= a <= 1 for a in alphas):
        raise ValueError("Fusion alpha must be within [0,1]")
    ks = sorted(set(int(k) for k in config.get("topk",[10,20,50])))
    if not ks or min(ks) < 1:
        raise ValueError("topk must be positive")
    # Two predeclared pure-local readouts; original nine-scorer v2 remains unchanged.
    specs = build_specs([{"name":"maxsim", "kind":"maxsim"},
                         {"name":"uniform_mean_control", "kind":"uniform_mean"}])
    special_test = special_tokens(model, txt_loader, img_loader, device, gf)
    vi, vt = build_eval_split_loaders(repo_args, "val", allow_missing=cfg.get("missing_validation")=="test_identity_holdout")
    vg = vh = special_val = None
    if vi is not None:
        val_meta = {**cache_meta, "split":"val"}
        if prepared_validation is not None:
            vg, vh = prepared_validation
        else:
            vg, vh, _ = load_or_extract_features(model, device, vt, vi, str(config.get("feature_dtype","float16")),
                                             cache_path.parent/"validation_features.pt", val_meta,
                                             bool(config.get("cache_features",True)), bool(config.get("force_recompute",False)))
        special_val = special_tokens(model, vt, vi, device, vg)
    si, ti, assignments, source = split_indices(vg, gf, cfg, cfg["seed"])
    write_csv(output/"split_assignments.csv", assignments)
    write_json(output/"special_token_audit.json", {"test":special_test[2], "validation":special_val[2] if special_val else None})
    choices = choices_for([s.name for s in specs], alphas, ks)
    protocol = {"source":source, "gallery":"all original gallery images; no positive injection",
                "fusion":"(1-alpha)*global_cosine + alpha*hidden_score; no per-query zscore",
                "selection":"R1, then lower harm, then no-op; per predeclared scorer/family",
                "hidden":"final projected content tokens and image patches; excludes SOS/EOS/CLS/padding",
                "bootstrap":cfg, "test_status":"exploratory; previous test results inspected",
                "git_commit":git_commit(), "implementation_sha256":sha256_file(Path(__file__)),
                "checkpoint_sha256":cache_meta["checkpoint_sha256"],
                "selection_queries":len(si), "test_queries":len(ti), "test_gallery":len(gf.gids),
                "similarities_saved":bool(config.get("save_full_similarity",False))}
    write_json(output/"protocol.json", protocol)
    args = (device, int(config.get("pair_batch_size",384)), output)
    selection, _ = score_split(vg if vg is not None else gf, vh if vh is not None else hf,
                               special_val if special_val else special_test, si, specs, choices, *args, "selection")
    grid = [{**c, "selection_r1":float(selection[c["key"]][:,0].mean()*100)} for c in choices]
    write_csv(output/"selection_grid.csv",grid)
    chosen = select_choices(choices,selection)
    write_json(output/"selected_before_test.json",chosen)
    metrics, top1 = score_split(gf,hf,special_test,ti,specs,chosen,*args,"test",
                               bool(config.get("save_full_similarity",False)))
    rows = report(chosen,metrics,gf.qids[ti].numpy(),cfg)
    write_csv(output/"test_summary.csv",rows)
    np.savez_compressed(output/"test_query_metrics.npz",pid=gf.qids[ti].numpy(),
                        **{c["key"]+"/"+name: metrics[c["key"]][:,i] for c in chosen
                           for i,name in enumerate(["r1","r5","r10","map","minp"])})
    write_csv(output/"test_top1.csv", [{"query_index":int(qi), "query_pid":int(gf.qids[qi]),
              "method":c["key"], "gallery_index":int(top1[c["key"]][j]),
              "gallery_pid":int(gf.gids[top1[c["key"]][j]])} for j,qi in enumerate(ti) for c in chosen])
    write_json(output/"summary.json", {"protocol":protocol,"results":rows})
    logging.info("Full-gallery addon complete: %s",output/"test_summary.csv")
    return output/"test_summary.csv"
