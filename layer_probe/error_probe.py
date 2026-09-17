"""Paper-inspired correctness probe and selective hidden reranking.

This is the CLIP analogue of the error detector in arXiv:2507.12379.  The
frozen global CLIP retrieval decision is treated as the model's current
"answer".  A small logistic probe predicts whether its top-1 image is
correct.  Only low-confidence queries are sent to the frozen hidden-layer
reranker.  No test labels are used for fitting or threshold selection.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layer_probe.core import normalize


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def balanced_weights(ids):
    ids = np.asarray(ids)
    unique, counts = np.unique(ids, return_counts=True)
    per_id = {u: 1.0 / c for u, c in zip(unique, counts)}
    w = np.asarray([per_id[x] for x in ids], dtype=np.float64)
    return w / w.mean()


def make_split(cache, modality, hidden_key, weight, topk, seed, permute=False):
    """Make correctness labels/features for the fixed text-to-image task."""
    text = cache["text"]
    image = cache["image"]
    text_global = normalize(text["baseline"].float())
    image_global = normalize(image["baseline"].float())
    global_scores = text_global @ image_global.T
    global_order = torch.argsort(global_scores, dim=1, descending=True)
    global_top1 = global_order[:, 0]
    top = global_order[:, : min(topk, global_order.shape[1])]
    global_score = global_scores.gather(1, global_top1[:, None])[:, 0]
    global_margin = global_scores.gather(1, global_order[:, :2])
    global_margin = global_margin[:, 0] - global_margin[:, 1]

    raw_hidden = (text if modality == "text" else image)[hidden_key].float()
    if permute:
        generator = torch.Generator().manual_seed(seed)
        raw_hidden = raw_hidden[torch.randperm(raw_hidden.shape[0], generator=generator)]

    hidden_space = normalize(normalize(raw_hidden) @ weight.float())
    if modality == "text":
        hidden_scores = hidden_space @ image_global.T
        probe_vector = hidden_space
    else:
        hidden_scores = text_global @ hidden_space.T
        # For image probing, the candidate image is the only selected-layer
        # object available for the current query.  Use the global top-1
        # candidate's projected hidden vector as the candidate-conditioned
        # probe input.
        probe_vector = hidden_space[global_top1]

    hidden_top = hidden_scores.gather(1, top)
    hidden_sorted = torch.sort(hidden_top, dim=1, descending=True).values
    hidden_best_pos = hidden_top.argmax(dim=1)
    hidden_choice = top.gather(1, hidden_best_pos[:, None])[:, 0]
    hidden_score = hidden_scores.gather(1, global_top1[:, None])[:, 0]
    hidden_margin = hidden_sorted[:, 0] - hidden_sorted[:, 1] if top.shape[1] > 1 else torch.zeros_like(hidden_score)

    global_correct = (np.asarray(image["ids"])[global_top1.numpy()] == np.asarray(text["ids"]))
    hidden_correct = (np.asarray(image["ids"])[hidden_choice.numpy()] == np.asarray(text["ids"]))
    global_score = global_score.numpy()
    global_margin = global_margin.numpy()
    hidden_score = hidden_score.numpy()
    hidden_margin = hidden_margin.numpy()
    probe_vector = probe_vector.numpy()

    scalars = np.column_stack([
        global_score,
        global_margin,
        hidden_score,
        hidden_margin,
        hidden_score - global_score,
    ]).astype(np.float32)
    global_scalars = scalars[:, :2]
    hidden_scalars = scalars[:, 2:]
    features = {
        "global_only": global_scalars,
        "hidden_only": np.column_stack([probe_vector, hidden_scalars]).astype(np.float32),
        "global_plus_hidden": np.column_stack([global_scalars, probe_vector, hidden_scalars]).astype(np.float32),
    }
    return {
        "features": features,
        "labels": global_correct.astype(np.int64),
        "global_top1": global_top1.numpy(),
        "hidden_choice": hidden_choice.numpy(),
        "global_correct": global_correct,
        "hidden_correct": hidden_correct,
        "text_ids": np.asarray(text["ids"]),
        "image_ids": np.asarray(image["ids"]),
    }


def fit_variant(train, select, variant):
    best = None
    for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=500, solver="lbfgs"),
        )
        model.fit(train["features"][variant], train["labels"],
                  logisticregression__sample_weight=balanced_weights(train["text_ids"]))
        p = model.predict_proba(select["features"][variant])[:, 1]
        loss = log_loss(select["labels"], p, labels=[0, 1])
        if best is None or loss < best[0]:
            best = (loss, c, model)
    return best


def probe_metrics(y, p):
    return {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "accuracy_at_0.5": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def gate_result(data, error_score, threshold):
    gate = error_score >= threshold
    final = data["global_top1"].copy()
    final[gate] = data["hidden_choice"][gate]
    base = data["global_correct"]
    corrected = np.asarray(data["image_ids"])[final] == data["text_ids"]
    rescued = (~base) & corrected
    harmed = base & (~corrected)
    return {
        "threshold_error_score": float(threshold),
        "gated_queries": int(gate.sum()),
        "gate_rate": float(gate.mean()),
        "baseline_r1": float(base.mean()),
        "selective_r1": float(corrected.mean()),
        "delta_r1": float(corrected.mean() - base.mean()),
        "rescued": int(rescued.sum()),
        "harmed": int(harmed.sum()),
        "harm_rate_among_baseline_correct": float(harmed.sum() / max(1, base.sum())),
    }


def choose_threshold(data, p, harm_budget):
    error = 1.0 - p
    candidates = np.unique(np.quantile(error, np.linspace(0.0, 1.0, 101)))
    choices = []
    for threshold in candidates:
        result = gate_result(data, error, threshold)
        if result["harm_rate_among_baseline_correct"] <= harm_budget:
            choices.append(result)
    if not choices:
        return gate_result(data, error, float("inf"))
    return max(choices, key=lambda x: (x["delta_r1"], -x["harm_rate_among_baseline_correct"], -x["gate_rate"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--modality", choices=["text", "image"], required=True)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--harm-budget", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or args.run_dir / "paper_error_probe"
    out.mkdir(parents=True, exist_ok=False)

    selection = read_json(args.run_dir / "selection.json")
    hidden_name = selection["chosen"]["ridge_hidden"]
    hidden_key = selection["specs"][hidden_name]["keys"][0]
    weight_name = selection["specs"][hidden_name]["weight"]
    weights = load_pt(args.run_dir / "adapters.pt")
    weight = weights[weight_name]

    caches = {split: load_pt(args.run_dir / f"{split}_features.pt") for split in ("fit", "select", "test")}
    data = {
        split: make_split(caches[split], args.modality, hidden_key, weight, args.topk, 1701 + i)
        for i, split in enumerate(("fit", "select", "test"))
    }
    variants = ("global_only", "hidden_only", "global_plus_hidden")
    rows = []
    predictions = {}
    for variant in variants:
        select_data = data["select"]
        fit = fit_variant(data["fit"], select_data, variant)
        select_loss, c, model = fit
        p_select = model.predict_proba(select_data["features"][variant])[:, 1]
        p_test = model.predict_proba(data["test"]["features"][variant])[:, 1]
        predictions[variant] = (p_select, p_test)
        row = {"variant": variant, "C": c, "selection_log_loss": select_loss}
        row.update({f"test_{k}": v for k, v in probe_metrics(data["test"]["labels"], p_test).items()})
        rows.append(row)

    # The null keeps the exact classifier protocol but breaks the pairing
    # between hidden activation and query/candidate.  It is never selected.
    null_data = {}
    for split_i, split in enumerate(("fit", "select", "test")):
        null_data[split] = make_split(
            caches[split], args.modality, hidden_key, weight, args.topk,
            9001 + split_i, permute=True,
        )
    null_fit = fit_variant(null_data["fit"], null_data["select"], "global_plus_hidden")
    null_model = null_fit[2]
    null_p_test = null_model.predict_proba(null_data["test"]["features"]["global_plus_hidden"])[:, 1]
    rows.append({
        "variant": "permuted_hidden_control",
        "C": null_fit[1],
        "selection_log_loss": null_fit[0],
        **{f"test_{k}": v for k, v in probe_metrics(null_data["test"]["labels"], null_p_test).items()},
    })

    chosen_variant = "global_plus_hidden"
    p_select, p_test = predictions[chosen_variant]
    selected_gate = choose_threshold(data["select"], p_select, args.harm_budget)
    test_gate = gate_result(data["test"], 1.0 - p_test, selected_gate["threshold_error_score"])
    report = {
        "paper": "arXiv:2507.12379",
        "adaptation": "predict whether frozen global top-1 retrieval is correct; selectively rerank global top-k with hidden probe",
        "modality": args.modality,
        "hidden_name": hidden_name,
        "hidden_key": hidden_key,
        "topk": args.topk,
        "harm_budget": args.harm_budget,
        "chosen_variant": chosen_variant,
        "selection_gate": selected_gate,
        "test_gate": test_gate,
        "test_baseline_r1": float(data["test"]["global_correct"].mean()),
        "test_hidden_rerank_r1": float(data["test"]["hidden_correct"].mean()),
    }
    (out / "selective_correction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "probe_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        out / "probe_predictions.npz",
        y_test=data["test"]["labels"],
        p_global_only=predictions["global_only"][1],
        p_hidden_only=predictions["hidden_only"][1],
        p_global_plus_hidden=p_test,
        p_permuted_hidden_control=null_p_test,
    )
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
