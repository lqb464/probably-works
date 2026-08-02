from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Sequence

import numpy as np


def grouped_bootstrap_ci(
    metric_fn: Callable[[np.ndarray], float],
    query_pids: Sequence[int],
    repetitions: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> list:
    pids = np.asarray(query_pids)
    unique = np.unique(pids)
    if unique.size == 0:
        return [0.0, 0.0]

    groups = {pid: np.flatnonzero(pids == pid) for pid in unique}
    rng = np.random.default_rng(seed)
    values = []

    for _ in range(int(repetitions)):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([groups[pid] for pid in sampled])
        value = metric_fn(indices)
        if not np.isnan(value):
            values.append(float(value))

    if not values:
        return [0.0, 0.0]

    alpha = 1.0 - float(confidence)
    low = np.percentile(values, 100.0 * alpha / 2.0)
    high = np.percentile(values, 100.0 * (1.0 - alpha / 2.0))
    return [float(low), float(high)]


def fixed_pair_ci_metrics(
    query_pids: Sequence[int],
    delta_global: np.ndarray,
    delta_hidden: np.ndarray,
    repetitions: int,
    confidence: float,
    seed: int,
) -> Dict[str, list]:
    delta_global = np.asarray(delta_global)
    delta_hidden = np.asarray(delta_hidden)

    def recovery(indices: np.ndarray) -> float:
        gfail = delta_global[indices] < 0
        den = gfail.sum()
        if den == 0:
            return np.nan
        return float(((delta_hidden[indices] > 0) & gfail).sum() / den)

    def harm(indices: np.ndarray) -> float:
        gcorrect = delta_global[indices] > 0
        den = gcorrect.sum()
        if den == 0:
            return np.nan
        return float(((delta_hidden[indices] < 0) & gcorrect).sum() / den)

    def complementarity_gap(indices: np.ndarray) -> float:
        g = delta_global[indices] > 0
        h = delta_hidden[indices] > 0
        oracle = g | h
        return float(oracle.mean() - max(g.mean(), h.mean()))

    return {
        "recovery_rate_ci95": grouped_bootstrap_ci(
            recovery, query_pids, repetitions, confidence, seed
        ),
        "harm_rate_ci95": grouped_bootstrap_ci(
            harm, query_pids, repetitions, confidence, seed + 1
        ),
        "complementarity_gap_ci95": grouped_bootstrap_ci(
            complementarity_gap, query_pids, repetitions, confidence, seed + 2
        ),
    }


def rescue_ci(
    query_pids: Sequence[int],
    delta_global: np.ndarray,
    positive_in_topk: np.ndarray,
    hidden_topk_correct: np.ndarray,
    repetitions: int,
    confidence: float,
    seed: int,
) -> list:
    delta_global = np.asarray(delta_global)
    positive_in_topk = np.asarray(positive_in_topk, dtype=bool)
    hidden_topk_correct = np.asarray(hidden_topk_correct, dtype=bool)

    def metric(indices: np.ndarray) -> float:
        denom = (delta_global[indices] < 0) & positive_in_topk[indices]
        den = denom.sum()
        if den == 0:
            return np.nan
        return float((denom & hidden_topk_correct[indices]).sum() / den)

    return grouped_bootstrap_ci(metric, query_pids, repetitions, confidence, seed)


def write_combined_summary(summary_rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "model_preset",
        "global_r1",
        "global_r5",
        "global_r10",
        "global_map",
        "num_global_failures",
        "recovery_rate",
        "recovery_ci_low",
        "recovery_ci_high",
        "harm_rate",
        "positive_in_top50_given_global_failure",
        "rescue_at_50",
        "hidden_correct_rate",
        "oracle_correct_rate",
        "complementarity_gap",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summary_rows:
            fixed = summary["fixed_pair"]
            top50 = summary["topk"].get("50", {})
            oracle = summary["oracle"]
            recovery_ci = fixed.get("recovery_rate_ci95", [0.0, 0.0])
            writer.writerow({
                "dataset": summary["dataset"],
                "model_preset": summary.get("model_preset", ""),
                "global_r1": summary["official_global"]["r1"],
                "global_r5": summary["official_global"]["r5"],
                "global_r10": summary["official_global"]["r10"],
                "global_map": summary["official_global"]["map"],
                "num_global_failures": fixed["global_failures"],
                "recovery_rate": fixed["recovery_rate"],
                "recovery_ci_low": recovery_ci[0],
                "recovery_ci_high": recovery_ci[1],
                "harm_rate": fixed["harm_rate"],
                "positive_in_top50_given_global_failure": top50.get(
                    "positive_in_topk_given_global_failure", 0.0
                ),
                "rescue_at_50": top50.get("rescue_at_k", 0.0),
                "hidden_correct_rate": oracle["hidden_correct_rate"],
                "oracle_correct_rate": oracle["oracle_correct_rate"],
                "complementarity_gap": oracle["complementarity_gap"],
            })
