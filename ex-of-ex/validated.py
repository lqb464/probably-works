"""Validation-selected experiments for hidden-token person retrieval.

The four experiment families intentionally answer different questions:

* setwise: an oracle representation diagnostic over all positives and hard negatives;
* probe: a low-capacity held-out-identity test of incremental hidden information;
* reranking: a deployable global top-K reranker selected on validation;
* gating: a validation-selected selective reranker under a harm budget.

This module never fine-tunes the image/text encoder.  Candidate pools are built
once from the global baseline and shared by every hidden-token scorer.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from uncertainty import grouped_bootstrap_ci
from diagnostic.global_hidden_recovery.evaluator import compute_global_similarity
from diagnostic.global_hidden_recovery.feature_extractor import GlobalFeatureSet, HiddenFeatureSet
from evaluation import candidate_score_variants, metrics_from_rankings, positive_in_prefix
from scorers import ScorerSpec, score_pairs_all


@dataclass
class SplitScores:
    name: str
    global_features: GlobalFeatureSet
    hidden_features: HiddenFeatureSet
    similarity: torch.Tensor
    rankings: torch.Tensor
    topk_indices: torch.Tensor
    global_topk_scores: torch.Tensor
    local_topk_scores: Dict[str, torch.Tensor]


@dataclass
class OracleSetScores:
    qids: torch.Tensor
    gids: torch.Tensor
    global_correct: torch.Tensor
    positive_indices: torch.Tensor
    positive_offsets: list[int]
    negative_indices: torch.Tensor
    positive_global_scores: torch.Tensor
    negative_global_scores: torch.Tensor
    positive_local_scores: Dict[str, torch.Tensor]
    negative_local_scores: Dict[str, torch.Tensor]


def build_eval_split_loaders(repo_args, split: str, allow_missing: bool = False):
    """Build deterministic val/test loaders without instantiating a train sampler."""
    from datasets.bases import ImageDataset, TextDataset
    from datasets.build import build_transforms
    from datasets.cuhkpedes import CUHKPEDES
    from datasets.icfgpedes import ICFGPEDES
    from datasets.rstpreid import RSTPReid

    factories = {
        "CUHK-PEDES": CUHKPEDES,
        "ICFG-PEDES": ICFGPEDES,
        "RSTPReid": RSTPReid,
    }
    if repo_args.dataset_name not in factories:
        raise ValueError(f"Unsupported validation dataset: {repo_args.dataset_name}")
    if split not in {"val", "test"}:
        raise ValueError(f"split must be val or test, got {split}")

    dataset = factories[repo_args.dataset_name](root=repo_args.root_dir)
    records = getattr(dataset, split)
    if not records["caption_pids"] or not records["image_pids"]:
        if allow_missing and split == "val":
            logging.getLogger("ex-of-ex.validated").warning(
                "No official validation records for %s; using explicit v2 identity-holdout protocol",
                repo_args.dataset_name,
            )
            return None, None
        raise ValueError(
            f"{repo_args.dataset_name} has no {split} records. Original diagnostic uses test only; "
            "validated experiments need a development split. See METHODS_V2.md."
        )
    transform = build_transforms(img_size=repo_args.img_size, is_train=False)
    image_set = ImageDataset(records["image_pids"], records["img_paths"], transform)
    text_set = TextDataset(
        records["caption_pids"], records["captions"], text_length=repo_args.text_length
    )
    batch_size = int(getattr(repo_args, "test_batch_size", repo_args.batch_size))
    common = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": int(repo_args.num_workers),
    }
    image_loader = DataLoader(image_set, **common)
    text_loader = DataLoader(text_set, **common)
    # The repository extractor reads captions through this compatibility attribute.
    image_loader.test_img_set = image_set
    text_loader.test_txt_set = text_set
    return image_loader, text_loader


def make_split_scores(
    name: str,
    global_features: GlobalFeatureSet,
    hidden_features: HiddenFeatureSet,
    specs: Sequence[ScorerSpec],
    topk_values: Sequence[int],
    device: torch.device,
    pair_batch_size: int,
    similarity: torch.Tensor | None = None,
    rankings: torch.Tensor | None = None,
    topk_indices: torch.Tensor | None = None,
    global_topk_scores: torch.Tensor | None = None,
    local_topk_scores: Mapping[str, torch.Tensor] | None = None,
) -> SplitScores:
    similarity = similarity if similarity is not None else compute_global_similarity(
        global_features.qfeats, global_features.gfeats
    )
    rankings = rankings if rankings is not None else torch.argsort(
        similarity, dim=1, descending=True
    ).cpu()
    max_k = min(max(int(k) for k in topk_values), similarity.shape[1])
    topk_indices = (
        topk_indices[:, :max_k].contiguous()
        if topk_indices is not None
        else rankings[:, :max_k].contiguous()
    )
    global_topk_scores = (
        global_topk_scores[:, :max_k].contiguous()
        if global_topk_scores is not None
        else similarity.gather(1, topk_indices)
    )
    if local_topk_scores is None:
        query_indices = torch.arange(global_features.qids.numel(), dtype=torch.long)
        flat_q = query_indices.repeat_interleave(max_k)
        flat_g = topk_indices.reshape(-1)
        scored = score_pairs_all(
            specs=specs,
            text_features=hidden_features.text_features,
            text_mask=hidden_features.text_mask,
            image_features=hidden_features.image_features,
            query_indices=flat_q,
            gallery_indices=flat_g,
            device=device,
            batch_size=pair_batch_size,
            query_global=global_features.qfeats,
            gallery_global=global_features.gfeats,
        )
        local_topk_scores = {
            scorer: values.view(global_features.qids.numel(), max_k)
            for scorer, values in scored.items()
        }
    else:
        local_topk_scores = {
            key: value[:, :max_k].contiguous() for key, value in local_topk_scores.items()
        }
    return SplitScores(
        name=name,
        global_features=global_features,
        hidden_features=hidden_features,
        similarity=similarity.cpu(),
        rankings=rankings.cpu(),
        topk_indices=topk_indices.cpu(),
        global_topk_scores=global_topk_scores.cpu(),
        local_topk_scores=dict(local_topk_scores),
    )


def run_validated_suite(
    config: Mapping[str, object],
    run_dir: Path,
    specs: Sequence[ScorerSpec],
    validation: SplitScores,
    test: SplitScores,
    device: torch.device,
    pair_batch_size: int,
    existing_test_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if int(dict(config.get("validated") or {}).get("protocol_version", 1)) == 2:
        from protocol_v2 import run_suite_v2
        return run_suite_v2(config, run_dir, specs, validation, test, device,
                            pair_batch_size, existing_test_rows)
    logger = logging.getLogger("ex-of-ex.validated")
    cfg = dict(config.get("validated") or {})
    output_dir = run_dir / "validated"
    output_dir.mkdir(parents=True, exist_ok=False)

    negative_counts = sorted({int(v) for v in cfg.get("setwise_negative_counts", [1, 5, 10])})
    if not negative_counts or min(negative_counts) <= 0:
        raise ValueError("validated.setwise_negative_counts must contain positive integers")
    max_negatives = max(max(negative_counts), int(cfg.get("probe_negatives", 10)))
    max_available = min(
        validation.global_features.gids.numel(), test.global_features.gids.numel()
    ) - 1
    if max_negatives > max_available:
        raise ValueError(
            f"Requested {max_negatives} hard negatives but a split has at most "
            f"{max_available} non-self candidates"
        )
    report_negatives = int(cfg.get("setwise_report_negatives", max(negative_counts)))
    if report_negatives not in negative_counts:
        raise ValueError("setwise_report_negatives must occur in setwise_negative_counts")

    logger.info("Validated suite: building shared oracle candidate packs")
    val_oracle = build_oracle_set_scores(
        validation, specs, max_negatives, device, pair_batch_size
    )
    test_oracle = build_oracle_set_scores(test, specs, max_negatives, device, pair_batch_size)

    bootstrap_repetitions = int(cfg.get("bootstrap_repetitions", 2000))
    confidence = float(cfg.get("bootstrap_confidence", 0.95))
    seed = int(config.get("seed", 42))
    setwise_summary, setwise_queries = evaluate_setwise(
        test_oracle,
        specs,
        negative_counts,
        report_negatives,
        bootstrap_repetitions,
        confidence,
        seed,
    )
    write_csv(output_dir / "setwise_summary.csv", setwise_summary)
    write_csv(output_dir / "setwise_query_results.csv", setwise_queries)

    logger.info("Validated suite: fitting low-capacity probes on validation identities")
    probe_rows = evaluate_probes(
        val_oracle,
        test_oracle,
        specs,
        num_negatives=min(int(cfg.get("probe_negatives", 10)), max_negatives),
        regularization_c=float(cfg.get("probe_regularization_c", 1.0)),
        seed=seed,
    )
    write_csv(output_dir / "probe_results.csv", probe_rows)

    topk_values = sorted({int(k) for k in config.get("topk", [10, 20, 50])})
    fusion_weights = [float(v) for v in config.get("fusion_weights", [0.1, 0.2, 0.3, 0.5])]
    coverage_rows = evaluate_coverage(validation, topk_values) + evaluate_coverage(
        test, topk_values
    )
    write_csv(output_dir / "coverage.csv", coverage_rows)

    logger.info("Validated suite: selecting top-K rerankers on validation")
    rerank_grid, selected, selected_test = evaluate_reranking(
        validation,
        test,
        specs,
        topk_values,
        fusion_weights,
        existing_test_rows,
    )
    write_csv(output_dir / "reranking_validation_grid.csv", rerank_grid)
    write_csv(output_dir / "reranking_selected_test.csv", selected_test)

    logger.info("Validated suite: selecting uncertainty gates under harm budgets")
    gating_rows = evaluate_gating(
        validation=validation,
        test=test,
        specs=specs,
        topk_values=topk_values,
        fusion_weights=fusion_weights,
        harm_budgets=[float(v) for v in cfg.get("gating_harm_budgets", [0.01, 0.02, 0.05])],
        threshold_steps=int(cfg.get("gating_threshold_steps", 41)),
    )
    write_csv(output_dir / "gating_summary.csv", gating_rows)

    summary = {
        "protocol": {
            "encoder": "frozen",
            "selection_split": "official validation identities",
            "evaluation_split": "official test identities",
            "candidate_policy": "global baseline candidates fixed before scorer comparison",
            "setwise_is_oracle": True,
            "reranking_injects_positive": False,
        },
        "setwise": setwise_summary,
        "probe": probe_rows,
        "reranking_selected": selected,
        "reranking_test": selected_test,
        "gating": gating_rows,
        "coverage": coverage_rows,
        "outputs": {
            "setwise_summary": str(output_dir / "setwise_summary.csv"),
            "setwise_queries": str(output_dir / "setwise_query_results.csv"),
            "probe_results": str(output_dir / "probe_results.csv"),
            "coverage": str(output_dir / "coverage.csv"),
            "reranking_validation_grid": str(output_dir / "reranking_validation_grid.csv"),
            "reranking_selected_test": str(output_dir / "reranking_selected_test.csv"),
            "gating_summary": str(output_dir / "gating_summary.csv"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    logger.info("Validated experiment outputs written to %s", output_dir)
    return summary


def build_oracle_set_scores(
    split: SplitScores,
    specs: Sequence[ScorerSpec],
    max_negatives: int,
    device: torch.device,
    pair_batch_size: int,
) -> OracleSetScores:
    qids = split.global_features.qids.cpu().long()
    gids = split.global_features.gids.cpu().long()
    pid_to_gallery: Dict[int, list[int]] = {}
    for gallery_index, pid in enumerate(gids.tolist()):
        pid_to_gallery.setdefault(int(pid), []).append(gallery_index)

    positive_indices: list[int] = []
    positive_queries: list[int] = []
    positive_offsets = [0]
    negative_rows: list[list[int]] = []
    gids_list = gids.tolist()
    for query_index, pid in enumerate(qids.tolist()):
        positives = pid_to_gallery.get(int(pid), [])
        if not positives:
            raise RuntimeError(f"Query {query_index} (pid={pid}) has no gallery positive")
        positive_indices.extend(positives)
        positive_queries.extend([query_index] * len(positives))
        positive_offsets.append(len(positive_indices))

        negatives = []
        for gallery_index in split.rankings[query_index].tolist():
            if gids_list[gallery_index] != pid:
                negatives.append(gallery_index)
                if len(negatives) == max_negatives:
                    break
        if len(negatives) != max_negatives:
            raise RuntimeError(f"Query {query_index} has only {len(negatives)} negatives")
        negative_rows.append(negatives)

    positive_index_tensor = torch.tensor(positive_indices, dtype=torch.long)
    positive_query_tensor = torch.tensor(positive_queries, dtype=torch.long)
    negative_index_tensor = torch.tensor(negative_rows, dtype=torch.long)
    negative_query_tensor = torch.arange(qids.numel(), dtype=torch.long).repeat_interleave(
        max_negatives
    )
    all_queries = torch.cat([positive_query_tensor, negative_query_tensor])
    all_gallery = torch.cat([positive_index_tensor, negative_index_tensor.reshape(-1)])
    scored = score_pairs_all(
        specs=specs,
        text_features=split.hidden_features.text_features,
        text_mask=split.hidden_features.text_mask,
        image_features=split.hidden_features.image_features,
        query_indices=all_queries,
        gallery_indices=all_gallery,
        device=device,
        batch_size=pair_batch_size,
        query_global=split.global_features.qfeats,
        gallery_global=split.global_features.gfeats,
    )
    num_positive_pairs = positive_index_tensor.numel()
    positive_local = {key: value[:num_positive_pairs] for key, value in scored.items()}
    negative_local = {
        key: value[num_positive_pairs:].view(qids.numel(), max_negatives)
        for key, value in scored.items()
    }
    positive_global = split.similarity[positive_query_tensor, positive_index_tensor]
    negative_global = split.similarity.gather(1, negative_index_tensor)
    global_top1 = split.rankings[:, 0]
    global_correct = gids.index_select(0, global_top1).eq(qids)
    return OracleSetScores(
        qids=qids,
        gids=gids,
        global_correct=global_correct,
        positive_indices=positive_index_tensor,
        positive_offsets=positive_offsets,
        negative_indices=negative_index_tensor,
        positive_global_scores=positive_global.float(),
        negative_global_scores=negative_global.float(),
        positive_local_scores=positive_local,
        negative_local_scores=negative_local,
    )


def evaluate_setwise(
    oracle: OracleSetScores,
    specs: Sequence[ScorerSpec],
    negative_counts: Sequence[int],
    report_negatives: int,
    bootstrap_repetitions: int,
    confidence: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    summary_rows: list[dict] = []
    query_rows: list[dict] = []
    decisions: Dict[tuple[str, int], np.ndarray] = {}
    global_correct = oracle.global_correct.numpy().astype(bool)
    pids = oracle.qids.numpy()
    for scorer_index, spec in enumerate(specs):
        positive_scores = oracle.positive_local_scores[spec.name]
        negative_scores = oracle.negative_local_scores[spec.name]
        for m in negative_counts:
            correct = np.zeros(len(oracle.positive_offsets) - 1, dtype=bool)
            reciprocal_ranks = np.zeros_like(correct, dtype=np.float64)
            pairwise_aucs = np.zeros_like(correct, dtype=np.float64)
            margins = np.zeros_like(correct, dtype=np.float64)
            for query_index in range(correct.size):
                start, stop = oracle.positive_offsets[query_index : query_index + 2]
                positives = positive_scores[start:stop].float()
                negatives = negative_scores[query_index, :m].float()
                best_positive = positives.max()
                best_negative = negatives.max()
                correct[query_index] = bool(best_positive > best_negative)
                rank = 1 + int((negatives >= best_positive).sum().item())
                reciprocal_ranks[query_index] = 1.0 / float(rank)
                comparisons = positives[:, None] - negatives[None, :]
                pairwise_aucs[query_index] = float(
                    ((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean().item()
                )
                margins[query_index] = float((best_positive - best_negative).item())
                if m == report_negatives:
                    query_rows.append({
                        "scorer": spec.name,
                        "query_index": query_index,
                        "pid": int(oracle.qids[query_index].item()),
                        "num_positives": int(stop - start),
                        "num_hard_negatives": m,
                        "global_correct": bool(global_correct[query_index]),
                        "setwise_correct": bool(correct[query_index]),
                        "first_positive_rank": rank,
                        "best_positive_score": float(best_positive.item()),
                        "best_negative_score": float(best_negative.item()),
                        "setwise_margin": margins[query_index],
                        "query_pairwise_auc": pairwise_aucs[query_index],
                    })

            global_fail = ~global_correct
            recovered = global_fail & correct
            harmed = global_correct & (~correct)
            decisions[(spec.name, int(m))] = correct.copy()
            recovery_rate = safe_numpy_rate(recovered, global_fail)
            harm_rate = safe_numpy_rate(harmed, global_correct)

            def recovery_fn(indices):
                return safe_numpy_rate(recovered[indices], global_fail[indices], nan_if_empty=True)

            def harm_fn(indices):
                return safe_numpy_rate(harmed[indices], global_correct[indices], nan_if_empty=True)

            recovery_ci = grouped_bootstrap_ci(
                recovery_fn, pids, bootstrap_repetitions, confidence, seed + scorer_index * 100 + m
            )
            harm_ci = grouped_bootstrap_ci(
                harm_fn, pids, bootstrap_repetitions, confidence, seed + scorer_index * 100 + m + 1
            )
            summary_rows.append({
                "scorer": spec.name,
                "num_hard_negatives": m,
                "num_queries": int(correct.size),
                "setwise_accuracy": float(correct.mean()),
                "pairwise_auc": float(pairwise_aucs.mean()),
                "mean_reciprocal_rank": float(reciprocal_ranks.mean()),
                "mean_setwise_margin": float(margins.mean()),
                "recovered": int(recovered.sum()),
                "recovery_rate": recovery_rate,
                "recovery_ci_low": recovery_ci[0],
                "recovery_ci_high": recovery_ci[1],
                "harmed": int(harmed.sum()),
                "harm_rate": harm_rate,
                "harm_ci_low": harm_ci[0],
                "harm_ci_high": harm_ci[1],
            })

    maxsim_names = [spec.name for spec in specs if spec.kind == "maxsim"]
    if maxsim_names:
        baseline_name = maxsim_names[0]
        for row_index, row in enumerate(summary_rows):
            m = int(row["num_hard_negatives"])
            method_correct = decisions[(str(row["scorer"]), m)]
            baseline_correct = decisions[(baseline_name, m)]
            global_fail = ~global_correct

            def recovery_delta(indices):
                mask = global_fail[indices]
                if not mask.any():
                    return np.nan
                return float(
                    method_correct[indices][mask].mean()
                    - baseline_correct[indices][mask].mean()
                )

            def harm_delta(indices):
                mask = global_correct[indices]
                if not mask.any():
                    return np.nan
                method_harm = (~method_correct[indices][mask]).mean()
                baseline_harm = (~baseline_correct[indices][mask]).mean()
                return float(method_harm - baseline_harm)

            delta_recovery = recovery_delta(np.arange(global_correct.size))
            delta_harm = harm_delta(np.arange(global_correct.size))
            recovery_delta_ci = grouped_bootstrap_ci(
                recovery_delta,
                pids,
                bootstrap_repetitions,
                confidence,
                seed + 5000 + row_index * 2,
            )
            harm_delta_ci = grouped_bootstrap_ci(
                harm_delta,
                pids,
                bootstrap_repetitions,
                confidence,
                seed + 5001 + row_index * 2,
            )
            row.update({
                "comparison_scorer": baseline_name,
                "delta_recovery_vs_maxsim": delta_recovery,
                "delta_recovery_ci_low": recovery_delta_ci[0],
                "delta_recovery_ci_high": recovery_delta_ci[1],
                "delta_harm_vs_maxsim": delta_harm,
                "delta_harm_ci_low": harm_delta_ci[0],
                "delta_harm_ci_high": harm_delta_ci[1],
            })
    return summary_rows, query_rows


def evaluate_probes(
    validation: OracleSetScores,
    test: OracleSetScores,
    specs: Sequence[ScorerSpec],
    num_negatives: int,
    regularization_c: float,
    seed: int,
) -> list[dict]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "The controlled probe requires scikit-learn; install requirements.txt"
        ) from exc

    val_base = build_probe_base(validation, num_negatives, seed)
    test_base = build_probe_base(test, num_negatives, seed + 1)
    rows: list[dict] = []

    def fit_and_evaluate(
        scorer: str,
        probe_name: str,
        val_columns: Sequence[np.ndarray],
        test_columns: Sequence[np.ndarray],
    ) -> dict:
        x_val = np.column_stack(val_columns)
        x_test = np.column_stack(test_columns)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=regularization_c,
                class_weight="balanced",
                max_iter=500,
                random_state=seed,
            ),
        )
        model.fit(
            x_val,
            val_base["labels"],
            logisticregression__sample_weight=val_base["weights"],
        )
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = probabilities >= 0.5
        labels = test_base["labels"].astype(bool)
        pair_correct = predictions == labels
        global_fail = test_base["raw_global_delta"] < 0
        global_correct = test_base["raw_global_delta"] > 0
        weights = test_base["weights"]

        def weighted_rate(values, mask):
            selected = weights[mask]
            return 0.0 if selected.size == 0 else float(np.average(values[mask], weights=selected))

        return {
            "scorer": scorer,
            "probe": probe_name,
            "train_split": "validation",
            "test_split": "test",
            "num_train_pairs": int(x_val.shape[0]),
            "num_test_pairs": int(x_test.shape[0]),
            "pair_accuracy": float(accuracy_score(labels, predictions, sample_weight=weights)),
            "roc_auc": float(roc_auc_score(labels, probabilities, sample_weight=weights)),
            "log_loss": float(log_loss(labels, probabilities, sample_weight=weights)),
            "recovery_rate_on_global_fail_pairs": weighted_rate(pair_correct, global_fail),
            "harm_rate_on_global_correct_pairs": weighted_rate(~pair_correct, global_correct),
        }

    rows.append(
        fit_and_evaluate(
            "__global__",
            "global_only",
            [val_base["oriented_global_delta"]],
            [test_base["oriented_global_delta"]],
        )
    )
    for scorer_index, spec in enumerate(specs):
        val_local = probe_local_delta(validation, spec.name, num_negatives)
        test_local = probe_local_delta(test, spec.name, num_negatives)
        oriented_val_local = val_local * val_base["orientation"]
        oriented_test_local = test_local * test_base["orientation"]
        rows.append(
            fit_and_evaluate(
                spec.name,
                "hidden_only",
                [oriented_val_local],
                [oriented_test_local],
            )
        )
        rows.append(
            fit_and_evaluate(
                spec.name,
                "global_plus_hidden",
                [val_base["oriented_global_delta"], oriented_val_local],
                [test_base["oriented_global_delta"], oriented_test_local],
            )
        )
        val_rng = np.random.default_rng(seed + 1000 + scorer_index)
        test_rng = np.random.default_rng(seed + 2000 + scorer_index)
        permuted_val = val_rng.permutation(oriented_val_local)
        permuted_test = test_rng.permutation(oriented_test_local)
        rows.append(
            fit_and_evaluate(
                spec.name,
                "permuted_hidden_control",
                [val_base["oriented_global_delta"], permuted_val],
                [test_base["oriented_global_delta"], permuted_test],
            )
        )
    return rows


def build_probe_base(oracle: OracleSetScores, num_negatives: int, seed: int) -> dict:
    global_parts = []
    pid_parts = []
    for query_index in range(len(oracle.positive_offsets) - 1):
        start, stop = oracle.positive_offsets[query_index : query_index + 2]
        positive = oracle.positive_global_scores[start:stop]
        negative = oracle.negative_global_scores[query_index, :num_negatives]
        global_parts.append((positive[:, None] - negative[None, :]).reshape(-1))
        pid_parts.append(
            torch.full(
                ((stop - start) * num_negatives,),
                int(oracle.qids[query_index].item()),
                dtype=torch.long,
            )
        )
    raw_global = torch.cat(global_parts).numpy().astype(np.float64)
    pids = torch.cat(pid_parts).numpy()
    rng = np.random.default_rng(seed)
    orientation = rng.choice(np.array([-1.0, 1.0]), size=raw_global.size)
    labels = (orientation > 0).astype(np.int64)
    unique, counts = np.unique(pids, return_counts=True)
    count_by_pid = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.asarray([1.0 / count_by_pid[int(pid)] for pid in pids], dtype=np.float64)
    weights *= weights.size / weights.sum()
    return {
        "raw_global_delta": raw_global,
        "oriented_global_delta": raw_global * orientation,
        "orientation": orientation,
        "labels": labels,
        "pids": pids,
        "weights": weights,
    }


def probe_local_delta(
    oracle: OracleSetScores, scorer: str, num_negatives: int
) -> np.ndarray:
    parts = []
    positive_scores = oracle.positive_local_scores[scorer]
    negative_scores = oracle.negative_local_scores[scorer]
    for query_index in range(len(oracle.positive_offsets) - 1):
        start, stop = oracle.positive_offsets[query_index : query_index + 2]
        positive = positive_scores[start:stop].float()
        negative = negative_scores[query_index, :num_negatives].float()
        parts.append((positive[:, None] - negative[None, :]).reshape(-1))
    return torch.cat(parts).numpy().astype(np.float64)


def evaluate_coverage(split: SplitScores, topk_values: Sequence[int]) -> list[dict]:
    qids = split.global_features.qids.cpu().long()
    gids = split.global_features.gids.cpu().long()
    global_correct = gids.index_select(0, split.rankings[:, 0]).eq(qids)
    global_fail = ~global_correct
    rows = []
    for k in topk_values:
        available = positive_in_prefix(split.rankings, qids, gids, k)
        eligible = global_fail & available
        rows.append({
            "split": split.name,
            "k": int(k),
            "num_queries": int(qids.numel()),
            "global_failures": int(global_fail.sum().item()),
            "global_failures_with_positive_in_topk": int(eligible.sum().item()),
            "positive_coverage_given_global_failure": tensor_rate(eligible, global_fail),
            "maximum_possible_overall_recovery_at_k": tensor_rate(eligible, global_fail),
        })
    return rows


def evaluate_reranking(
    validation: SplitScores,
    test: SplitScores,
    specs: Sequence[ScorerSpec],
    topk_values: Sequence[int],
    fusion_weights: Sequence[float],
    existing_test_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict], list[dict], list[dict]]:
    validation_rows = []
    global_row = {
        "selection_split": "validation", "scorer": "__global__", "kind": "global",
        "k": 1, "mode": "noop", "fusion_weight": "",
        **top1_variant_summary(validation, validation.global_topk_scores[:, :1], 1),
    }
    validation_rows.append(global_row)
    for spec, k, mode, weight, candidate_scores in iter_variants(
        validation, specs, topk_values, fusion_weights
    ):
        row = top1_variant_summary(validation, candidate_scores, k)
        validation_rows.append({
            "selection_split": "validation",
            "scorer": spec.name,
            "kind": spec.kind,
            "k": k,
            "mode": mode,
            "fusion_weight": "" if weight is None else weight,
            **row,
        })

    selected = []
    for spec in specs:
        candidates = [row for row in validation_rows if row["scorer"] == spec.name] + [global_row]
        selected.append(max(candidates, key=rerank_selection_key))
    overall = max(validation_rows, key=rerank_selection_key)
    selected.append({**overall, "selected_as": "overall"})

    test_rows = []
    for selected_row in selected:
        match = find_existing_test_row(existing_test_rows, selected_row)
        output = {
            "selected_as": selected_row.get("selected_as", "per_scorer"),
            "selected_on": "validation",
            "scorer": selected_row["scorer"],
            "kind": selected_row["kind"],
            "k": selected_row["k"],
            "mode": selected_row["mode"],
            "fusion_weight": selected_row["fusion_weight"],
            "validation_r1": selected_row["r1"],
            "validation_rescued": selected_row["rescued"],
            "validation_harmed": selected_row["harmed"],
        }
        if match is not None:
            for key in (
                "r1", "r5", "r10", "map", "minp", "rescued", "harmed", "net_rescued",
                "rescue_rate_among_eligible", "harm_rate_among_global_correct",
            ):
                output[f"test_{key}"] = match.get(key, "")
        else:
            scores = get_variant_scores(test, selected_row)
            top1 = top1_variant_summary(test, scores, int(selected_row["k"]))
            for key, value in top1.items():
                output[f"test_{key}"] = value
            if selected_row.get("selected_as") == "overall":
                full = streaming_mixed_metrics(
                    test,
                    scores,
                    int(selected_row["k"]),
                    torch.ones(test.global_features.qids.numel(), dtype=torch.bool),
                )
                for key, value in full.items():
                    output[f"test_{key}"] = value
        test_rows.append(output)
    return validation_rows, selected, test_rows


def evaluate_gating(
    validation: SplitScores,
    test: SplitScores,
    specs: Sequence[ScorerSpec],
    topk_values: Sequence[int],
    fusion_weights: Sequence[float],
    harm_budgets: Sequence[float],
    threshold_steps: int,
) -> list[dict]:
    if threshold_steps < 2:
        raise ValueError("gating_threshold_steps must be at least 2")
    qids = validation.global_features.qids.cpu().long()
    gids = validation.global_features.gids.cpu().long()
    global_pred = validation.rankings[:, 0]
    global_correct = gids.index_select(0, global_pred).eq(qids)
    global_margin = validation.global_topk_scores[:, 0] - validation.global_topk_scores[:, 1]
    best_by_budget: Dict[float, dict | None] = {float(budget): None for budget in harm_budgets}

    for spec, k, mode, weight, candidate_scores in iter_variants(
        validation, specs, topk_values, fusion_weights
    ):
        rerank_position = candidate_scores[:, :k].argmax(dim=1)
        rerank_pred = validation.topk_indices[:, :k].gather(
            1, rerank_position.view(-1, 1)
        ).view(-1)
        rerank_correct = gids.index_select(0, rerank_pred).eq(qids)
        disagreement = rerank_pred.ne(global_pred)
        margins = global_margin[disagreement]
        if margins.numel():
            quantiles = torch.linspace(0.0, 1.0, threshold_steps)
            thresholds = torch.unique(torch.quantile(margins.float(), quantiles)).tolist()
        else:
            thresholds = []
        thresholds = [float("-inf"), *[float(value) for value in thresholds]]
        for threshold in thresholds:
            gate = disagreement & global_margin.le(threshold)
            final_correct = torch.where(gate, rerank_correct, global_correct)
            rescued = (~global_correct) & final_correct
            harmed = global_correct & (~final_correct)
            candidate = {
                "scorer": spec.name,
                "kind": spec.kind,
                "k": k,
                "mode": mode,
                "fusion_weight": "" if weight is None else weight,
                "global_margin_threshold": threshold,
                "gated_queries": int(gate.sum().item()),
                "gate_coverage": float(gate.float().mean().item()),
                "r1": float(final_correct.float().mean().item() * 100.0),
                "rescued": int(rescued.sum().item()),
                "harmed": int(harmed.sum().item()),
                "net_rescued": int(rescued.sum().item() - harmed.sum().item()),
                "recovery_rate": tensor_rate(rescued, ~global_correct),
                "harm_rate": tensor_rate(harmed, global_correct),
            }
            for budget in best_by_budget:
                if candidate["harm_rate"] <= budget + 1e-12:
                    current = best_by_budget[budget]
                    if current is None or gate_selection_key(candidate) > gate_selection_key(current):
                        best_by_budget[budget] = candidate

    rows = []
    for budget, selected in sorted(best_by_budget.items()):
        if selected is None:
            continue
        rows.append({"split": "validation", "harm_budget": budget, **selected})

        candidate_scores = get_variant_scores(test, selected)
        k = int(selected["k"])
        rerank_position = candidate_scores[:, :k].argmax(dim=1)
        rerank_pred = test.topk_indices[:, :k].gather(
            1, rerank_position.view(-1, 1)
        ).view(-1)
        qids_test = test.global_features.qids.cpu().long()
        gids_test = test.global_features.gids.cpu().long()
        global_pred_test = test.rankings[:, 0]
        global_correct_test = gids_test.index_select(0, global_pred_test).eq(qids_test)
        rerank_correct_test = gids_test.index_select(0, rerank_pred).eq(qids_test)
        global_margin_test = test.global_topk_scores[:, 0] - test.global_topk_scores[:, 1]
        disagreement_test = rerank_pred.ne(global_pred_test)
        threshold = float(selected["global_margin_threshold"])
        gate_test = disagreement_test & global_margin_test.le(threshold)
        final_correct = torch.where(gate_test, rerank_correct_test, global_correct_test)
        rescued = (~global_correct_test) & final_correct
        harmed = global_correct_test & (~final_correct)
        full_metrics = streaming_mixed_metrics(
            test,
            candidate_scores,
            k,
            gate_test,
        )
        rows.append({
            "split": "test",
            "harm_budget": budget,
            "scorer": selected["scorer"],
            "kind": selected["kind"],
            "k": k,
            "mode": selected["mode"],
            "fusion_weight": selected["fusion_weight"],
            "global_margin_threshold": threshold,
            "gated_queries": int(gate_test.sum().item()),
            "gate_coverage": float(gate_test.float().mean().item()),
            **full_metrics,
            "rescued": int(rescued.sum().item()),
            "harmed": int(harmed.sum().item()),
            "net_rescued": int(rescued.sum().item() - harmed.sum().item()),
            "recovery_rate": tensor_rate(rescued, ~global_correct_test),
            "harm_rate": tensor_rate(harmed, global_correct_test),
        })
    return rows


def iter_variants(
    split: SplitScores,
    specs: Sequence[ScorerSpec],
    topk_values: Sequence[int],
    fusion_weights: Sequence[float],
):
    available_k = split.topk_indices.shape[1]
    for spec in specs:
        local = split.local_topk_scores[spec.name]
        for raw_k in topk_values:
            k = min(int(raw_k), available_k)
            for mode, weight, candidate_scores in candidate_score_variants(
                local[:, :k], split.global_topk_scores[:, :k], fusion_weights
            ):
                yield spec, k, mode, weight, candidate_scores


def top1_variant_summary(split: SplitScores, candidate_scores: torch.Tensor, k: int) -> dict:
    qids = split.global_features.qids.cpu().long()
    gids = split.global_features.gids.cpu().long()
    position = candidate_scores[:, :k].argmax(dim=1)
    predicted = split.topk_indices[:, :k].gather(1, position.view(-1, 1)).view(-1)
    correct = gids.index_select(0, predicted).eq(qids)
    global_correct = gids.index_select(0, split.rankings[:, 0]).eq(qids)
    available = positive_in_prefix(split.rankings, qids, gids, k)
    eligible = (~global_correct) & available
    rescued = (~global_correct) & correct
    harmed = global_correct & (~correct)
    return {
        "r1": float(correct.float().mean().item() * 100.0),
        "rescued": int(rescued.sum().item()),
        "harmed": int(harmed.sum().item()),
        "net_rescued": int(rescued.sum().item() - harmed.sum().item()),
        "rescue_rate_among_eligible": tensor_rate(rescued & eligible, eligible),
        "harm_rate_among_global_correct": tensor_rate(harmed, global_correct),
    }


def get_variant_scores(split: SplitScores, selected: Mapping[str, object]) -> torch.Tensor:
    if selected["mode"] == "noop":
        return split.global_topk_scores[:, :1]
    scorer = str(selected["scorer"])
    k = int(selected["k"])
    target_mode = str(selected["mode"])
    target_weight = selected.get("fusion_weight", "")
    weights = [] if target_weight in {"", None} else [float(target_weight)]
    variants = candidate_score_variants(
        split.local_topk_scores[scorer][:, :k],
        split.global_topk_scores[:, :k],
        weights,
    )
    for mode, _, scores in variants:
        if mode == target_mode:
            return scores
    raise KeyError(f"Variant not found: {scorer}, K={k}, mode={target_mode}")


def streaming_mixed_metrics(
    split: SplitScores,
    candidate_scores: torch.Tensor,
    k: int,
    gate: torch.Tensor,
    batch_queries: int = 256,
) -> dict:
    """Exact ranking metrics without materializing another full ranking matrix."""
    qids = split.global_features.qids.cpu().long()
    gids = split.global_features.gids.cpu().long()
    recall_sums = {1: 0.0, 5: 0.0, 10: 0.0}
    ap_sum = 0.0
    inp_sum = 0.0
    inp_count = 0
    num_queries = qids.numel()
    for start in range(0, num_queries, batch_queries):
        stop = min(start + batch_queries, num_queries)
        rankings = split.rankings[start:stop].clone()
        scores = candidate_scores[start:stop, :k]
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        reranked_prefix = rankings[:, :k].gather(1, order)
        local_gate = gate[start:stop]
        rankings[:, :k] = torch.where(
            local_gate.view(-1, 1), reranked_prefix, rankings[:, :k]
        )
        labels = gids.index_select(0, rankings.reshape(-1)).view_as(rankings)
        matches = labels.eq(qids[start:stop].view(-1, 1))
        cumulative = matches.cumsum(dim=1).float()
        for cutoff in recall_sums:
            kk = min(cutoff, rankings.shape[1])
            recall_sums[cutoff] += float(matches[:, :kk].any(dim=1).float().sum().item())
        ranks = torch.arange(1, rankings.shape[1] + 1, dtype=torch.float32).view(1, -1)
        num_relevant = matches.sum(dim=1).clamp_min(1)
        ap = ((cumulative / ranks) * matches).sum(dim=1) / num_relevant
        ap_sum += float(ap.sum().item())
        for row_index, match_row in enumerate(matches):
            hits = match_row.nonzero(as_tuple=False).flatten()
            if hits.numel():
                last = int(hits[-1].item())
                inp_sum += float(cumulative[row_index, last].item() / float(last + 1))
                inp_count += 1
    return {
        "r1": 100.0 * recall_sums[1] / num_queries,
        "r5": 100.0 * recall_sums[5] / num_queries,
        "r10": 100.0 * recall_sums[10] / num_queries,
        "map": 100.0 * ap_sum / num_queries,
        "minp": 0.0 if inp_count == 0 else 100.0 * inp_sum / inp_count,
    }


def find_existing_test_row(
    rows: Sequence[Mapping[str, object]], selected: Mapping[str, object]
) -> Mapping[str, object] | None:
    target_weight = selected.get("fusion_weight", "")
    for row in rows:
        if row.get("scorer") != selected.get("scorer"):
            continue
        if int(row.get("k") or -1) != int(selected["k"]):
            continue
        if row.get("mode") != selected.get("mode"):
            continue
        row_weight = row.get("fusion_weight", "")
        if target_weight in {"", None} and row_weight in {"", None}:
            return row
        if target_weight not in {"", None} and row_weight not in {"", None}:
            if abs(float(target_weight) - float(row_weight)) < 1e-12:
                return row
    return None


def rerank_selection_key(row: Mapping[str, object]):
    return (
        float(row["r1"]),
        int(row["net_rescued"]),
        -int(row["harmed"]),
        -int(row["k"]),
    )


def gate_selection_key(row: Mapping[str, object]):
    return (
        int(row["net_rescued"]),
        float(row["r1"]),
        -int(row["harmed"]),
        -int(row["gated_queries"]),
    )


def tensor_rate(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    den = int(denominator.sum().item())
    return 0.0 if den == 0 else float(numerator.sum().item()) / float(den)


def safe_numpy_rate(
    numerator: np.ndarray, denominator: np.ndarray, nan_if_empty: bool = False
) -> float:
    den = int(np.asarray(denominator, dtype=bool).sum())
    if den == 0:
        return float("nan") if nan_if_empty else 0.0
    return float(np.asarray(numerator, dtype=bool).sum()) / float(den)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False)


def json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value
