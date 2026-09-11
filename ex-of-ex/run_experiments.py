"""Run many frozen hidden-token scorers from one feature extraction.

Example:
    python ex-of-ex/run_experiments.py \
        --config ex-of-ex/configs/rstp_fast.yaml \
        --root-dir /kaggle/input/itself/benchmark \
        --checkpoint /kaggle/input/checkpoints/RSTP/best.pth \
        --run-name rstp-final-hidden
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import random
import re
import subprocess
import sys
import time
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from diagnostic.global_hidden_recovery.evaluator import (  # noqa: E402
    build_global_decisions,
    compute_global_similarity,
)
from diagnostic.global_hidden_recovery.feature_extractor import (  # noqa: E402
    GlobalFeatureSet,
    GlobalHiddenExtractor,
    HiddenFeatureSet,
    freeze_model,
    inspect_clip_projection,
)
from diagnostic.global_hidden_recovery.run_diagnostic import (  # noqa: E402
    dataset_output_key,
    hidden_from_dict,
    hidden_to_dict,
    load_checkpoint,
    make_repo_args,
    normalize_model_preset,
    resolve_device,
    safe_torch_load,
)
from evaluation import (  # noqa: E402
    candidate_score_variants,
    decision_summary,
    fixed_pair_summary,
    metrics_from_rankings,
    positive_in_prefix,
    rerank_prefix,
)
from scorers import build_specs, score_pairs_all  # noqa: E402


CITATIONS = {
    "maxsim": "FILIP (Yao et al., ICLR 2022), https://openreview.net/forum?id=cpDhcsEDC2",
    "topm": "CFine-inspired (Yan et al., IEEE TIP 2023), https://doi.org/10.1109/TIP.2023.3327924",
    "softmax_t2i": "SCAN (Lee et al., ECCV 2018), https://openaccess.thecvf.com/content_ECCV_2018/html/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.html",
    "smooth_chamfer": "Smooth-Chamfer (Kim et al., CVPR 2023), https://openaccess.thecvf.com/content/CVPR2023/html/Kim_Improving_Cross-Modal_Retrieval_With_Set_of_Diverse_Embeddings_CVPR_2023_paper.html",
    "cfine_cross_grained": "CFine (Yan et al., IEEE TIP 2023), https://doi.org/10.1109/TIP.2023.3327924",
    "tokenflow_stable": "TokenFlow (Zou et al., arXiv:2209.13822), https://arxiv.org/abs/2209.13822",
    "sinkhorn_ot": "Sinkhorn Distances (Cuturi, NeurIPS 2013), https://proceedings.neurips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html",
    "uniform_mean": "Uniform local-similarity baseline (control)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot sweep of final-layer hidden-token retrieval scorers"
    )
    parser.add_argument("--config", nargs="+", required=True, help="One or more YAML suites")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint override (one config only)")
    parser.add_argument("--root-dir", "--root_dir", dest="root_dir", default=None, help="Dataset root override")
    parser.add_argument("--dataset", default=None, help="Dataset name override")
    parser.add_argument("--model-preset", choices=["clip", "itself"], default=None)
    parser.add_argument("--output-dir", default=None, help="Output root override")
    parser.add_argument("--cache-dir", default=None, help="Feature-cache root override")
    parser.add_argument("--batch-size", type=int, default=None, help="Encoder batch size")
    parser.add_argument("--pair-batch-size", type=int, default=None, help="Pair scoring batch size")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--run-name", default=None, help="Human-readable suffix for this run")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore feature cache")
    parser.add_argument("--no-cache", action="store_true", help="Do not read/write feature cache")
    parser.add_argument("--seed", type=int, default=None)
    full_mode = parser.add_mutually_exclusive_group()
    full_mode.add_argument("--full-gallery-only", action="store_true", help="Run additive full-gallery suite instead of repeating v2")
    full_mode.add_argument("--with-full-gallery", action="store_true", help="Run existing suite AND full-gallery experiments in the same run")
    parser.add_argument("--save-full-similarity", action="store_true", help="Save full-gallery float32 similarity matrices (large files)")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if len(cli.config) > 1 and cli.checkpoint:
        raise ValueError("--checkpoint override is only allowed with one config")

    completed = []
    for raw_path in cli.config:
        config_path = Path(raw_path).resolve()
        config = load_config(config_path)
        config = apply_overrides(config, cli)
        output_root = Path(str(config.get("output_dir", HERE / "outputs"))).resolve()
        dataset_key = dataset_output_key(str(config["dataset"]))
        run_name = cli.run_name or str(config.get("run_name", "sweep"))
        if len(cli.config) > 1:
            run_name = f"{run_name}-{config_path.stem}"
        run_id = make_run_id(run_name)
        run_dir = output_root / dataset_key / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        log_path = run_dir / "run.log"
        with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
            tee_out = Tee(sys.__stdout__, log_handle)
            tee_err = Tee(sys.__stderr__, log_handle)
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    stream=sys.stdout,
                    force=True,
                )
                logging.getLogger("PersonSearch.checkpoint").setLevel(logging.WARNING)
                try:
                    summary_path = run_one(config, config_path, run_dir, run_id)
                    completed.append(summary_path)
                except Exception:
                    logging.exception("Run failed")
                    raise
                finally:
                    logging.shutdown()
                    logging.getLogger().handlers.clear()

    print("Completed:")
    for path in completed:
        print(f"  {path}")


def run_one(
    config: Mapping[str, object],
    config_path: Path,
    run_dir: Path,
    run_id: str,
) -> Path:
    started = time.perf_counter()
    logger = logging.getLogger("ex-of-ex")
    dataset_name = str(config["dataset"])
    checkpoint = Path(str(config.get("checkpoint") or "")).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(str(config.get("device", "auto")))
    model_preset = normalize_model_preset(config.get("model_preset", "clip"))
    topk_values = sorted({int(k) for k in config.get("topk", [10, 20, 50])})
    if not topk_values or min(topk_values) <= 0:
        raise ValueError("topk must contain positive integers")
    pair_batch_size = int(config.get("pair_batch_size", 384))
    feature_dtype = str(config.get("feature_dtype", "float16"))
    fusion_weights = [float(v) for v in config.get("fusion_weights", [0.1, 0.2, 0.3, 0.5])]
    specs = build_specs(config.get("scorers", []))
    validated_cfg = config.get("validated") or {}
    validated_enabled = bool(validated_cfg.get("enabled", False))
    legacy_test_sweep = bool(validated_cfg.get("legacy_test_sweep", True))

    logger.info("Run ID: %s", run_id)
    logger.info("Dataset: %s | preset: %s | device: %s", dataset_name, model_preset, device)
    logger.info("Methods (%d): %s", len(specs), ", ".join(spec.name for spec in specs))
    logger.info("Top-K: %s | fusion weights: %s", topk_values, fusion_weights)
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(json_safe(config), handle, sort_keys=False, allow_unicode=True)

    checkpoint_sha = sha256_file(checkpoint)
    repo_args = make_repo_args(config)
    cache_meta = build_cache_meta(
        config,
        repo_args,
        checkpoint,
        checkpoint_sha,
        dataset_name,
        model_preset,
        feature_dtype,
    )
    cache_root = Path(str(config.get("cache_dir", HERE / "cache"))).resolve()
    cache_path = (
        cache_root
        / dataset_output_key(dataset_name)
        / model_preset
        / checkpoint_sha[:12]
        / "final_layer_features.pt"
    )

    from datasets import build_dataloader
    from model import build_model

    stage = time.perf_counter()
    test_img_loader, test_txt_loader, num_classes = build_dataloader(repo_args)
    logger.info("Dataloaders ready in %.1fs", time.perf_counter() - stage)

    stage = time.perf_counter()
    model = build_model(repo_args, num_classes)
    load_checkpoint(model, str(checkpoint))
    model.to(device)
    if device.type == "cpu":
        model.float()
    freeze_model(model)
    verify_frozen(model)
    logger.info("Checkpoint loaded and frozen in %.1fs", time.perf_counter() - stage)

    stage = time.perf_counter()
    global_features, hidden_features, cache_status = load_or_extract_features(
        model=model,
        device=device,
        txt_loader=test_txt_loader,
        img_loader=test_img_loader,
        feature_dtype=feature_dtype,
        cache_path=cache_path,
        cache_meta=cache_meta,
        use_cache=bool(config.get("cache_features", True)),
        force=bool(config.get("force_recompute", False)),
    )
    logger.info(
        "Features %s in %.1fs | queries=%d gallery=%d text=%d patches=%d",
        cache_status,
        time.perf_counter() - stage,
        global_features.qids.numel(),
        global_features.gids.numel(),
        hidden_features.text_features.shape[1],
        hidden_features.image_features.shape[1],
    )

    if config.get("full_gallery_only", False):
        from full_gallery import run_addon
        return run_addon(config, run_dir, model, device, repo_args, global_features,
                         hidden_features, test_txt_loader, test_img_loader,
                         cache_path, cache_meta)

    similarity = compute_global_similarity(global_features.qfeats, global_features.gfeats)
    global_rankings = torch.argsort(similarity, dim=1, descending=True).cpu()
    global_metrics = metrics_from_rankings(
        global_rankings, global_features.qids, global_features.gids
    )
    decisions = build_global_decisions(
        similarity, global_features.qids, global_features.gids, topk_values
    )
    max_k = min(max(topk_values), similarity.shape[1])
    topk_values = [min(k, max_k) for k in topk_values]
    topk_values = sorted(set(topk_values))
    topk_indices = global_rankings[:, :max_k].contiguous()
    global_topk_scores = similarity.gather(1, topk_indices)
    logger.info(
        "Global baseline | R1 %.2f R5 %.2f R10 %.2f mAP %.2f",
        global_metrics["r1"], global_metrics["r5"], global_metrics["r10"], global_metrics["map"],
    )

    num_queries = global_features.qids.numel()
    query_indices = torch.arange(num_queries, dtype=torch.long)
    best_positive = decisions["best_positive_index"].long()
    best_negative = decisions["best_negative_index"].long()
    if (best_positive < 0).any() or (best_negative < 0).any():
        raise RuntimeError("Every query must have a positive and a negative gallery example")

    fixed_scores = {}
    if legacy_test_sweep:
        # One call scores fixed positive and negative pairs for every method.
        stage = time.perf_counter()
        fixed_all = score_pairs_all(
            specs=specs,
            text_features=hidden_features.text_features,
            text_mask=hidden_features.text_mask,
            image_features=hidden_features.image_features,
            query_indices=torch.cat([query_indices, query_indices]),
            gallery_indices=torch.cat([best_positive, best_negative]),
            device=device,
            batch_size=pair_batch_size,
            query_global=global_features.qfeats,
            gallery_global=global_features.gfeats,
        )
        fixed_scores = {
            name: (values[:num_queries], values[num_queries:])
            for name, values in fixed_all.items()
        }
        logger.info("Fixed-pair sweep completed in %.1fs", time.perf_counter() - stage)
    else:
        logger.info("Skipping legacy fixed-pair/test grid in validation-selected mode")

    # The expensive local similarity tensor is also shared by all scorers here.
    stage = time.perf_counter()
    flat_q = query_indices.repeat_interleave(max_k)
    flat_g = topk_indices.reshape(-1)
    topk_all = score_pairs_all(
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
    topk_all = {name: values.view(num_queries, max_k) for name, values in topk_all.items()}
    scoring_seconds = time.perf_counter() - stage
    logger.info("Top-%d scorer fan-out completed in %.1fs", max_k, scoring_seconds)

    summary_rows = [{
        "run_id": run_id,
        "dataset": dataset_name,
        "scorer": "global_baseline",
        "kind": "global",
        "params": "{}",
        "k": "",
        "mode": "global",
        "fusion_weight": "",
        **global_metrics,
        "rescued": 0,
        "harmed": 0,
        "net_rescued": 0,
        "rescue_rate_among_eligible": 0.0,
        "harm_rate_among_global_correct": 0.0,
        "proper_oracle_r1": global_metrics["r1"],
        "fixed_recovery_rate": "",
        "fixed_harm_rate": "",
    }]
    scorer_summaries = {}
    delta_global = decisions["delta_global"].float()
    global_correct = global_features.gids.index_select(0, global_rankings[:, 0]).eq(
        global_features.qids
    )

    scorers_root = run_dir / "scorers"
    scorers_root.mkdir(parents=True, exist_ok=True)
    legacy_specs = specs if legacy_test_sweep else []
    for spec in legacy_specs:
        logger.info("Evaluating %s (%s)", spec.name, spec.kind)
        positive_scores, negative_scores = fixed_scores[spec.name]
        fixed = fixed_pair_summary(delta_global, positive_scores, negative_scores)
        local_topk = topk_all[spec.name]
        variant_summaries = []
        query_rows = []

        for k in topk_values:
            positive_available = positive_in_prefix(
                global_rankings, global_features.qids, global_features.gids, k
            )
            variants = candidate_score_variants(
                local_topk[:, :k], global_topk_scores[:, :k], fusion_weights
            )
            for mode, weight, candidate_scores in variants:
                rankings = rerank_prefix(global_rankings, candidate_scores, k)
                metrics, correct = decision_summary(
                    rankings,
                    global_rankings,
                    global_features.qids,
                    global_features.gids,
                    positive_available,
                )
                variant = {
                    "k": int(k),
                    "mode": mode,
                    "fusion_weight": weight,
                    **metrics,
                }
                variant_summaries.append(variant)
                summary_rows.append({
                    "run_id": run_id,
                    "dataset": dataset_name,
                    "scorer": spec.name,
                    "kind": spec.kind,
                    "params": json.dumps(dict(spec.params), sort_keys=True),
                    **variant,
                    "fixed_recovery_rate": fixed["recovery_rate"],
                    "fixed_harm_rate": fixed["harm_rate"],
                })

                predicted_gallery = rankings[:, 0]
                predicted_pid = global_features.gids.index_select(0, predicted_gallery)
                rescued = (~global_correct) & correct
                harmed = global_correct & (~correct)
                delta_local = positive_scores - negative_scores
                for qi in range(num_queries):
                    query_rows.append({
                        "query_index": qi,
                        "pid": int(global_features.qids[qi].item()),
                        "caption": global_features.captions[qi] if qi < len(global_features.captions) else "",
                        "k": int(k),
                        "mode": mode,
                        "fusion_weight": "" if weight is None else weight,
                        "global_top1_gallery_index": int(global_rankings[qi, 0].item()),
                        "global_top1_pid": int(global_features.gids[global_rankings[qi, 0]].item()),
                        "predicted_gallery_index": int(predicted_gallery[qi].item()),
                        "predicted_pid": int(predicted_pid[qi].item()),
                        "global_correct": bool(global_correct[qi].item()),
                        "correct": bool(correct[qi].item()),
                        "rescued": bool(rescued[qi].item()),
                        "harmed": bool(harmed[qi].item()),
                        "positive_in_global_topk": bool(positive_available[qi].item()),
                        "delta_global_fixed_pair": float(delta_global[qi].item()),
                        "delta_local_fixed_pair": float(delta_local[qi].item()),
                    })

        scorer_dir = scorers_root / slugify(spec.name)
        scorer_dir.mkdir(parents=True, exist_ok=False)
        scorer_summary = {
            "name": spec.name,
            "kind": spec.kind,
            "params": dict(spec.params),
            "citation": CITATIONS.get(spec.kind, ""),
            "fixed_pair": fixed,
            "variants": variant_summaries,
        }
        write_json(scorer_dir / "summary.json", scorer_summary)
        write_csv(scorer_dir / "query_results.csv", query_rows)
        scorer_summaries[spec.name] = scorer_summary

        best = max(variant_summaries, key=lambda item: (item["r1"], item["map"]))
        logger.info(
            "%s best | K=%d %s | R1 %.2f mAP %.2f | rescued=%d harmed=%d",
            spec.name,
            best["k"],
            best["mode"],
            best["r1"],
            best["map"],
            best["rescued"],
            best["harmed"],
        )

    summary_path = run_dir / "summary.csv"
    write_csv(summary_path, summary_rows)

    validated_summary = None
    prepared_validation = None
    if validated_enabled:
        from validated import (
            build_eval_split_loaders,
            make_split_scores,
            run_validated_suite,
        )

        logger.info("Starting validation-selected experiment suite")
        stage = time.perf_counter()
        validated_cfg = dict(config.get("validated") or {})
        allow_missing_val = (
            int(validated_cfg.get("protocol_version", 1)) == 2
            and validated_cfg.get("missing_validation") == "test_identity_holdout"
        )
        val_img_loader, val_txt_loader = build_eval_split_loaders(
            repo_args, "val", allow_missing=allow_missing_val
        )
        val_cache_meta = dict(cache_meta)
        val_cache_meta["split"] = "val"
        val_cache_path = cache_path.parent / "validation_features.pt"
        validation_state = None
        if val_img_loader is not None:
            val_global, val_hidden, val_cache_status = load_or_extract_features(
                model=model, device=device, txt_loader=val_txt_loader, img_loader=val_img_loader,
                feature_dtype=feature_dtype, cache_path=val_cache_path, cache_meta=val_cache_meta,
                use_cache=bool(config.get("cache_features", True)),
                force=bool(config.get("force_recompute", False)),
            )
            logger.info("Validation features %s | queries=%d gallery=%d",
                        val_cache_status, val_global.qids.numel(), val_global.gids.numel())
            validation_state = make_split_scores(
                name="validation", global_features=val_global, hidden_features=val_hidden,
                specs=specs, topk_values=topk_values, device=device, pair_batch_size=pair_batch_size,
            )
            prepared_validation = (val_global, val_hidden)
        test_state = make_split_scores(
            name="test",
            global_features=global_features,
            hidden_features=hidden_features,
            specs=specs,
            topk_values=topk_values,
            device=device,
            pair_batch_size=pair_batch_size,
            similarity=similarity,
            rankings=global_rankings,
            topk_indices=topk_indices,
            global_topk_scores=global_topk_scores,
            local_topk_scores=topk_all,
        )
        validated_summary = run_validated_suite(
            config=config,
            run_dir=run_dir,
            specs=specs,
            validation=validation_state,
            test=test_state,
            device=device,
            pair_batch_size=pair_batch_size,
            existing_test_rows=summary_rows,
        )
        logger.info(
            "Validation-selected suite completed in %.1fs", time.perf_counter() - stage
        )

    full_gallery_summary = None
    if config.get("with_full_gallery", False):
        from full_gallery import run_addon
        full_gallery_summary = run_addon(
            config, run_dir, model, device, repo_args, global_features, hidden_features,
            test_txt_loader, test_img_loader, cache_path, cache_meta,
            prepared_validation=prepared_validation)

    total_seconds = time.perf_counter() - started
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "config_path": str(config_path),
        "config": json_safe(config),
        "dataset": dataset_name,
        "model_preset": model_preset,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "cache": {"status": cache_status, "path": str(cache_path)},
        "device": str(device),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "git_commit": git_commit(),
        "clip_projection": inspect_clip_projection(model),
        "features": {
            "layer": "final",
            "special_tokens": "SOT/EOT/padding excluded; image CLS excluded",
            "feature_dtype": hidden_features.feature_dtype,
            "patch_grid": list(hidden_features.patch_grid),
            "num_queries": int(num_queries),
            "num_gallery": int(global_features.gids.numel()),
        },
        "global_metrics": global_metrics,
        "timing_seconds": {"topk_scorer_fanout": scoring_seconds, "total": total_seconds},
        "citations": CITATIONS,
        "outputs": {
            "full_gallery_summary": str(full_gallery_summary) if full_gallery_summary else None,
            "log": str(run_dir / "run.log"),
            "summary_csv": str(summary_path),
            "scorers_dir": str(scorers_root),
            "validated_dir": str(run_dir / ("validated_v2" if
                int(dict(config.get("validated") or {}).get("protocol_version", 1)) == 2
                else "validated")) if validated_summary else None,
        },
        "scorers": scorer_summaries,
        "validated": validated_summary,
    }
    write_json(run_dir / "manifest.json", manifest)
    logger.info("All outputs written to %s", run_dir)
    logger.info("Total runtime: %.1fs", total_seconds)
    return summary_path


def load_or_extract_features(
    model,
    device: torch.device,
    txt_loader,
    img_loader,
    feature_dtype: str,
    cache_path: Path,
    cache_meta: Mapping[str, object],
    use_cache: bool,
    force: bool,
) -> tuple[GlobalFeatureSet, HiddenFeatureSet, str]:
    logger = logging.getLogger("ex-of-ex")
    if use_cache and cache_path.is_file() and not force:
        payload = safe_torch_load(cache_path, map_location="cpu")
        if payload.get("meta") == dict(cache_meta):
            logger.info("Using feature cache: %s", cache_path)
            return global_from_dict(payload["global"]), hidden_from_dict(payload["hidden"]), "loaded"
        logger.warning("Ignoring cache with mismatched metadata: %s", cache_path)

    extractor = GlobalHiddenExtractor(model, device=device, feature_dtype=feature_dtype)
    global_features = extract_official_global(model, img_loader, txt_loader)
    hidden_features = extractor.extract_hidden_features(txt_loader, img_loader)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        torch.save({
            "meta": dict(cache_meta),
            "global": global_to_dict(global_features),
            "hidden": hidden_to_dict(hidden_features),
        }, temporary)
        os.replace(temporary, cache_path)
        logger.info("Feature cache written: %s", cache_path)
    return global_features, hidden_features, "extracted"


def extract_official_global(model, img_loader, txt_loader) -> GlobalFeatureSet:
    """Use the repository's official embedding implementation exactly once."""
    from utils.metrics import Evaluator as OfficialEvaluator

    evaluator = OfficialEvaluator(img_loader, txt_loader, Namespace(only_global=True))
    qfeats, gfeats, qids, gids = evaluator._compute_embedding(model)
    txt_set = getattr(txt_loader, "test_txt_set", None)
    captions = list(getattr(txt_set, "captions", []))
    if not captions:
        captions = [""] * int(qids.numel())
    return GlobalFeatureSet(
        qfeats=qfeats.float().cpu(),
        gfeats=gfeats.float().cpu(),
        qids=qids.long().cpu(),
        gids=gids.long().cpu(),
        captions=captions,
    )


def global_to_dict(features: GlobalFeatureSet) -> Dict[str, object]:
    return {
        "qfeats": features.qfeats,
        "gfeats": features.gfeats,
        "qids": features.qids,
        "gids": features.gids,
        "captions": features.captions,
    }


def global_from_dict(value: Mapping[str, object]) -> GlobalFeatureSet:
    return GlobalFeatureSet(
        qfeats=value["qfeats"],
        gfeats=value["gfeats"],
        qids=value["qids"],
        gids=value["gids"],
        captions=list(value.get("captions", [])),
    )


def build_cache_meta(
    config: Mapping[str, object],
    repo_args: Namespace,
    checkpoint: Path,
    checkpoint_sha: str,
    dataset_name: str,
    model_preset: str,
    feature_dtype: str,
) -> Dict[str, object]:
    return {
        "schema": 1,
        "dataset": dataset_name,
        "root_dir": str(Path(str(repo_args.root_dir)).expanduser().resolve()),
        "split": str(config.get("split", "test")),
        "model_preset": model_preset,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "pretrain_choice": str(repo_args.pretrain_choice),
        "img_size": list(repo_args.img_size),
        "stride_size": int(repo_args.stride_size),
        "text_length": int(repo_args.text_length),
        "feature_dtype": feature_dtype,
        "token_policy": "final_layer_without_sot_eot_padding_or_image_cls",
    }


def apply_overrides(config: Mapping[str, object], cli: argparse.Namespace) -> Dict[str, object]:
    config = dict(config)
    mapping = {
        "checkpoint": cli.checkpoint,
        "root_dir": cli.root_dir,
        "dataset": cli.dataset,
        "model_preset": cli.model_preset,
        "output_dir": cli.output_dir,
        "cache_dir": cli.cache_dir,
        "batch_size": cli.batch_size,
        "pair_batch_size": cli.pair_batch_size,
        "device": cli.device,
        "seed": cli.seed,
    }
    for key, value in mapping.items():
        if value is not None:
            config[key] = value
    if cli.force_recompute:
        config["force_recompute"] = True
    if cli.no_cache:
        config["cache_features"] = False
    config["full_gallery_only"] = getattr(cli, "full_gallery_only", False)
    config["with_full_gallery"] = getattr(cli, "with_full_gallery", False)
    config["save_full_similarity"] = getattr(cli, "save_full_similarity", False)
    return config


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    if "dataset" not in value:
        raise ValueError(f"Config is missing dataset: {path}")
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen(model) -> None:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Model must be in eval mode with all parameters frozen")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_id(name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}__{slugify(name)}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip()).strip("-._")
    return slug or "run"


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


if __name__ == "__main__":
    main()
