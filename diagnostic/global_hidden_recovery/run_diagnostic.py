from __future__ import annotations

import argparse
import logging
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostic.global_hidden_recovery.analysis import (
    fixed_pair_ci_metrics,
    rescue_ci,
    write_combined_summary,
)
from diagnostic.global_hidden_recovery.evaluator import (
    assign_categories,
    build_global_decisions,
    compare_metrics,
    compute_global_similarity,
    compute_retrieval_metrics,
    summarize_fixed_pair,
    summarize_oracle,
    summarize_topk,
    write_query_csv,
    write_summary_json,
)
from diagnostic.global_hidden_recovery.feature_extractor import (
    GlobalHiddenExtractor,
    HiddenFeatureSet,
    freeze_model,
    inspect_clip_projection,
)
from diagnostic.global_hidden_recovery.hidden_scorer import HiddenScorer
from diagnostic.global_hidden_recovery.plotting import (
    plot_margin_scatter,
    write_recovery_funnel,
)


DEFAULT_REPO_ARGS = {
    "tau": 0.015,
    "select_ratio": 0.4,
    "margin": 0.1,
    "lambda1_weight": 0.5,
    "lambda2_weight": 3.5,
    "local_rank": 0,
    "output_dir": "run_logs",
    "name": "ITSELF",
    "log_period": 20,
    "eval_period": 1,
    "val_dataset": "test",
    "resume": False,
    "resume_ckpt_file": "",
    "finetune": "",
    "pretrain": "",
    "pretrain_choice": "ViT-B/16",
    "temperature": 0.02,
    "img_aug": True,
    "txt_aug": True,
    "loss_names": "tal+cid",
    "img_size": (384, 128),
    "stride_size": 16,
    "text_length": 77,
    "vocab_size": 49408,
    "optimizer": "Adam",
    "lr": 1e-5,
    "bias_lr_factor": 2.0,
    "lr_factor": 5.0,
    "momentum": 0.9,
    "weight_decay": 4e-5,
    "weight_decay_bias": 0.0,
    "alpha": 0.9,
    "beta": 0.999,
    "num_epoch": 60,
    "milestones": (45, 50),
    "gamma": 0.1,
    "warmup_factor": 0.1,
    "warmup_epochs": 5,
    "warmup_method": "linear",
    "lrscheduler": "cosine",
    "target_lr": 0,
    "power": 0.9,
    "dataset_name": "CUHK-PEDES",
    "sampler": "identity",
    "num_instance": 2,
    "root_dir": "data",
    "batch_size": 128,
    "test_batch_size": 128,
    "num_workers": 4,
    "training": False,
    "only_global": True,
    "return_all": False,
    "topk_type": "mean",
    "layer_index": -1,
    "average_attn_weights": True,
    "modify_k": False,
    "distributed": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ITSELF global-hidden recovery diagnostic")
    parser.add_argument("--config", nargs="+", required=True, help="One or more diagnostic YAML configs")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--dataset", default=None, help="Override dataset name")
    parser.add_argument(
        "--root_dir",
        "--root-dir",
        dest="root_dir",
        default=None,
        help="Override dataset root directory",
    )
    parser.add_argument(
        "--model-preset",
        choices=["clip", "itself"],
        default=None,
        help="Architecture/evaluation flag preset: clip -> --only_global; itself -> --return_all --topk_type custom --modify_k",
    )
    parser.add_argument("--output-dir", default=None, help="Override diagnostic output root")
    parser.add_argument("--batch-size", type=int, default=None, help="Override test batch size")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, or auto")
    parser.add_argument("--cache-features", action="store_true", help="Cache hidden features")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore hidden feature cache")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if cli.verbose else logging.WARNING,
        format="%(message)s",
    )

    summaries = []
    output_root = None
    for config_path in cli.config:
        config = load_config(Path(config_path))
        config = apply_cli_overrides(config, cli)
        summary = run_one(config, cli.verbose)
        summaries.append(summary)
        output_root = Path(config.get("output_dir", default_output_root()))

    if len(summaries) > 1:
        assert output_root is not None
        combined_path = output_root / "combined_summary.csv"
        write_combined_summary(summaries, combined_path)
        print(f"[combined] outputs -> {combined_path}")


def run_one(config: Mapping[str, object], verbose: bool = False) -> Dict[str, object]:
    dataset_name = str(config["dataset"])
    model_preset = normalize_model_preset(config.get("model_preset", "clip"))
    dataset_key = dataset_output_key(dataset_name)
    checkpoint = str(config.get("checkpoint") or "")
    if not checkpoint:
        raise ValueError(f"[{dataset_name}] checkpoint is required")

    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(str(config.get("device", "cuda")))
    output_root = Path(config.get("output_dir", default_output_root()))
    output_dir = output_root / dataset_key
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    topk_values = [int(k) for k in config.get("topk", [10, 20, 50, 100])]
    topk_values = sorted(set(topk_values))
    feature_dtype = str(config.get("feature_dtype", "float16"))
    pair_batch_size = int(config.get("pair_batch_size", 512))

    repo_args = make_repo_args(config)
    from datasets import build_dataloader
    from model import build_model

    test_img_loader, test_txt_loader, num_classes = build_dataloader(repo_args)
    model = build_model(repo_args, num_classes)
    load_checkpoint(model, checkpoint)
    model.to(device)
    if device.type == "cpu":
        model.float()
    freeze_model(model)
    verify_frozen(model)
    label = run_label(dataset_name, model_preset)
    print(f"[{label}] checkpoint loaded; model frozen")

    extractor = GlobalHiddenExtractor(model, device=device, feature_dtype=feature_dtype)
    projection_info = inspect_clip_projection(model)

    official_global = compute_official_global_metrics(model, test_img_loader, test_txt_loader)
    diagnostic_global = extractor.extract_global_features(test_txt_loader, test_img_loader)
    diagnostic_similarity = compute_global_similarity(
        diagnostic_global.qfeats,
        diagnostic_global.gfeats,
    )
    diagnostic_metrics = compute_retrieval_metrics(
        diagnostic_similarity,
        diagnostic_global.qids,
        diagnostic_global.gids,
    )

    tolerance = float(config.get("fidelity_tolerance", 1e-4))
    fidelity = compare_metrics(official_global, diagnostic_metrics, tolerance)
    if not fidelity["passed"]:
        raise RuntimeError(
            f"[{label}] global fidelity FAILED: diffs={fidelity}, "
            f"official={official_global}, diagnostic={diagnostic_metrics}"
        )
    print(
        f"[{label}] global fidelity: PASS | "
        f"R@1={official_global['r1']:.2f} R@5={official_global['r5']:.2f} "
        f"R@10={official_global['r10']:.2f} mAP={official_global['map']:.2f}"
    )

    decisions = build_global_decisions(
        diagnostic_similarity,
        diagnostic_global.qids,
        diagnostic_global.gids,
        topk_values,
    )

    hidden = load_or_extract_hidden(
        extractor,
        test_txt_loader,
        test_img_loader,
        cache_dir,
        config,
        dataset_name,
        model_preset,
        checkpoint,
        force=bool(config.get("force_recompute", False)),
    )
    print(f"[{label}] hidden features cached")

    scorer = HiddenScorer(device)
    query_indices = torch.arange(diagnostic_global.qids.numel(), dtype=torch.long)
    fixed_pos_scores = scorer.score_pairs(
        hidden.text_features,
        hidden.text_mask,
        hidden.image_features,
        query_indices,
        decisions["best_positive_index"],
        batch_size=pair_batch_size,
    )
    fixed_neg_scores = scorer.score_pairs(
        hidden.text_features,
        hidden.text_mask,
        hidden.image_features,
        query_indices,
        decisions["best_negative_index"],
        batch_size=pair_batch_size,
    )
    delta_hidden = (fixed_pos_scores - fixed_neg_scores).float()
    delta_global = decisions["delta_global"].float()
    categories = assign_categories(delta_global, delta_hidden)

    max_k = min(max(topk_values), decisions["topk_indices"].shape[1])
    flat_q = torch.arange(diagnostic_global.qids.numel(), dtype=torch.long).repeat_interleave(max_k)
    flat_g = decisions["topk_indices"][:, :max_k].reshape(-1).long()
    hidden_topk_scores = scorer.score_pairs(
        hidden.text_features,
        hidden.text_mask,
        hidden.image_features,
        flat_q,
        flat_g,
        batch_size=pair_batch_size,
    ).view(diagnostic_global.qids.numel(), max_k)

    topk_summary = summarize_topk(
        topk_values,
        decisions["topk_indices"],
        hidden_topk_scores,
        diagnostic_global.qids,
        diagnostic_global.gids,
        delta_global.numpy(),
        decisions["positive_in_topk"],
    )

    fixed_summary = summarize_fixed_pair(delta_global.numpy(), delta_hidden.numpy())
    oracle_summary = summarize_oracle(delta_global.numpy(), delta_hidden.numpy())

    bootstrap_cfg = config.get("bootstrap", {}) or {}
    repetitions = int(bootstrap_cfg.get("repetitions", 2000))
    confidence = float(bootstrap_cfg.get("confidence", 0.95))
    ci = fixed_pair_ci_metrics(
        diagnostic_global.qids.numpy(),
        delta_global.numpy(),
        delta_hidden.numpy(),
        repetitions,
        confidence,
        seed,
    )
    fixed_summary.update({
        "recovery_rate_ci95": ci["recovery_rate_ci95"],
        "harm_rate_ci95": ci["harm_rate_ci95"],
    })
    oracle_summary["complementarity_gap_ci95"] = ci["complementarity_gap_ci95"]

    for k in topk_values:
        kkey = str(int(k))
        topk_summary[kkey]["rescue_at_k_ci95"] = rescue_ci(
            diagnostic_global.qids.numpy(),
            delta_global.numpy(),
            decisions["positive_in_topk"][int(k)].numpy(),
            topk_summary[kkey]["hidden_correct"].numpy(),
            repetitions,
            confidence,
            seed + int(k),
        )

    query_csv_path = output_dir / "query_diagnostics.csv"
    write_query_csv(
        query_csv_path,
        diagnostic_global.qids,
        diagnostic_global.captions,
        decisions,
        fixed_pos_scores,
        fixed_neg_scores,
        categories,
        topk_summary,
        topk_values,
    )

    serializable_topk = strip_tensor_values(topk_summary)
    summary = {
        "dataset": dataset_name,
        "model_preset": model_preset,
        "model_flags": {
            "only_global": bool(repo_args.only_global),
            "return_all": bool(repo_args.return_all),
            "topk_type": str(repo_args.topk_type),
            "modify_k": bool(repo_args.modify_k),
        },
        "checkpoint": checkpoint,
        "num_queries": int(diagnostic_global.qids.numel()),
        "num_gallery": int(diagnostic_global.gids.numel()),
        "official_global": {
            "r1": official_global["r1"],
            "r5": official_global["r5"],
            "r10": official_global["r10"],
            "map": official_global["map"],
        },
        "diagnostic_global": diagnostic_metrics,
        "fidelity_passed": True,
        "fidelity_diffs": fidelity,
        "clip_projection": projection_info,
        "fixed_pair": fixed_summary,
        "topk": serializable_topk,
        "oracle": oracle_summary,
        "seed": seed,
        "bootstrap": {
            "repetitions": repetitions,
            "confidence": confidence,
            "group_unit": "identity",
        },
    }

    write_summary_json(output_dir / "summary.json", summary)
    plot_margin_scatter(
        delta_global.numpy(),
        delta_hidden.numpy(),
        categories,
        output_dir / "margin_scatter.pdf",
        f"{dataset_name} ({model_preset}): Global vs Hidden Margins",
    )
    write_recovery_funnel(
        output_dir / "recovery_funnel.txt",
        int(diagnostic_global.qids.numel()),
        fixed_summary,
        serializable_topk.get("50", {}),
    )

    recovery_ci = fixed_summary["recovery_rate_ci95"]
    print(
        f"[{label}] fixed-pair recovery: "
        f"{100.0 * fixed_summary['recovery_rate']:.1f}% "
        f"[{100.0 * recovery_ci[0]:.1f}, {100.0 * recovery_ci[1]:.1f}]"
    )
    if "50" in serializable_topk:
        print(f"[{label}] Rescue@50: {100.0 * serializable_topk['50']['rescue_at_k']:.1f}%")
    print(f"[{label}] complementarity gap: {100.0 * oracle_summary['complementarity_gap']:.1f}%")
    print(f"[{label}] outputs -> {output_dir}")

    return summary


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def apply_cli_overrides(config: Dict[str, object], cli: argparse.Namespace) -> Dict[str, object]:
    config = dict(config)
    if cli.checkpoint is not None:
        config["checkpoint"] = cli.checkpoint
    if cli.dataset is not None:
        config["dataset"] = cli.dataset
    if cli.root_dir is not None:
        config["root_dir"] = cli.root_dir
    if cli.model_preset is not None:
        config["model_preset"] = cli.model_preset
    if cli.output_dir is not None:
        config["output_dir"] = cli.output_dir
    if cli.batch_size is not None:
        config["batch_size"] = cli.batch_size
    if cli.device is not None:
        config["device"] = cli.device
    if cli.cache_features:
        config["cache_features"] = True
    if cli.force_recompute:
        config["force_recompute"] = True
    if cli.seed is not None:
        config["seed"] = cli.seed
    return config


def make_repo_args(config: Mapping[str, object]) -> Namespace:
    values = dict(DEFAULT_REPO_ARGS)

    train_config = config.get("train_config")
    if train_config:
        values.update(read_yaml(Path(str(train_config))))

    checkpoint = str(config.get("checkpoint") or "")
    if checkpoint:
        auto_train_config = Path(checkpoint).resolve().parent / "configs.yaml"
        if auto_train_config.exists():
            values.update(read_yaml(auto_train_config))

    values["dataset_name"] = str(config.get("dataset", values["dataset_name"]))
    values["root_dir"] = str(config.get("root_dir", values["root_dir"]))
    values["training"] = False
    batch_size = int(config.get("batch_size", values["test_batch_size"]))
    values["batch_size"] = batch_size
    values["test_batch_size"] = batch_size
    values["num_workers"] = int(config.get("num_workers", values["num_workers"]))
    if "img_size" in config:
        values["img_size"] = tuple(config["img_size"])
    if "text_length" in config:
        values["text_length"] = int(config["text_length"])
    if "pretrain_choice" in config:
        values["pretrain_choice"] = config["pretrain_choice"]
    if "stride_size" in config:
        values["stride_size"] = int(config["stride_size"])

    apply_model_preset(values, config.get("model_preset", "clip"))

    for key, value in (config.get("model_flags") or {}).items():
        if key in {"only_global", "return_all", "modify_k"}:
            values[key] = bool(value)
        elif key == "topk_type":
            values[key] = str(value)
        else:
            values[key] = value

    for key in ("only_global", "return_all", "topk_type", "modify_k"):
        if key in config:
            if key in {"only_global", "return_all", "modify_k"}:
                values[key] = bool(config[key])
            else:
                values[key] = str(config[key])

    return Namespace(**values)


def apply_model_preset(values: Dict[str, object], model_preset: object) -> None:
    preset = normalize_model_preset(model_preset)
    if preset == "clip":
        values["only_global"] = True
        values["return_all"] = False
        values["topk_type"] = "mean"
        values["modify_k"] = False
    elif preset == "itself":
        values["only_global"] = False
        values["return_all"] = True
        values["topk_type"] = "custom"
        values["modify_k"] = True
    else:
        raise ValueError(f"Unsupported model_preset: {model_preset}")


def normalize_model_preset(model_preset: object) -> str:
    preset = str(model_preset or "clip").strip().lower()
    aliases = {
        "global": "clip",
        "only_global": "clip",
        "only-global": "clip",
        "clip_global": "clip",
        "clip-global": "clip",
        "itself_global_grab": "itself",
        "global_grab": "itself",
    }
    return aliases.get(preset, preset)


def read_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "img_size" in data and isinstance(data["img_size"], list):
        data["img_size"] = tuple(data["img_size"])
    return data


def load_checkpoint(model, checkpoint_path: str) -> None:
    from utils.checkpoint import load_state_dict

    checkpoint = safe_torch_load(checkpoint_path, map_location=torch.device("cpu"))
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    load_state_dict(model, state_dict)


def verify_frozen(model) -> None:
    model.eval()
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Model is not fully frozen")


def compute_official_global_metrics(model, img_loader, txt_loader) -> Dict[str, float]:
    from utils.metrics import Evaluator as OfficialEvaluator

    official = OfficialEvaluator(img_loader, txt_loader, Namespace(only_global=True))
    qfeats, gfeats, qids, gids = official._compute_embedding(model)
    similarity = compute_global_similarity(qfeats, gfeats)
    return compute_retrieval_metrics(similarity, qids, gids)


def load_or_extract_hidden(
    extractor: GlobalHiddenExtractor,
    txt_loader,
    img_loader,
    cache_dir: Path,
    config: Mapping[str, object],
    dataset_name: str,
    model_preset: str,
    checkpoint: str,
    force: bool = False,
):
    cache_path = cache_dir / "hidden_features.pt"
    use_cache = bool(config.get("cache_features", True))
    if use_cache and cache_path.exists() and not force:
        cached = safe_torch_load(cache_path, map_location="cpu")
        meta = cached.get("meta", {})
        if (
            meta.get("dataset") == dataset_name
            and meta.get("model_preset") == model_preset
            and meta.get("checkpoint") == checkpoint
        ):
            return hidden_from_dict(cached["features"])

    features = extractor.extract_hidden_features(txt_loader, img_loader)
    if use_cache:
        torch.save({
            "meta": {
                "dataset": dataset_name,
                "model_preset": model_preset,
                "checkpoint": checkpoint,
                "feature_dtype": features.feature_dtype,
                "num_queries": int(features.text_features.shape[0]),
                "num_gallery": int(features.image_features.shape[0]),
            },
            "features": hidden_to_dict(features),
        }, cache_path)
    return features


def hidden_to_dict(features: HiddenFeatureSet) -> Dict[str, object]:
    return {
        "text_features": features.text_features,
        "text_mask": features.text_mask,
        "token_ids": features.token_ids,
        "image_features": features.image_features,
        "patch_grid": list(features.patch_grid),
        "feature_dtype": features.feature_dtype,
    }


def hidden_from_dict(data: Mapping[str, object]) -> HiddenFeatureSet:
    return HiddenFeatureSet(
        text_features=data["text_features"],
        text_mask=data["text_mask"],
        token_ids=data["token_ids"],
        image_features=data["image_features"],
        patch_grid=tuple(data.get("patch_grid", (None, None))),
        feature_dtype=str(data.get("feature_dtype", "float16")),
    )


def safe_torch_load(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def strip_tensor_values(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): strip_tensor_values(v) for k, v in value.items() if k != "hidden_correct"}
    if isinstance(value, list):
        return [strip_tensor_values(v) for v in value]
    return value


def resolve_device(name: str) -> torch.device:
    name = name.lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_output_root() -> Path:
    return Path(__file__).resolve().parent / "outputs"


def run_label(dataset_name: str, model_preset: str) -> str:
    return f"{dataset_name}/{model_preset}"


def dataset_output_key(dataset_name: str) -> str:
    mapping = {
        "CUHK-PEDES": "cuhk",
        "ICFG-PEDES": "icfg",
        "RSTPReid": "rstp",
    }
    return mapping.get(dataset_name, dataset_name.lower().replace("-", "").replace("_", ""))


if __name__ == "__main__":
    main()
