"""Run protocol v2 directly from the feature caches produced by run_experiments.

No checkpoint load, model download or dataset loader required. Uses exactly the
same v2 suite as the main runner. Only use trusted feature caches from your runs.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import torch
import yaml

from diagnostic.global_hidden_recovery.feature_extractor import GlobalFeatureSet, HiddenFeatureSet
from validated import make_split_scores, write_json
from protocol_v2 import run_suite_v2
from scorers import build_specs


def load_cache(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    g, h = payload["global"], payload["hidden"]
    if h["text_features"].ndim != 3 or h["image_features"].ndim != 3:
        raise ValueError("Expected ex-of-ex final-layer cache (3D), not a multilayer diagnostic cache")
    return (GlobalFeatureSet(**{key:g[key] for key in ["qfeats","gfeats","qids","gids","captions"]}),
            HiddenFeatureSet(**{key:h[key] for key in ["text_features","text_mask","token_ids","image_features","patch_grid","feature_dtype"]}),
            payload["meta"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="cached-validated-v2")
    parser.add_argument("--output-root", type=Path, default=HERE/"outputs")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in {".",".."}:
        parser.error("run-name must be a single directory name")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config.setdefault("validated", {})["protocol_version"] = 2
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    key = {"RSTPReid":"rstp","CUHK-PEDES":"cuhk","ICFG-PEDES":"icfg"}[config["dataset"]]
    run = args.output_root/key/(datetime.now().strftime("%Y%m%d-%H%M%S")+"__"+args.run_name)
    run.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(run/"run.log", encoding="utf-8"), logging.StreamHandler()])
    specs = build_specs(config["scorers"])
    try:
        states, metas = {}, {}
        for name, path in [("test",args.test_cache),("validation",args.validation_cache)]:
            if path is None:
                states[name] = None
                continue
            logging.info("Loading %s cache %s", name, path)
            g, h, meta = load_cache(path)
            metas[name] = meta
            if meta.get("dataset") not in {None, config["dataset"]}:
                raise ValueError("Cache dataset differs from config")
            states[name] = make_split_scores(name, g, h, specs, config["topk"], device,
                                              int(config.get("pair_batch_size",384)))
        if "validation" in metas:
            # Ignore only the split label; all checkpoint/extraction metadata must match.
            def common(meta):
                return {k:v for k,v in meta.items() if k != "split"}
            if common(metas["test"]) != common(metas["validation"]):
                raise ValueError("Test/validation cache extraction metadata differ")
        summary = run_suite_v2(config, run, specs, states["validation"], states["test"], device,
                               int(config.get("pair_batch_size",384)))
        write_json(run/"manifest.json", {"config":config, "feature_cache_metadata":metas,
                   "test_cache":str(args.test_cache.resolve()),
                   "validation_cache":str(args.validation_cache.resolve()) if args.validation_cache else None,
                   "device":str(device), "torch":torch.__version__, "validated":summary["protocol"]})
    except Exception:
        logging.exception("Cached v2 run failed")
        raise


if __name__ == "__main__":
    main()
