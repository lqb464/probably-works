# ITSELF Global-Hidden Recovery Diagnostic

This diagnostic asks whether global retrieval failures made by a frozen ITSELF
checkpoint can be reversed using the final non-pooled CLIP token and patch
representations from the same checkpoint.

It is analysis-only. It does not train, fine-tune, add losses, or edit
checkpoints.

## Run

```bash
python diagnostic/global_hidden_recovery/run_diagnostic.py \
  --config diagnostic/global_hidden_recovery/configs/cuhk.yaml \
  --checkpoint /path/to/cuhk/best.pth
```

Multiple configs can be run with one command:

```bash
python diagnostic/global_hidden_recovery/run_diagnostic.py \
  --config diagnostic/global_hidden_recovery/configs/cuhk.yaml \
           diagnostic/global_hidden_recovery/configs/icfg.yaml \
           diagnostic/global_hidden_recovery/configs/rstp.yaml
```

CLI overrides include:

```text
--config
--checkpoint
--dataset
--output-dir
--batch-size
--device
--cache-features
--force-recompute
--seed
--verbose
```

The diagnostic forces `only_global=True` because the official global CLIP path is
the baseline under test.

## Outputs

Each run writes:

```text
outputs/<dataset>/
├── summary.json
├── query_diagnostics.csv
├── margin_scatter.pdf
├── recovery_funnel.txt
└── cache/
```

When multiple configs are provided, it also writes:

```text
outputs/combined_summary.csv
```

## Fidelity

Stage 0 compares the diagnostic global path against the repository evaluator
using the same model, dataloaders, feature normalization, similarity matrix, and
identity relevance rule. The run aborts if R@1, R@5, R@10, or mAP differs beyond
the configured tolerance.

## Hidden Scorer

The primary hidden scorer is parameter-free text-to-image MaxSim:

```text
score(Q, I) = mean_t max_p dot(text_token_t, image_patch_p)
```

Text hidden vectors exclude padding, start-of-text, and end-of-text. Image hidden
vectors exclude the visual CLS token. Both sides are projected through the
checkpoint's CLIP projection path and L2-normalized per token/patch.

