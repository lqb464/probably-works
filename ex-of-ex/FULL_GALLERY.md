# Additive full-gallery experiments

The old diagnostic and validated_v2 suites are unchanged. Use the existing
run_experiments.py with --full-gallery-only to run this separate supplement.
It reuses compatible feature caches; it does not require old result files.

For a completely fresh session running everything in one invocation per checkpoint,
use --with-full-gallery instead of --full-gallery-only. This runs validated_v2 first,
then full_gallery, under the SAME timestamped directory/run.log. The two flags are
mutually exclusive. The complete fresh-session notebook is testing_all.ipynb beside
this document. It includes clone/install/download/preflight, all four checkpoints,
both result displays, and ZIP export. Source must first be published or uploaded
to Kaggle; cloning an older commit will fail the notebook's CLI preflight check.

## Questions and rows

- global: official cosine retrieval over all gallery images.
- special_token_control: independently extract final post-LayerNorm,
  post-projection text EOS and image CLS, then cosine retrieval. In this CLIP
  implementation these are the global vectors. special_token_audit.json checks
  equality; do not present equivalent results as new hidden information.
- hidden_full: MaxSim or uniform-mean content-token/patch similarity over ALL
  gallery images, with no global candidate filter or positive injection.
- rerank_topk: global selects K; choose alpha and K on selection, preserving all
  ranks outside the prefix. This supplements, not replaces, the existing v2 reranker.
- fusion_full: rank ALL gallery images by (1-alpha)*global + alpha*hidden.
  Alpha is the hidden weight. Alpha=0 is the no-op reference, alpha=1 hidden-only.

Local scorers reuse scorers.py (FILIP-inspired MaxSim and uniform-mean control).
No new learned token matcher or loss is introduced. Fusion is simple score-level
interpolation, not a reproduction of a new paper method. It uses raw score scales;
the older top-K fusion can use a different normalization and is not numerically
identical. Raw preprojection EOS/CLS are not compared: image and text spaces are
different and may have different widths. No arbitrary padding/projection is used.

Selection uses the same identity split seeds as v2; CUHK/RSTP official validation,
ICFG explicit 20% development / 80% evaluation identity holdout with full original
gallery. Only the selection partition tunes fusion; unused fit/calibration IDs
remain recorded. Test reports each predeclared hidden scorer and each independently
selection-chosen family. Do not choose a winner by test. Paired identity bootstrap
uses 2000 draws, 95% percentile CIs conditional on fixed model/gallery/selection,
not simultaneous intervals or a fresh confirmation on already inspected tests.

## Kaggle cells after fresh-session setup

First use the same clone, dependency installation and checkpoint-download cells
as before, with the updated source containing full_gallery.py. Then run:

```python
import sys, subprocess
from pathlib import Path
REPO = Path('/kaggle/working/probably-works')
ROOT = '/kaggle/input/datasets/hoanggv/tbps-benchmark/benchmark'
MODEL_DIR = Path('/kaggle/input/models/hoanggv/clip-baseline/pytorch/default/1')
jobs = [
    ('rstp', REPO/'rstp.pth', 'rstp-gdown-full-gallery'),
    ('icfg', MODEL_DIR/'icfg.pth', 'icfg-full-gallery'),
    ('cuhk', MODEL_DIR/'cuhk.pth', 'cuhk-full-gallery'),
    ('rstp', Path('/kaggle/input/models/hoanggvo/tbps-clip-bs256/pytorch/default/1/best.pth'), 'rstp-bs256-full-gallery'),
]
failed = []
for dataset, checkpoint, name in jobs:
    assert checkpoint.is_file(), checkpoint
    result = subprocess.run([
        sys.executable, '-u', 'ex-of-ex/run_experiments.py',
        '--config', f'ex-of-ex/configs/{dataset}_validated.yaml',
        '--root_dir', ROOT, '--checkpoint', str(checkpoint),
        '--model-preset', 'clip', '--full-gallery-only',
        '--run-name', name,
    ], cwd=REPO)
    if result.returncode:
        failed.append(name)
assert not failed, failed
```

Every run has its own timestamped output directory and run.log, including errors.
Start with one RSTP checkpoint to measure runtime. Full-gallery scoring is much
more expensive than top-50: it scores N_queries*N_gallery pairs, including selection.
Memory is bounded by --pair-batch-size (default 384); reduce this if GPU OOM.
The default supplement only scores MaxSim and uniform mean, not all nine methods.

Add --save-full-similarity if full float32 matrices are needed. This writes test
global, projected EOS/CLS, MaxSim and uniform-mean .npy arrays. Their rows follow
test__index.npz query_index/qids; columns follow gids. Cost is about
4 methods * 4 bytes * N_test_queries * N_gallery. For 15879*19948 this is about
5.1 GB decimal. Files from an interrupted run can be incomplete; only use them
when summary.json exists. Without this option, metrics and top-1 IDs are still saved.

```python
import pandas as pd
from IPython.display import display
for p in sorted((REPO/'ex-of-ex/outputs').glob('*/*__*full-gallery/full_gallery/test_summary.csv')):
    print(p.parent.parent.name)
    display(pd.read_csv(p)[['family','scorer','alpha','k','r1','map',
        'delta_r1','delta_r1_ci_low','delta_r1_ci_high','recovery_rate','harm_rate']])
```

Outputs: protocol.json, special_token_audit.json, split_assignments.csv,
selection_grid.csv, selected_before_test.json, test_summary.csv,
test_query_metrics.npz, test_top1.csv, index NPZs, optional similarity matrices,
and final summary.json completion marker. Keep the usual ZIP/download cell.
