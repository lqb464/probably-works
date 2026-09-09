# Kaggle cells cho bốn thí nghiệm validation-selected

Mỗi cell chạy dataset/checkpoint dưới đây tự thực hiện cả setwise diagnostic,
controlled probe, top-K reranking và selective gating. Không cần bốn cell riêng cho
bốn loại thí nghiệm.

## Cell 1 — cập nhật code và dependency

```python
%cd /kaggle/working

import os

REPO = "/kaggle/working/probably-works"
if not os.path.isdir(os.path.join(REPO, ".git")):
    !git clone --branch diagnostic https://github.com/lqb464/probably-works.git "$REPO"

%cd {REPO}
!git remote set-url origin https://github.com/lqb464/probably-works.git
!git checkout diagnostic
!git pull --ff-only origin diagnostic
!pip install -q -r requirements.txt
```

## Cell 2 — khai báo đường dẫn và kiểm tra GPU

```python
ROOT = "/kaggle/input/datasets/hoanggv/tbps-benchmark/benchmark"
MODEL_DIR = "/kaggle/input/models/hoanggv/clip-baseline/pytorch/default/1"
RSTP_BS256 = "/kaggle/input/models/hoanggvo/tbps-clip-bs256/pytorch/default/1/best.pth"

import os
import torch

assert os.path.isdir(ROOT), ROOT
assert os.path.isfile(f"{MODEL_DIR}/icfg.pth")
assert os.path.isfile(f"{MODEL_DIR}/cuhk.pth")
assert os.path.isfile(RSTP_BS256)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## Cell 3 — RSTP checkpoint `rstp.pth`

Cell này giả sử cell download cũ của notebook đã tạo
`/kaggle/working/probably-works/rstp.pth`. Nếu file nằm chỗ khác, chỉ sửa biến
`RSTP_GDOWN`.

```python
RSTP_GDOWN = "/kaggle/working/probably-works/rstp.pth"
assert os.path.isfile(RSTP_GDOWN), RSTP_GDOWN

!python ex-of-ex/run_experiments.py \
  --config ex-of-ex/configs/rstp_validated.yaml \
  --root_dir "$ROOT" \
  --checkpoint "$RSTP_GDOWN" \
  --model-preset clip \
  --run-name rstp-gdown-validated
```

## Cell 4 — ICFG

```python
!python ex-of-ex/run_experiments.py \
  --config ex-of-ex/configs/icfg_validated.yaml \
  --root_dir "$ROOT" \
  --checkpoint "$MODEL_DIR/icfg.pth" \
  --model-preset clip \
  --run-name icfg-validated
```

## Cell 5 — CUHK

```python
!python ex-of-ex/run_experiments.py \
  --config ex-of-ex/configs/cuhk_validated.yaml \
  --root_dir "$ROOT" \
  --checkpoint "$MODEL_DIR/cuhk.pth" \
  --model-preset clip \
  --run-name cuhk-validated
```

## Cell 6 — RSTP batch-size 256 checkpoint

```python
!python ex-of-ex/run_experiments.py \
  --config ex-of-ex/configs/rstp_validated.yaml \
  --root_dir "$ROOT" \
  --checkpoint "$RSTP_BS256" \
  --model-preset clip \
  --run-name rstp-bs256-validated
```

## Cell 7 — gom và xem các kết quả chính

```python
from pathlib import Path
import pandas as pd

output_root = Path("/kaggle/working/probably-works/ex-of-ex/outputs")
for run in sorted(output_root.glob("*/*__*-validated")):
    validated = run / "validated"
    print("\n===", run.name, "===")

    coverage = pd.read_csv(validated / "coverage.csv")
    print("\nCoverage ceiling")
    display(coverage)

    setwise = pd.read_csv(validated / "setwise_summary.csv")
    print("\nBest setwise recovery, 10 hard negatives")
    display(
        setwise[setwise.num_hard_negatives == 10]
        .sort_values(["recovery_rate", "harm_rate"], ascending=[False, True])
        .head(5)
    )

    probes = pd.read_csv(validated / "probe_results.csv")
    print("\nGlobal + hidden probes")
    display(
        probes[probes.probe == "global_plus_hidden"]
        .sort_values("roc_auc", ascending=False)
        .head(5)
    )

    print("\nValidation-selected reranker on test")
    display(pd.read_csv(validated / "reranking_selected_test.csv"))

    print("\nSelective gate")
    display(pd.read_csv(validated / "gating_summary.csv"))
```

## Cell 8 — nén output để tải về

```python
%cd /kaggle/working/probably-works
!zip -qr /kaggle/working/validated-results.zip ex-of-ex/outputs
print("Download: /kaggle/working/validated-results.zip")
```

Feature cache nằm trong `ex-of-ex/cache/<dataset>/clip/<checkpoint-hash>/`.
Nếu rerun cùng checkpoint và config, cả test lẫn validation features sẽ được load
lại từ cache. Không dùng `--force-recompute` trừ khi checkpoint hoặc feature policy
đã thay đổi.
