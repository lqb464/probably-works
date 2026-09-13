# Kaggle cells — hai notebook text và image

Dùng cùng sáu cell dưới đây. Notebook text đặt `MODALITY = "text"`; notebook image đặt `MODALITY = "image"`. Các file `.ipynb` đã tách sẵn. Bật Internet, attach benchmark và checkpoint datasets, chọn GPU T4 x2.

## Cell 1

```python
from pathlib import Path
import os, subprocess, sys

REPO = Path("/kaggle/working/probably-works")
URL = "https://github.com/lqb464/probably-works.git"
BRANCH = "codex/clip-layer-probe"
if not REPO.exists():
    subprocess.run(["git", "clone", "--branch", BRANCH, URL, str(REPO)], check=True)
else:
    subprocess.run(["git", "-C", str(REPO), "fetch", URL, BRANCH], check=True)
    exists = subprocess.run(["git", "-C", str(REPO), "show-ref", "--verify", "--quiet", "refs/heads/" + BRANCH]).returncode == 0
    checkout = ["checkout", BRANCH] if exists else ["checkout", "-b", BRANCH, "FETCH_HEAD"]
    subprocess.run(["git", "-C", str(REPO), *checkout], check=True)
    subprocess.run(["git", "-C", str(REPO), "merge", "--ff-only", "FETCH_HEAD"], check=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "layer_probe/requirements.txt"], check=True)
subprocess.run([sys.executable, "layer_probe/tests.py"], check=True)
```

## Cell 2

```python
import torch, json
from datetime import datetime

MODALITY = "text"  # notebook còn lại đổi thành "image"
assert torch.cuda.device_count() >= 2, "Chọn Accelerator: GPU T4 x2"
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

ROOT = "/kaggle/input/datasets/hoanggv/tbps-benchmark/benchmark"
CHECKPOINTS = {
    "rstp": "/kaggle/input/models/hoanggvo/tbps-clip-bs256/pytorch/default/1/best.pth",
    "cuhk": "/kaggle/input/models/hoanggv/clip-baseline/pytorch/default/1/cuhk.pth",
}
# Tuỳ chọn: ICFG dùng identity holdout riêng, KHÔNG phải official full test.
# CHECKPOINTS["icfg"] = "/kaggle/input/models/hoanggv/clip-baseline/pytorch/default/1/icfg.pth"

assert Path(ROOT).is_dir(), ROOT
for path in CHECKPOINTS.values():
    assert Path(path).is_file(), path
OUT = Path("/kaggle/working/layer-results") / (MODALITY + "-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
OUT.mkdir(parents=True, exist_ok=False)
jobs = []
for dataset, checkpoint in CHECKPOINTS.items():
    job = ["--modality", MODALITY, "--dataset", dataset, "--root", ROOT,
           "--checkpoint", checkpoint, "--output", str(OUT / dataset),
           "--batch-size", "32", "--workers", "2",
           "--image-size", "384", "128", "--mix-seeds", "42", "43", "44"]
    if dataset == "icfg":
        job += ["--allow-test-holdout"]
    jobs.append(job)
JOBS = OUT / "jobs.json"
JOBS.write_text(json.dumps(jobs, indent=2))
print("Output:", OUT)
```

## Cell 3

```python
# Hai dataset chạy song song, mỗi dataset trên một T4.
# Nếu chỉ giữ một dataset trong CHECKPOINTS thì chỉ dùng một GPU.
subprocess.run([sys.executable, "layer_probe/launch.py", "--jobs", str(JOBS), "--gpus", "2"], check=True)
```

## Cell 4

```python
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

for dataset in CHECKPOINTS:
    folder = OUT / dataset
    print("\n", dataset, MODALITY)
    chosen = json.loads((folder / "selection.json").read_text())
    print("Selection:", chosen["chosen"])
    result = pd.DataFrame(json.loads((folder / "test_results.json").read_text()))
    display(result[["name", "r1", "r5", "r10", "map", "minp"]])
    grid = pd.read_csv(folder / "validation_grid.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    for pool in ["global", "hidden"]:
        rows = grid[grid.name.str.startswith("ridge_" + pool + "_")].copy()
        rows["layer"] = rows.name.str.split("_").str[2].astype(int)
        curve = rows.groupby("layer").r1.max()
        ax.plot(curve.index, curve.values * 100, marker="o", label=pool)
    ax.set(xlabel="Residual block", ylabel="Selection R@1 (%)",
           title=dataset + " / " + MODALITY + " — validation curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "validation_layers.png", dpi=160)
    plt.show()
```

## Cell 5

```python
# Paired CI cho câu hỏi chính: best hidden layer so với final hidden layer.
import numpy as np
from layer_probe.core import paired_ci

for dataset in CHECKPOINTS:
    folder = OUT / dataset
    chosen = json.loads((folder / "selection.json").read_text())["chosen"]
    a, b = chosen["ridge_hidden"], chosen["ridge_final_hidden"]
    fa = torch.from_numpy(np.load(folder / ("queries_" + a + ".npy")))
    fb = torch.from_numpy(np.load(folder / ("queries_" + b + ".npy")))
    cache = torch.load(folder / "test_features.pt", map_location="cpu", weights_only=True)
    comparison = {"selected": a, "final": b,
                  "delta_r1": float((fa[:, 0] - fb[:, 0]).mean()),
                  "ci95": paired_ci(fa, fb, cache["text"]["ids"])}
    (folder / "intermediate_vs_final.json").write_text(json.dumps(comparison, indent=2))
    print(dataset, comparison)
    del cache
```

## Cell 6

```python
# Archive gọn: bỏ pooled feature caches; adapters, metrics, IDs, plots và logs vẫn giữ.
import zipfile
from IPython.display import FileLink

archive = OUT.with_suffix(".zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for path in OUT.rglob("*"):
        if path.is_file() and not path.name.endswith("_features.pt"):
            z.write(path, path.relative_to(OUT.parent))
display(FileLink(str(archive)))
```
