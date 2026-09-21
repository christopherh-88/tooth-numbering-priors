"""Seed 5 runner for RT-DETR-l on local Apple Silicon (MPS). Same image split
as the Kaggle CUDA runner train_rtdetr_seed5.py (image_split_seed5.csv). Thin
override wrapper around train_rtdetr.py; differs from the CUDA runner only in:
  - DEVICE = "mps"
  - BATCH = 4 instead of 10 (RT-DETR-l at batch 4 uses ~5.5 GB on MPS in a
    smoke test; batch 10 would need ~14 GB on a 16 GB Mac). Ultralytics
    accumulates gradients to a nominal batch of 64, so the optimizer update
    size is nearly unchanged, but batch-norm statistics differ - a second
    disclosed difference besides the backend.
  - separate run/eval directories (rtdetr_mps_seed5split,
    eval_results/rtdetr_mps/seed5) so it never overwrites CUDA results.

Pause/resume: kill and rerun freely; train_rtdetr.py resumes from
runs/rtdetr_mps_seed5split/weights/last.pt.

Usage (local): python train_rtdetr_mps_seed5.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_rtdetr as base

SEED = 5
base.SEED = SEED
base.SPLIT_FILE = base.base.REPO_ROOT / "cpu_repro" / "coord_baseline" / f"image_split_seed{SEED}.csv"
base.PREPARED_DIR = Path(__file__).resolve().parent / "prepared" / f"seed{SEED}"
base.RUN_NAME = f"rtdetr_mps_seed{SEED}split"
base.EVAL_RESULTS_DIR = Path(__file__).resolve().parent / "eval_results" / "rtdetr_mps" / f"seed{SEED}"
base.DEVICE = "mps"
base.BATCH = 4


def main():
    data_yaml = base.prepare_dataset()
    base.train(data_yaml)
    best_weights = Path(base.PROJECT_DIR) / base.RUN_NAME / "weights" / "best.pt"
    base.evaluate(best_weights)


if __name__ == "__main__":
    main()
