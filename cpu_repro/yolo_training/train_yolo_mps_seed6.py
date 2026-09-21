"""Seed 6 runner for YOLOv8x on local Apple Silicon (MPS). Same image split
as the Kaggle CUDA runner train_yolo_seed6.py (image_split_seed6.csv), so the
two backends can be compared split-for-split. Thin override wrapper around
train_yolo.py; differs from the CUDA runner only in:
  - DEVICE = "mps"
  - BATCH = 2 instead of 10 (YOLOv8x at batch 10 needs ~15.6 GB and swaps on
    a 16 GB Mac, 607 s/step in a smoke test; batch 2 uses ~4.8 GB at ~1 s/step).
    Ultralytics accumulates gradients to a nominal batch of 64, so the
    optimizer update size is nearly unchanged, but batch-norm statistics
    differ - a second disclosed difference besides the backend.
  - separate run/eval directories (yolov8_mps_seed6split,
    eval_results/yolo_mps/seed6) so it never overwrites CUDA results.

Pause/resume: kill and rerun freely; train_yolo.py resumes from
runs/yolov8_mps_seed6split/weights/last.pt.

Usage (local): python train_yolo_mps_seed6.py
"""

from pathlib import Path

import train_yolo as base

SEED = 6
base.SEED = SEED
base.SPLIT_FILE = base.REPO_ROOT / "cpu_repro" / "coord_baseline" / f"image_split_seed{SEED}.csv"
base.PREPARED_DIR = Path(__file__).resolve().parent / "prepared" / f"seed{SEED}"
base.EVAL_RESULTS_DIR = Path(__file__).resolve().parent / "eval_results" / "yolo_mps" / f"seed{SEED}"
base.RUN_NAME = f"yolov8_mps_seed{SEED}split"
base.DEVICE = "mps"
base.BATCH = 2


def main():
    data_yaml = base.prepare_yolo_dataset()
    base.train(data_yaml)
    best_weights = Path(base.PROJECT_DIR) / base.RUN_NAME / "weights" / "best.pt"
    base.evaluate(best_weights)


if __name__ == "__main__":
    main()
