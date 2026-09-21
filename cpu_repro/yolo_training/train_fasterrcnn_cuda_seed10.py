"""Seed 10 runner for Faster R-CNN on Kaggle CUDA. Split: image_split_seed10.csv.
Thin override wrapper, same pattern as train_fasterrcnn_seed4.py, using the
default TRAIN_DEVICE="cuda" and writing to fasterrcnn_cuda_seed10split and
eval_results/fasterrcnn_cuda/seed10.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_coco_dataset as bcd
import train_fasterrcnn as base

SEED = 10
bcd.SEED = SEED
bcd.SPLIT_FILE = bcd.REPO_ROOT / "cpu_repro" / "coord_baseline" / f"image_split_seed{SEED}.csv"
bcd.OUT_DIR = Path(__file__).resolve().parent / "prepared_coco" / f"seed{SEED}"

base.SEED = SEED
base.COCO_DIR = bcd.OUT_DIR
base.TRAIN_JSON = base.COCO_DIR / "instances_train.json"
base.VAL_JSON = base.COCO_DIR / "instances_val.json"
base.RUN_NAME = f"fasterrcnn_cuda_seed{SEED}split"
base.EVAL_RESULTS_DIR = Path(__file__).resolve().parent / "eval_results" / "fasterrcnn_cuda" / f"seed{SEED}"


def main():
    bcd.convert()
    bcd.verify()
    base.train()
    best_weights = base.PROJECT_DIR / base.RUN_NAME / "best.pt"
    base.evaluate(best_weights)


if __name__ == "__main__":
    main()
