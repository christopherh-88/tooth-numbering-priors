"""Seed 4 runner for Faster R-CNN training (RESULTS.md Section 44's
5-seed extension) - thin override wrapper, same pattern as
train_rtdetr_seed4.py. See train_fasterrcnn.py for the actual pipeline
and build_coco_dataset.py for the COCO-JSON data prep it now depends on
for a non-zero seed.

Seed 0 is NOT re-run here or by this file's siblings - Section 44 already
trained and evaluated it directly via train_fasterrcnn.py, and its
prepared_coco/instances_*.json + runs/fasterrcnn_seed0split/best.pt are
reused unmodified as the seed-0 data point in the 5-seed comparison.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_coco_dataset as bcd
import train_fasterrcnn as base

SEED = 4
bcd.SEED = SEED
bcd.SPLIT_FILE = bcd.REPO_ROOT / "cpu_repro" / "coord_baseline" / f"image_split_seed{SEED}.csv"
bcd.OUT_DIR = Path(__file__).resolve().parent / "prepared_coco" / f"seed{SEED}"

base.SEED = SEED
base.COCO_DIR = bcd.OUT_DIR
base.TRAIN_JSON = base.COCO_DIR / "instances_train.json"
base.VAL_JSON = base.COCO_DIR / "instances_val.json"
base.RUN_NAME = f"fasterrcnn_seed{SEED}split"
base.EVAL_RESULTS_DIR = Path(__file__).resolve().parent / "eval_results" / "fasterrcnn" / f"seed{SEED}"


def main():
    bcd.convert()
    bcd.verify()
    base.train()
    best_weights = base.PROJECT_DIR / base.RUN_NAME / "best.pt"
    base.evaluate(best_weights)


if __name__ == "__main__":
    main()
