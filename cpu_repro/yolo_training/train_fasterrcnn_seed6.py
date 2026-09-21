"""Seed 6 runner for Faster R-CNN training - extends the 5-seed
replication (RESULTS.md Section 45) with one additional seed, run
locally on this machine's Apple Silicon GPU (MPS) rather than Kaggle,
since Kaggle GPU access was unavailable when this was scoped. Same
thin-wrapper pattern as train_fasterrcnn_seed1.py, plus TRAIN_DEVICE
overridden to "mps".

Not part of the pre-registered 5-seed replication (Sections 32/45) -
seeds 0-4 are that fixed, already-analyzed set. This is an extra,
sixth data point requested afterward to further tighten the CIs, using
a different hardware backend than every prior training run in this
project (Kaggle CUDA). That's a real difference worth disclosing
alongside whatever result comes out of this, not silently folded in as
if it were seed 6 of the original set.

Pause/resume: kill and rerun this script freely - train_fasterrcnn.py's
train() reloads model/optimizer/scheduler/epoch state from
runs/fasterrcnn_seed6split/last.pt if it exists, so nothing is lost
beyond whatever epoch was in progress at kill time.

Usage (local, MPS): python train_fasterrcnn_seed6.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_coco_dataset as bcd
import train_fasterrcnn as base

SEED = 6
bcd.SEED = SEED
bcd.SPLIT_FILE = bcd.REPO_ROOT / "cpu_repro" / "coord_baseline" / f"image_split_seed{SEED}.csv"
bcd.OUT_DIR = Path(__file__).resolve().parent / "prepared_coco" / f"seed{SEED}"

base.SEED = SEED
base.COCO_DIR = bcd.OUT_DIR
base.TRAIN_JSON = base.COCO_DIR / "instances_train.json"
base.VAL_JSON = base.COCO_DIR / "instances_val.json"
base.RUN_NAME = f"fasterrcnn_seed{SEED}split"
base.EVAL_RESULTS_DIR = Path(__file__).resolve().parent / "eval_results" / "fasterrcnn" / f"seed{SEED}"
base.TRAIN_DEVICE = "mps"


def main():
    bcd.convert()
    bcd.verify()
    base.train()
    best_weights = base.PROJECT_DIR / base.RUN_NAME / "best.pt"
    base.evaluate(best_weights)


if __name__ == "__main__":
    main()
