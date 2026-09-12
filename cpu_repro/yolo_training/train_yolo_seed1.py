"""Multi-seed replication (RESULTS.md Section 32): seed 1 arm.

Not a copy-paste-diverged fork - imports train_yolo.py's own
prepare_yolo_dataset()/train()/evaluate() unchanged and overrides only the
seed-derived constants (SPLIT_FILE, PREPARED_DIR, EVAL_RESULTS_DIR,
RUN_NAME) on the shared module object, exactly as train_yolo_zerojitter.py
already does for TRANSLATE/SCALE/RUN_NAME/EVAL_RESULTS_DIR. Every other
hyperparameter (epochs, batch, imgsz, dropout, translate=0.1/scale=0.5 -
i.e. the same jittered condition as Section 21, TRANSLATE/SCALE are not
touched here - and the fliplr=0.0 fix) is read from train_yolo directly,
so this run differs from Section 21 only in which persisted image-level
split it trains/evaluates on.

Usage (on Kaggle GPU): python train_yolo_seed1.py
"""

from pathlib import Path

import train_yolo as base

SEED = 1
base.SEED = SEED
base.SPLIT_FILE = base.REPO_ROOT / "cpu_repro" / "coord_baseline" / f"image_split_seed{SEED}.csv"
base.PREPARED_DIR = Path(__file__).resolve().parent / "prepared" / f"seed{SEED}"
base.EVAL_RESULTS_DIR = Path(__file__).resolve().parent / "eval_results" / f"seed{SEED}"
base.RUN_NAME = f"yolov8_seed{SEED}split"


def main():
    data_yaml = base.prepare_yolo_dataset()
    base.train(data_yaml)
    best_weights = Path(base.PROJECT_DIR) / base.RUN_NAME / "weights" / "best.pt"
    base.evaluate(best_weights)


if __name__ == "__main__":
    main()
