"""Seed 5 runner for RT-DETR training (RESULTS.md Section 38/39's
scoped replication) - thin override wrapper, same pattern as
train_yolo_seed5.py. See train_rtdetr.py for the actual pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_rtdetr as base

SEED = 5
base.SEED = SEED
base.SPLIT_FILE = base.base.REPO_ROOT / "cpu_repro" / "coord_baseline" / f"image_split_seed{SEED}.csv"
base.PREPARED_DIR = Path(__file__).resolve().parent / "prepared" / f"seed{SEED}"
base.RUN_NAME = f"rtdetr_seed{SEED}split"
base.EVAL_RESULTS_DIR = Path(__file__).resolve().parent / "eval_results" / "rtdetr" / f"seed{SEED}"


def main():
    data_yaml = base.prepare_dataset()
    base.train(data_yaml)
    best_weights = Path(base.PROJECT_DIR) / base.RUN_NAME / "weights" / "best.pt"
    base.evaluate(best_weights)


if __name__ == "__main__":
    main()
