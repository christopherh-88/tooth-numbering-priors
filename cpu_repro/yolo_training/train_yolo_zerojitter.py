"""Mitigation experiment, near-zero-jitter comparison arm.

See train_yolo.py's TRANSLATE/SCALE comment block for why this script
exists: Section 21's run (train_yolo.py, unmodified) used Ultralytics'
default translate=0.1/scale=0.5, which already exceeds or matches the
mitigation design's own pre-registered "2x natural positional spread"
jitter target (cpu_repro/coord_baseline/mitigation/README.md) - so it is
correctly read as the *jittered* condition, not a jitter-free baseline.
This script is the actual near-zero-jitter counterpart: everything else
(epochs, batch, imgsz, dropout, fliplr fix, split, evaluation protocol)
is identical to train_yolo.py, so TRANSLATE/SCALE is the only intended
difference between the two runs' results.

Not a copy-paste-diverged fork of the pipeline logic - imports every
function from train_yolo.py unchanged except train(), which is
reimplemented here only to swap in TRANSLATE=0/SCALE=0 and a distinct
RUN_NAME/output directory so this run can never overwrite Section 21's
checkpoint or eval_results/summary.csv.

Usage (same as train_yolo.py, on Kaggle GPU):

    python train_yolo_zerojitter.py
"""

from pathlib import Path

import train_yolo as base
from train_yolo import (  # noqa: F401
    get_or_create_model,
    prepare_yolo_dataset,
    evaluate as base_evaluate,
)

# Override just the two augmentation knobs and the run identity - everything
# else (EPOCHS, BATCH, IMGSZ, DEVICE, DROPOUT, CLOSE_MOSAIC, COS_LR,
# WARMUP_EPOCHS, LRF, SINGLE_CLS, SAVE_PERIOD, BASE_WEIGHTS, the fliplr=0.0
# fix) is read from train_yolo directly so the two runs differ only in
# jitter magnitude, not in any other hyperparameter.
base.RUN_NAME = "yolov8_seed0split_zerojitter"
base.TRANSLATE = 0.0
base.SCALE = 0.0
base.EVAL_RESULTS_DIR = Path(__file__).resolve().parent / "eval_results" / "zerojitter"


def train(data_yaml):
    model, resume = get_or_create_model()
    if resume:
        results = model.train(resume=True)
    else:
        results = model.train(
            data=str(data_yaml),
            epochs=base.EPOCHS,
            batch=base.BATCH,
            imgsz=base.IMGSZ,
            device=base.DEVICE,
            project=base.PROJECT_DIR,
            name=base.RUN_NAME,
            exist_ok=True,
            dropout=base.DROPOUT,
            close_mosaic=base.CLOSE_MOSAIC,
            cos_lr=base.COS_LR,
            warmup_epochs=base.WARMUP_EPOCHS,
            lrf=base.LRF,
            single_cls=base.SINGLE_CLS,
            translate=base.TRANSLATE,   # 0.0 - the only intended difference from train_yolo.py
            scale=base.SCALE,           # 0.0 - the only intended difference from train_yolo.py
            save_period=base.SAVE_PERIOD,
            val=True,
            fliplr=0.0,  # same fix as train_yolo.py - see that file's comment
        )
    return results


def main():
    data_yaml = prepare_yolo_dataset()
    train(data_yaml)
    best_weights = Path(base.PROJECT_DIR) / base.RUN_NAME / "weights" / "best.pt"
    base_evaluate(best_weights)


if __name__ == "__main__":
    main()
