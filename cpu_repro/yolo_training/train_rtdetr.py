"""RT-DETR full training run (RESULTS.md Section 38/39's scoped second-
detector-architecture replication). Seed-parameterized like train_yolo.py -
SEED=0 here, train_rtdetr_seed{1,2,3,4}.py override it.

Purpose: replicate Claim B's non-reliance finding (Sections 21/33 for
YOLOv8) with an architecturally distinct detector (transformer encoder-
decoder, anchor-free, NMS-free - see Section 38's reasoning), reusing the
identical data pipeline, split, and evaluation methodology
(build_coord_baseline.evaluate() on IoU-matched detections) so any
difference in the resulting gap is attributable to architecture, not to
a different eval protocol.

Epoch count/hyperparameters otherwise mirror train_yolo.py's defaults
(30 epochs, batch 10, imgsz 640) for a like-for-like per-seed comparison,
using Ultralytics' own default RT-DETR recipe rather than a hand-tuned
one (Section 38's hyperparameter-disclosure note, carried over from
Section 28's YOLO precedent). Base weights are rtdetr-l.pt, matching the
smoke test (Section 39) that measured this variant's real per-epoch cost
(~86.4s/epoch on a T4, ~1.21x YOLOv8x's 71.3s/epoch) - not rtdetr-x,
which was not measured.

Usage on Kaggle (GPU T4 x2 - the machine_shape kernel-metadata.json
field / --accelerator NvidiaTeslaT4 push flag reliably attached this in
Section 39's 4 consecutive attempts, no manual web UI step needed, but
that fix is flagged "very likely, not yet guaranteed" - confirm it holds
before assuming a manual step is unnecessary):

    python train_rtdetr.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_yolo as base  # noqa: E402 - reuse data prep / matching / eval, not the YOLO class itself

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coord_baseline"))
from build_coord_baseline import evaluate as coord_evaluate, SEEDS  # noqa: E402

# --------------------------------------------------------------------------
SEED = 0
assert SEED in SEEDS, f"SEED={SEED} not in this project's SEEDS={SEEDS}"
SPLIT_FILE = base.REPO_ROOT / "cpu_repro" / "coord_baseline" / f"image_split_seed{SEED}.csv"
PREPARED_DIR = Path(__file__).resolve().parent / "prepared" / (f"seed{SEED}" if SEED != 0 else "")

BASE_WEIGHTS = "rtdetr-l.pt"   # measured variant (Section 39) - not rtdetr-x, which is unmeasured
EPOCHS = 30                    # matches train_yolo.py's default for a like-for-like comparison
BATCH = base.BATCH             # 10
IMGSZ = base.IMGSZ             # 640
DEVICE = base.DEVICE           # 0 on Kaggle GPU
PROJECT_DIR = str(Path(__file__).resolve().parent / "runs")
RUN_NAME = f"rtdetr_seed{SEED}split"
EVAL_RESULTS_DIR = Path(__file__).resolve().parent / "eval_results" / "rtdetr" / (f"seed{SEED}" if SEED != 0 else "seed0")
# --------------------------------------------------------------------------


def prepare_dataset():
    """Point train_yolo's SPLIT_FILE/PREPARED_DIR at this seed before
    calling its (unmodified) prepare_yolo_dataset(), same override
    pattern as train_yolo_seed1.py etc."""
    base.SEED = SEED
    base.SPLIT_FILE = SPLIT_FILE
    base.PREPARED_DIR = PREPARED_DIR
    return base.prepare_yolo_dataset()


def get_or_create_model():
    from ultralytics import RTDETR

    last_ckpt = Path(PROJECT_DIR) / RUN_NAME / "weights" / "last.pt"
    if last_ckpt.exists():
        print(f"Found existing checkpoint at {last_ckpt} - resuming training.")
        return RTDETR(str(last_ckpt)), True
    print(f"No existing checkpoint for run '{RUN_NAME}' - starting fresh from {BASE_WEIGHTS}.")
    return RTDETR(BASE_WEIGHTS), False


def train(data_yaml):
    model, resume = get_or_create_model()
    if resume:
        return model.train(resume=True)
    return model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        device=DEVICE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        val=True,
        fliplr=0.0,  # FDI class ids encode left/right quadrant - see train_yolo.py's train() for why
    )


def evaluate(weights_path):
    """Identical protocol to train_yolo.evaluate() - same
    coord_evaluate() call, same matched-IoU convention - only the model
    class differs, so results are directly comparable to Sections 21/33."""
    from ultralytics import RTDETR
    from sklearn.metrics import confusion_matrix

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = RTDETR(str(weights_path))
    val_paths = (PREPARED_DIR / "val.txt").read_text().splitlines()
    train_paths = (PREPARED_DIR / "train.txt").read_text().splitlines()

    def label_path_for(image_path_str):
        p = Path(image_path_str)
        return p.parents[1] / "labels" / f"{p.stem}.txt"

    train_y = []
    for image_path in train_paths:
        classes, _ = base.load_gt_boxes(label_path_for(image_path))
        train_y.extend(classes)
    train_y = np.array(train_y)

    all_true, all_pred = [], []
    total_unmatched_gt, total_unmatched_pred, total_gt = 0, 0, 0

    for image_path in val_paths:
        gt_classes, gt_boxes = base.load_gt_boxes(label_path_for(image_path))
        total_gt += len(gt_classes)

        result = model.predict(source=image_path, conf=base.EVAL_CONF, iou=base.EVAL_NMS_IOU,
                                device=DEVICE, verbose=False)[0]
        pred_classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
        pred_boxes = result.boxes.xyxyn.cpu().numpy().tolist()

        matched_true, matched_pred, n_unmatched_gt, n_unmatched_pred = base.match_boxes(
            gt_boxes, gt_classes, pred_boxes, pred_classes
        )
        all_true.extend(matched_true)
        all_pred.extend(matched_pred)
        total_unmatched_gt += n_unmatched_gt
        total_unmatched_pred += n_unmatched_pred

    y_true = np.array(all_true, dtype=int)
    y_pred = np.array(all_pred, dtype=int)

    metrics = coord_evaluate(y_true, y_pred, train_y)
    metrics["seed"] = SEED
    metrics["classifier"] = "rtdetr"
    metrics["detection_recall"] = (
        (total_gt - total_unmatched_gt) / total_gt if total_gt > 0 else float("nan")
    )
    metrics["n_unmatched_gt_missed_detections"] = total_unmatched_gt
    metrics["n_unmatched_pred_spurious_detections"] = total_unmatched_pred

    pd.DataFrame([metrics]).to_csv(EVAL_RESULTS_DIR / "summary.csv", index=False)

    if len(y_true) == 0:
        print("WARNING: zero matched detections across the whole val split - skipping confusion matrix.")
    else:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(32)))
        pd.DataFrame(cm, index=base.FDI_CODES, columns=base.FDI_CODES).to_csv(
            EVAL_RESULTS_DIR / f"confusion_matrix_rtdetr_seed{SEED}.csv"
        )

    print("\n" + "=" * 70)
    print(f"RT-DETR evaluation (matched detections vs. ground truth, seed {SEED} split)")
    print("=" * 70)
    for key in ["top1_acc", "quadrant_acc", "tooth_type_acc", "majority_baseline_acc",
                "mirror_quadrant_error_frac", "neighbor_error_frac", "detection_recall"]:
        print(f"  {key:32s} {metrics[key]}")
    print(f"  missed detections (unmatched GT)  {total_unmatched_gt} / {total_gt}")
    print(f"  spurious detections (unmatched pred) {total_unmatched_pred}")
    print(f"\nSaved summary.csv to {EVAL_RESULTS_DIR}")
    return metrics


def main():
    data_yaml = prepare_dataset()
    train(data_yaml)
    best_weights = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    evaluate(best_weights)


if __name__ == "__main__":
    main()
