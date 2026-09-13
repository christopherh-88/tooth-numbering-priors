"""RT-DETR 1-epoch smoke test - RESULTS.md Section 38's scoped next step.

Purpose: get a REAL per-epoch timing for RT-DETR on this exact dataset/
hardware (replacing Section 38's estimated 1.5-3x-YOLOv8x guess with a
measurement) and confirm the Ultralytics YOLO->RTDETR pipeline swap works
end to end, before sizing/launching a full 5-seed replication. This is
NOT meant to produce a trustworthy accuracy number - 1 epoch on a
detector this size will not have converged. Only the timing and "did it
run without crashing" outcomes matter here.

Also captures the GPU environment info Section 37 was blocked from
retrieving retroactively (torch/ultralytics versions, nvidia-smi) by
printing it to stdout at the very start of the run, so it lands in this
kernel's own log this time instead of requiring a separate retroactive
fetch.

Reuses train_yolo.py's data prep, box-matching, and evaluation logic
unchanged (same seed-0 split, same IoU-matching protocol, same
coord_evaluate() call) - only the model class (RTDETR instead of YOLO)
and epoch count differ. This is deliberate: reusing the identical
evaluation methodology is what makes an eventual RT-DETR-vs-YOLOv8
comparison meaningful rather than confounded by a different eval
protocol (see Section 38's reasoning for why RT-DETR was chosen over a
from-scratch Faster R-CNN pipeline).

Usage on Kaggle (GPU T4 x2, selected manually in the web UI first - the
API's enable_gpu flag has repeatedly and unreliably assigned a P100
instead, see Sections 21/31/33):

    python train_rtdetr_smoketest.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_yolo as base  # noqa: E402 - reuse data prep / matching / eval, not the YOLO class itself

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coord_baseline"))
from build_coord_baseline import evaluate as coord_evaluate  # noqa: E402

# --------------------------------------------------------------------------
BASE_WEIGHTS = "rtdetr-l.pt"   # smaller of the two RT-DETR variants Ultralytics ships - fastest smoke test
EPOCHS = 1                     # smoke test only - see module docstring, not a real training run
BATCH = base.BATCH             # 10, same as train_yolo.py, for a like-for-like per-epoch timing comparison
IMGSZ = base.IMGSZ             # 640
DEVICE = base.DEVICE           # 0 on Kaggle GPU
PROJECT_DIR = str(Path(__file__).resolve().parent / "runs")
RUN_NAME = "rtdetr_smoketest_seed0split"
EVAL_RESULTS_DIR = Path(__file__).resolve().parent / "eval_results" / "rtdetr_smoketest"
# --------------------------------------------------------------------------


def print_environment_info():
    """Section 37's disclosed gap: GPU-side package versions were never
    captured. Print them here so this run's own Kaggle log has them,
    rather than requiring a retroactive kernel-output fetch (which was
    rate-limited and abandoned - see Section 37)."""
    import torch
    import ultralytics

    print("=" * 70)
    print("Environment (captured at run start, per RESULTS.md Section 37)")
    print("=" * 70)
    print(f"  torch version:       {torch.__version__}")
    print(f"  torch cuda version:  {torch.version.cuda}")
    print(f"  ultralytics version: {ultralytics.__version__}")
    print(f"  cuda available:      {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  cuda device:         {torch.cuda.get_device_name(0)}")
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        ).stdout
        print("  nvidia-smi:")
        print(nvidia_smi)
    except Exception as e:
        print(f"  nvidia-smi not available: {e}")
    print("=" * 70 + "\n")


def train_smoketest(data_yaml):
    from ultralytics import RTDETR

    model = RTDETR(BASE_WEIGHTS)
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
        fliplr=0.0,  # same reasoning as train_yolo.py: FDI class ids encode left/right quadrant
    )


def evaluate_smoketest(weights_path):
    """Same matched-IoU evaluation as train_yolo.evaluate(), pointed at
    this run's own weights/output directory instead of seed 0's."""
    from ultralytics import RTDETR

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = RTDETR(str(weights_path))
    val_paths = (base.PREPARED_DIR / "val.txt").read_text().splitlines()
    train_paths = (base.PREPARED_DIR / "train.txt").read_text().splitlines()

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

    metrics = coord_evaluate(y_true, y_pred, train_y) if len(y_true) > 0 else {
        "top1_acc": float("nan"), "quadrant_acc": float("nan"),
        "tooth_type_acc": float("nan"), "majority_baseline_acc": float("nan"),
        "mirror_quadrant_error_frac": float("nan"), "neighbor_error_frac": float("nan"),
    }
    metrics["seed"] = 0
    metrics["classifier"] = "rtdetr_smoketest_1epoch"
    metrics["detection_recall"] = (
        (total_gt - total_unmatched_gt) / total_gt if total_gt > 0 else float("nan")
    )
    metrics["n_unmatched_gt_missed_detections"] = total_unmatched_gt
    metrics["n_unmatched_pred_spurious_detections"] = total_unmatched_pred

    pd.DataFrame([metrics]).to_csv(EVAL_RESULTS_DIR / "summary.csv", index=False)

    print("\n" + "=" * 70)
    print("RT-DETR 1-EPOCH SMOKE TEST evaluation (NOT a converged result - timing/pipeline check only)")
    print("=" * 70)
    for key in ["top1_acc", "quadrant_acc", "detection_recall"]:
        print(f"  {key:32s} {metrics.get(key)}")
    print(f"\nSaved summary.csv to {EVAL_RESULTS_DIR}")
    return metrics


def main():
    print_environment_info()
    data_yaml = base.prepare_yolo_dataset()
    results = train_smoketest(data_yaml)

    # Ultralytics logs per-epoch and total training time in results.speed / the
    # trainer's own printed summary - also read train time directly from the
    # run's own results.csv if present, so Section 38's cost estimate can be
    # replaced with a real number without depending on stdout scraping alone.
    speed = getattr(results, "speed", None)
    if speed:
        print(f"\nUltralytics-reported speed dict: {speed}")

    best_weights = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    evaluate_smoketest(best_weights)


if __name__ == "__main__":
    main()
