"""YOLOv8 training run, prepared to start the moment GPU quota is back.

Do NOT run this on CPU as a real training run - it's written for a Kaggle
GPU session. Everything needed to start immediately is here: it reads the
coordinate baseline's persisted image-level split (does not recompute it),
resumes cleanly across Kaggle's 12-hour session limit, and evaluates with
the exact same metrics/format as cpu_repro/coord_baseline so the two are
directly comparable.

Usage on Kaggle (same command whether this is the first run or a resume
after a session got killed - see get_or_create_model() below):

    python train_yolo.py

Pipeline, each stage idempotent / safe to rerun:
  1. prepare_yolo_dataset() - build train.txt/val.txt + data.yaml from the
     persisted split file (../coord_baseline/image_split_seed0.csv).
  2. get_or_create_model() + train() - resume from the last checkpoint if
     one exists for RUN_NAME, else start fresh from BASE_WEIGHTS.
  3. evaluate() - run the trained model on the held-out (val) images,
     IoU-match predictions to ground truth, and report per-tooth FDI
     numbering accuracy using build_coord_baseline.evaluate() (the exact
     same function the coordinate baseline uses), so results land in the
     same columns as cpu_repro/coord_baseline/per_seed_results.csv.

Checkpoint/resume: ultralytics writes weights/last.pt after every epoch by
default. If a Kaggle session is killed (hard VM stop - there's no reliable
way to intercept that in a notebook), the fix is simply: rerun this script.
get_or_create_model() checks for an existing weights/last.pt under
PROJECT_DIR/RUN_NAME and, if found, resumes via `model.train(resume=True)`
(ultralytics reloads the original training args from that run's args.yaml
automatically - don't re-pass epochs/batch/etc. when resuming, see below).
SAVE_PERIOD additionally keeps periodic numbered snapshots as a safety net
in case last.pt itself is ever corrupted by a mid-write kill.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coord_baseline"))
from build_coord_baseline import FDI_CODES, SEEDS  # noqa: E402

# --------------------------------------------------------------------------
# CONFIG - edit these, nothing else, to change the run.
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
YOLO_SOURCE_ROOT = REPO_ROOT / "Dataset" / "yolo_train_dataset"
SPLIT_FILE = REPO_ROOT / "cpu_repro" / "coord_baseline" / "image_split_seed0.csv"

PREPARED_DIR = Path(__file__).resolve().parent / "prepared"
EVAL_RESULTS_DIR = Path(__file__).resolve().parent / "eval_results"

# Training hyperparameters - mirrors notebooks/yolov8/yolov8_train.ipynb's
# CLI call as closely as possible, so this run is comparable to the
# repo's original YOLOv8 training, just on our fixed comparable split.
BASE_WEIGHTS = "yolov8x.pt"   # matches the original notebook; use yolov8n.pt for a fast local smoke test
PROJECT_DIR = str(Path(__file__).resolve().parent / "runs")
RUN_NAME = "yolov8_seed0split"
EPOCHS = 30
BATCH = 10
IMGSZ = 640
DEVICE = 0            # GPU device index for Kaggle; set to "cpu" only for a local smoke test
DROPOUT = 0.6
CLOSE_MOSAIC = 0
COS_LR = True
WARMUP_EPOCHS = 10
LRF = 0.005
SINGLE_CLS = False
SAVE_PERIOD = 5        # extra numbered checkpoint every N epochs, on top of the always-on last.pt/best.pt

# Evaluation
EVAL_CONF = 0.5         # matches conf=0.5 used in notebooks/yolov8/yolo_test.ipynb and yolov8+unet predict calls
EVAL_NMS_IOU = 0.7      # matches iou=0.7 used in the same notebooks (NMS threshold, not the matching threshold below)
MATCH_IOU_THRESHOLD = 0.5  # IoU required to count a prediction as matching a ground-truth box
# --------------------------------------------------------------------------

AUG_SUFFIX_RE = re.compile(r"_jpg\.rf\.[0-9a-f]+$")


def base_image_id(label_stem: str) -> str:
    """Identical logic to build_coord_baseline.base_image_id - duplicated
    (not imported) only because it's a one-line regex and this module
    should be runnable standalone by copy-pasting to a Kaggle notebook
    without needing coord_baseline importable at the same relative path."""
    return AUG_SUFFIX_RE.sub("", label_stem)


def load_split() -> dict:
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"{SPLIT_FILE} not found. This script reads the coordinate baseline's "
            f"persisted split - it does not regenerate it. Run "
            f"cpu_repro/coord_baseline/export_split.py once first."
        )
    split_df = pd.read_csv(SPLIT_FILE)
    return dict(zip(split_df["image_id"], split_df["split"]))


def prepare_yolo_dataset():
    """Build train.txt / val.txt (absolute image paths) + data.yaml from the
    persisted split, pointing at the existing Dataset/yolo_train_dataset
    images/labels in place (no copying)."""
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    split_map = load_split()

    label_files = sorted(YOLO_SOURCE_ROOT.glob("*/labels/*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files found under {YOLO_SOURCE_ROOT}")

    train_paths, val_paths = [], []
    unmapped = 0
    for lf in label_files:
        image_id = base_image_id(lf.stem)
        split = split_map.get(image_id)
        if split is None:
            unmapped += 1
            continue
        image_path = lf.parents[1] / "images" / f"{lf.stem}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Expected image not found: {image_path}")
        (train_paths if split == "train" else val_paths).append(str(image_path))

    if unmapped:
        print(f"WARNING: {unmapped} label files had an image_id not present in the split file "
              f"and were skipped.")

    train_txt = PREPARED_DIR / "train.txt"
    val_txt = PREPARED_DIR / "val.txt"
    train_txt.write_text("\n".join(train_paths) + "\n")
    val_txt.write_text("\n".join(val_paths) + "\n")

    data_yaml = PREPARED_DIR / "data.yaml"
    names_block = "\n".join(f"  {i}: {code}" for i, code in enumerate(FDI_CODES))
    data_yaml.write_text(
        f"train: {train_txt}\n"
        f"val: {val_txt}\n"
        f"nc: {len(FDI_CODES)}\n"
        f"names:\n{names_block}\n"
    )

    print(f"Prepared {len(train_paths)} train / {len(val_paths)} val images -> {data_yaml}")
    return data_yaml


def get_or_create_model():
    """Return (model, resume) - resume=True if a checkpoint for RUN_NAME
    already exists, so training continues rather than restarts."""
    from ultralytics import YOLO

    last_ckpt = Path(PROJECT_DIR) / RUN_NAME / "weights" / "last.pt"
    if last_ckpt.exists():
        print(f"Found existing checkpoint at {last_ckpt} - resuming training.")
        return YOLO(str(last_ckpt)), True
    print(f"No existing checkpoint for run '{RUN_NAME}' - starting fresh from {BASE_WEIGHTS}.")
    return YOLO(BASE_WEIGHTS), False


def train(data_yaml):
    model, resume = get_or_create_model()

    if resume:
        # Per ultralytics convention: when resuming, pass resume=True only.
        # The trainer reloads the original epochs/batch/imgsz/etc. from
        # that run's runs/.../args.yaml automatically; re-passing them here
        # would be redundant and, for some args, is rejected by ultralytics
        # when resume=True.
        results = model.train(resume=True)
    else:
        results = model.train(
            data=str(data_yaml),
            epochs=EPOCHS,
            batch=BATCH,
            imgsz=IMGSZ,
            device=DEVICE,
            project=PROJECT_DIR,
            name=RUN_NAME,
            exist_ok=True,
            dropout=DROPOUT,
            close_mosaic=CLOSE_MOSAIC,
            cos_lr=COS_LR,
            warmup_epochs=WARMUP_EPOCHS,
            lrf=LRF,
            single_cls=SINGLE_CLS,
            save_period=SAVE_PERIOD,
            val=True,
        )
    return results


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two arrays of boxes in normalized xyxy format,
    shapes (N,4) and (M,4). Returns an (N,M) IoU matrix. Pure numpy, no
    ultralytics dependency, so it's unit-testable without a GPU or a
    trained model."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.clip(x2 - x1, 0, None)
    inter_h = np.clip(y2 - y1, 0, None)
    inter = inter_w * inter_h

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)


def match_boxes(gt_boxes, gt_classes, pred_boxes, pred_classes, iou_threshold=MATCH_IOU_THRESHOLD):
    """Greedy one-to-one IoU matching between ground-truth and predicted
    boxes for a single image. Returns (matched_true, matched_pred,
    n_unmatched_gt, n_unmatched_pred).

    This is the core "given detection, is the class right" comparison that
    makes the model comparable to the coordinate baseline, which assumes
    box positions are already given. n_unmatched_gt / n_unmatched_pred
    (missed detections / spurious detections) are NOT part of the FDI
    accuracy numbers below - they're reported separately as detection
    recall/precision context, since the coordinate baseline has no
    equivalent (it's never given a "wrong" box, only a ground-truth one)."""
    gt_boxes = np.asarray(gt_boxes, dtype=float)
    pred_boxes = np.asarray(pred_boxes, dtype=float)
    gt_classes = np.asarray(gt_classes, dtype=int)
    pred_classes = np.asarray(pred_classes, dtype=int)

    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    if n_gt == 0 or n_pred == 0:
        return [], [], n_gt, n_pred

    iou = iou_xyxy(gt_boxes, pred_boxes)
    matched_gt, matched_pred = set(), set()
    pairs = [(iou[i, j], i, j) for i in range(n_gt) for j in range(n_pred) if iou[i, j] >= iou_threshold]
    pairs.sort(key=lambda t: t[0], reverse=True)

    matched_true, matched_pred_cls = [], []
    for _, i, j in pairs:
        if i in matched_gt or j in matched_pred:
            continue
        matched_gt.add(i)
        matched_pred.add(j)
        matched_true.append(int(gt_classes[i]))
        matched_pred_cls.append(int(pred_classes[j]))

    n_unmatched_gt = n_gt - len(matched_gt)
    n_unmatched_pred = n_pred - len(matched_pred)
    return matched_true, matched_pred_cls, n_unmatched_gt, n_unmatched_pred


def xywhn_to_xyxyn(x, y, w, h):
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def load_gt_boxes(label_path: Path):
    classes, boxes = [], []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        x, y, w, h = (float(v) for v in parts[1:5])
        classes.append(cls)
        boxes.append(xywhn_to_xyxyn(x, y, w, h))
    return classes, boxes


def evaluate(weights_path):
    """Run the trained model on the held-out val split, IoU-match against
    ground truth, and report the same metrics/columns as
    cpu_repro/coord_baseline/per_seed_results.csv, plus detection
    recall/precision as separate context."""
    from ultralytics import YOLO
    from build_coord_baseline import evaluate as coord_evaluate
    from sklearn.metrics import confusion_matrix

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    val_paths = (PREPARED_DIR / "val.txt").read_text().splitlines()
    train_paths = (PREPARED_DIR / "train.txt").read_text().splitlines()

    def label_path_for(image_path_str):
        p = Path(image_path_str)
        return p.parents[1] / "labels" / f"{p.stem}.txt"

    train_y = []
    for image_path in train_paths:
        classes, _ = load_gt_boxes(label_path_for(image_path))
        train_y.extend(classes)
    train_y = np.array(train_y)

    all_true, all_pred = [], []
    total_unmatched_gt, total_unmatched_pred, total_gt = 0, 0, 0

    for image_path in val_paths:
        gt_classes, gt_boxes = load_gt_boxes(label_path_for(image_path))
        total_gt += len(gt_classes)

        result = model.predict(source=image_path, conf=EVAL_CONF, iou=EVAL_NMS_IOU,
                                device=DEVICE, verbose=False)[0]
        pred_classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
        pred_boxes = result.boxes.xyxyn.cpu().numpy().tolist()

        matched_true, matched_pred, n_unmatched_gt, n_unmatched_pred = match_boxes(
            gt_boxes, gt_classes, pred_boxes, pred_classes
        )
        all_true.extend(matched_true)
        all_pred.extend(matched_pred)
        total_unmatched_gt += n_unmatched_gt
        total_unmatched_pred += n_unmatched_pred

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    metrics = coord_evaluate(y_true, y_pred, train_y)
    metrics["seed"] = SEEDS[0]  # the split file is generated from this same seed - see export_split.py
    metrics["classifier"] = "yolov8"
    metrics["detection_recall"] = (
        (total_gt - total_unmatched_gt) / total_gt if total_gt > 0 else float("nan")
    )
    metrics["n_unmatched_gt_missed_detections"] = total_unmatched_gt
    metrics["n_unmatched_pred_spurious_detections"] = total_unmatched_pred

    pd.DataFrame([metrics]).to_csv(EVAL_RESULTS_DIR / "summary.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(32)))
    pd.DataFrame(cm, index=FDI_CODES, columns=FDI_CODES).to_csv(
        EVAL_RESULTS_DIR / "confusion_matrix_yolov8_seed0.csv"
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 10))
        im = ax.imshow(cm, cmap="viridis")
        ax.set_xticks(range(32))
        ax.set_yticks(range(32))
        ax.set_xticklabels(FDI_CODES, rotation=90, fontsize=7)
        ax.set_yticklabels(FDI_CODES, fontsize=7)
        ax.set_xlabel("Predicted FDI code")
        ax.set_ylabel("True FDI code")
        ax.set_title("yolov8 - confusion matrix (matched detections, seed 0 split)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(EVAL_RESULTS_DIR / "confusion_matrix_yolov8_seed0.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    print("\n" + "=" * 70)
    print("YOLOv8 evaluation (matched detections vs. ground truth, seed-0 split)")
    print("=" * 70)
    for key in ["top1_acc", "quadrant_acc", "tooth_type_acc", "majority_baseline_acc",
                "mirror_quadrant_error_frac", "neighbor_error_frac", "detection_recall"]:
        print(f"  {key:32s} {metrics[key]}")
    print(f"  missed detections (unmatched GT)  {total_unmatched_gt} / {total_gt}")
    print(f"  spurious detections (unmatched pred) {total_unmatched_pred}")
    print(f"\nSaved summary.csv and confusion_matrix_yolov8_seed0.csv/.png to {EVAL_RESULTS_DIR}")

    return metrics


def main():
    data_yaml = prepare_yolo_dataset()
    train(data_yaml)

    best_weights = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    evaluate(best_weights)


if __name__ == "__main__":
    main()
