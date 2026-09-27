"""Per-tooth predictions for YOLOv8x CUDA seeds 7 and 9 - the two seeds
flagged in BACKEND_COMPARISON.md and RESULTS.md Section 49 with a
missed-tooth-rate spike (~4%). Breakdown in RESULTS.md Section 53.

The checkpoints come from the Kaggle kernels' Output downloads, which carry
no seed in their filenames. Each was assigned to a seed only after its
inference output reproduced that seed's summary.csv (top1_acc, n_test,
n_missed); the matched/missed/correct check below repeats that verification.

Same protocol as per_tooth_predictions_cuda_10_14.py: regenerate the val
split from image_split_seed{N}.csv, conf=EVAL_CONF, NMS iou=EVAL_NMS_IOU,
greedy one-to-one IoU matching, and verify matched/missed/correct counts
against the existing summary.csv before accepting the per-tooth output.

Weights: runs/yolov8_cuda_seed{7,9}/weights/best.pt
Writes: eval_results/seed{7,9}/per_tooth.csv

Usage: python per_tooth_predictions_yolo_cuda_7_9.py [--check]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_yolo as ty  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEEDS = [7, 9]


def weights_path(seed):
    return HERE / "runs" / f"yolov8_cuda_seed{seed}" / "weights" / "best.pt"


def eval_dir(seed):
    return HERE / "eval_results" / f"seed{seed}"


def label_path(image_path):
    p = Path(image_path)
    return p.parents[1] / "labels" / f"{p.stem}.txt"


def prep_yolo_split(seed):
    out = HERE / "prepared" / f"seed{seed}"
    if (out / "val.txt").exists():
        return out
    ty.SEED = seed
    ty.SPLIT_FILE = HERE.parents[0] / "coord_baseline" / f"image_split_seed{seed}.csv"
    ty.PREPARED_DIR = out
    ty.prepare_yolo_dataset()
    return out


def check_inputs(seed):
    problems = []
    if not weights_path(seed).exists():
        problems.append(f"missing weights: {weights_path(seed)}")
    if not (eval_dir(seed) / "summary.csv").exists():
        problems.append(f"missing summary.csv: {eval_dir(seed) / 'summary.csv'}")
    return problems


def run_seed(seed):
    from ultralytics import YOLO
    prep = prep_yolo_split(seed)
    val_txt = prep / "val.txt"
    net = YOLO(str(weights_path(seed)))

    rows = []
    for image_path in val_txt.read_text().splitlines():
        gt_cls, gt_boxes = ty.load_gt_boxes(label_path(image_path))
        res = net.predict(source=image_path, conf=ty.EVAL_CONF, iou=ty.EVAL_NMS_IOU,
                           device=DEVICE, verbose=False)[0]
        pred_cls = res.boxes.cls.cpu().numpy().astype(int).tolist()
        pred_boxes = res.boxes.xyxyn.cpu().numpy().tolist()

        match = {}
        if gt_boxes and pred_boxes:
            iou = ty.iou_xyxy(np.asarray(gt_boxes, float), np.asarray(pred_boxes, float))
            pairs = sorted(((iou[i, j], i, j) for i in range(len(gt_boxes)) for j in range(len(pred_boxes))
                            if iou[i, j] >= ty.MATCH_IOU_THRESHOLD), key=lambda t: t[0], reverse=True)
            used_p = set()
            for _, i, j in pairs:
                if i in match or j in used_p:
                    continue
                match[i] = pred_cls[j]
                used_p.add(j)

        for i, (c, b) in enumerate(zip(gt_cls, gt_boxes)):
            rows.append({"image_id": Path(image_path).stem, "gt_idx": i, "true_class": c,
                         "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                         "pred_class": match.get(i, "")})

    d = eval_dir(seed)
    summ = next(csv.DictReader(open(d / "summary.csv")))
    n_missed = sum(r["pred_class"] == "" for r in rows)
    n_matched = len(rows) - n_missed
    n_correct = sum(r["pred_class"] != "" and int(r["pred_class"]) == r["true_class"] for r in rows)
    exp_matched, exp_missed = int(summ["n_test"]), int(summ["n_unmatched_gt_missed_detections"])
    exp_correct = round(float(summ["top1_acc"]) * exp_matched)
    ok = (n_matched, n_missed, n_correct) == (exp_matched, exp_missed, exp_correct)

    dest = d / ("per_tooth.csv" if ok else "per_tooth.csv.mismatch")
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"yolo seed {seed}: {len(rows)} teeth, matched {n_matched}, missed {n_missed}, correct {n_correct}; "
          f"summary.csv says matched {exp_matched}, missed {exp_missed}, correct {exp_correct} "
          f"-> {'OK' if ok else 'MISMATCH (wrote ' + dest.name + ')'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        failed = False
        for s in SEEDS:
            problems = check_inputs(s)
            print(f"seed {s}: {'OK' if not problems else 'PROBLEM'}")
            for pr in problems:
                print("   ", pr)
            failed |= bool(problems)
        sys.exit(1 if failed else 0)
    results = [run_seed(s) for s in SEEDS]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
