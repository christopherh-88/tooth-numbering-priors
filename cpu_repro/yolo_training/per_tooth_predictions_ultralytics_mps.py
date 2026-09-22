"""Per-tooth predictions for the MPS-trained YOLOv8x and RT-DETR-l seeds (5-9).

Counterpart of per_tooth_predictions_mps.py (Faster R-CNN). The training runs
only saved aggregate metrics, so per-tooth analyses (missed-tooth breakdown,
cross-architecture error agreement) need inference rerun on each best.pt.
Protocol is identical to train_yolo.evaluate() / train_rtdetr.evaluate():
same val split (prepared/seed{N}/val.txt), conf=EVAL_CONF, NMS iou=EVAL_NMS_IOU,
the same device the training run used (mps), and greedy one-to-one IoU
matching at MATCH_IOU_THRESHOLD (same rule as train_yolo.match_boxes, but
keeping the per-tooth assignment that function does not return).

Writes eval_results/{yolo_mps,rtdetr_mps}/seed{N}/per_tooth.csv, same columns
as eval_results/fasterrcnn/seed{N}/per_tooth.csv: one row per labeled tooth
with its normalized box, class, and matched prediction (empty if missed).

Each seed is verified against that seed's summary.csv: number of matched
teeth, number of missed teeth, and number of correct top-1 matches must all
equal the training-time values, otherwise the seed is reported MISMATCH, its
per_tooth.csv is written to per_tooth.csv.mismatch instead, and the exit code
is 1. So a per_tooth.csv only exists if it reproduces the training numbers.

Usage:
  python per_tooth_predictions_ultralytics_mps.py --check          # no inference: verify inputs exist
  python per_tooth_predictions_ultralytics_mps.py [model ...] [--seeds 5 6 ...]
      model is yolo and/or rtdetr (default both); seeds default to 5-9
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_yolo as ty  # noqa: E402  (label loading, iou_xyxy, eval constants)

MODELS = {
    "yolo": {"run": "yolov8_mps_seed{s}split", "eval": "yolo_mps", "cls": "YOLO"},
    "rtdetr": {"run": "rtdetr_mps_seed{s}split", "eval": "rtdetr_mps", "cls": "RTDETR"},
}
DEVICE = "mps"


def paths(model, seed):
    m = MODELS[model]
    eval_dir = HERE / "eval_results" / m["eval"] / f"seed{seed}"
    return {"weights": HERE / "runs" / m["run"].format(s=seed) / "weights" / "best.pt",
            "val": HERE / "prepared" / f"seed{seed}" / "val.txt",
            "eval_dir": eval_dir, "summary": eval_dir / "summary.csv"}


def label_path(image_path):
    p = Path(image_path)
    return p.parents[1] / "labels" / f"{p.stem}.txt"


def check_inputs(model, seed):
    p = paths(model, seed)
    problems = [f"missing {k}: {v}" for k, v in p.items() if k != "eval_dir" and not v.exists()]
    if not problems:
        val = p["val"].read_text().splitlines()
        missing = [x for x in val if not Path(x).exists() or not label_path(x).exists()]
        if not val:
            problems.append(f"empty {p['val']}")
        if missing:
            problems.append(f"{len(missing)}/{len(val)} val images or labels not found, e.g. {missing[0]}")
    return problems


def run_seed(model, seed):
    from ultralytics import RTDETR, YOLO
    p = paths(model, seed)
    net = {"yolo": YOLO, "rtdetr": RTDETR}[model](str(p["weights"]))

    rows = []
    for image_path in p["val"].read_text().splitlines():
        gt_cls, gt_boxes = ty.load_gt_boxes(label_path(image_path))
        res = net.predict(source=image_path, conf=ty.EVAL_CONF, iou=ty.EVAL_NMS_IOU,
                          device=DEVICE, verbose=False)[0]
        pred_cls = res.boxes.cls.cpu().numpy().astype(int).tolist()
        pred_boxes = res.boxes.xyxyn.cpu().numpy().tolist()

        match = {}  # gt index -> predicted class
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

    summ = next(csv.DictReader(open(p["summary"])))
    n_missed = sum(r["pred_class"] == "" for r in rows)
    n_matched = len(rows) - n_missed
    n_correct = sum(r["pred_class"] != "" and int(r["pred_class"]) == r["true_class"] for r in rows)
    exp_matched, exp_missed = int(summ["n_test"]), int(summ["n_unmatched_gt_missed_detections"])
    exp_correct = round(float(summ["top1_acc"]) * exp_matched)
    ok = (n_matched, n_missed, n_correct) == (exp_matched, exp_missed, exp_correct)

    dest = p["eval_dir"] / ("per_tooth.csv" if ok else "per_tooth.csv.mismatch")
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{model} seed {seed}: {len(rows)} teeth, matched {n_matched}, missed {n_missed}, correct {n_correct}; "
          f"summary.csv says matched {exp_matched}, missed {exp_missed}, correct {exp_correct} "
          f"-> {'OK' if ok else 'MISMATCH (wrote ' + dest.name + ')'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="yolo and/or rtdetr (default both)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    ap.add_argument("--check", action="store_true", help="verify inputs exist; run no inference")
    args = ap.parse_args()
    models = args.models or list(MODELS)
    bad = [m for m in models if m not in MODELS]
    if bad:
        ap.error(f"unknown model(s) {bad}; choose from {list(MODELS)}")

    if args.check:
        failed = False
        for m in models:
            for s in args.seeds:
                problems = check_inputs(m, s)
                print(f"{m} seed {s}: {'OK' if not problems else 'PROBLEM'}")
                for pr in problems:
                    print("   ", pr)
                failed |= bool(problems)
        sys.exit(1 if failed else 0)

    results = [run_seed(m, s) for m in models for s in args.seeds]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
