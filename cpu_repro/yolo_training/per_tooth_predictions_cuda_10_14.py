"""Per-tooth predictions for the Kaggle CUDA-trained seeds 10-14 (RT-DETR-l,
YOLOv8x, Faster R-CNN).

Counterpart of per_tooth_predictions_ultralytics_mps.py / per_tooth_predictions_mps.py
for the CUDA backend. Training ran on Kaggle and only saved aggregate metrics,
so per-tooth analyses (missed-tooth breakdown, cross-architecture error
agreement) need inference rerun on each downloaded best.pt. Protocol is
identical to the MPS scripts: same val split (regenerated here from
image_split_seed{N}.csv, since these seeds never trained locally), conf=EVAL_CONF,
NMS iou=EVAL_NMS_IOU, and greedy one-to-one IoU matching at
MATCH_IOU_THRESHOLD. Inference runs locally (Faster R-CNN on CPU, RT-DETR-l and
YOLOv8x on mps), not on the CUDA device the models were trained and evaluated
on. A different device can shift outputs numerically, so the summary.csv check
below is what establishes that the rerun matches the training-time evaluation.

Weights: runs/{rtdetr_cuda,yolov8_cuda,fasterrcnn_cuda}_seed{N}/weights/best.pt,
copied from the Kaggle kernels' Output downloads. Those downloads carry no seed
in their filenames; each was assigned to a seed only after its inference output
reproduced that seed's summary.csv (top1_acc, n_test, n_missed).
RT-DETR has seeds 10-14; YOLOv8x and Faster R-CNN have 11-14 only (seed 10
weights were never downloaded for those two).

Writes eval_results/rtdetr/seed{N}/per_tooth.csv, eval_results/seed{N}/per_tooth.csv
(YOLOv8x: bare "seed" dir is this repo's convention for CUDA YOLO, see
BACKEND_COMPARISON.md), eval_results/fasterrcnn_cuda/seed{N}/per_tooth.csv,
same columns as the MPS per_tooth.csv files.

Each seed is verified against that seed's existing summary.csv: number of
matched teeth, number of missed teeth, and number of correct top-1 matches
must all equal the training-time values, otherwise the seed is reported
MISMATCH, its per_tooth.csv is written to per_tooth.csv.mismatch instead, and
the exit code is 1.

Usage:
  python per_tooth_predictions_cuda_10_14.py --check                  # verify inputs, no inference
  python per_tooth_predictions_cuda_10_14.py [model ...] [--seeds 10 11 ...]
      model is yolo, rtdetr, and/or fasterrcnn (default all three)
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

MODEL_SEEDS = {
    "rtdetr": [10, 11, 12, 13, 14],
    "yolo": [11, 12, 13, 14],
    "fasterrcnn": [11, 12, 13, 14],
}
RUN_DIRS = {
    "rtdetr": "rtdetr_cuda_seed{s}",
    "yolo": "yolov8_cuda_seed{s}",
    "fasterrcnn": "fasterrcnn_cuda_seed{s}",
}


def eval_dir_for(model, seed):
    if model == "yolo":
        return HERE / "eval_results" / f"seed{seed}"
    if model == "rtdetr":
        return HERE / "eval_results" / "rtdetr" / f"seed{seed}"
    return HERE / "eval_results" / "fasterrcnn_cuda" / f"seed{seed}"


def weights_path(model, seed):
    return HERE / "runs" / RUN_DIRS[model].format(s=seed) / "weights" / "best.pt"


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


def prep_coco_split(seed):
    import build_coco_dataset as bcd
    out = HERE / "prepared_coco" / f"seed{seed}"
    if (out / "instances_val.json").exists():
        return out
    bcd.SEED = seed
    bcd.SPLIT_FILE = HERE.parents[0] / "coord_baseline" / f"image_split_seed{seed}.csv"
    bcd.OUT_DIR = out
    bcd.convert()
    return out


def check_inputs(model, seed):
    problems = []
    wpath = weights_path(model, seed)
    if not wpath.exists():
        problems.append(f"missing weights: {wpath}")
    summ = eval_dir_for(model, seed) / "summary.csv"
    if not summ.exists():
        problems.append(f"missing summary.csv: {summ}")
    return problems


def run_ultralytics(model, seed):
    from ultralytics import RTDETR, YOLO
    prep = prep_yolo_split(seed)
    val_txt = prep / "val.txt"
    net = {"yolo": YOLO, "rtdetr": RTDETR}[model](str(weights_path(model, seed)))

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
    return rows


def run_fasterrcnn(seed):
    import train_fasterrcnn as base
    prep = prep_coco_split(seed)
    val_json = prep / "instances_val.json"
    base.SEED = seed
    base.VAL_JSON = val_json

    model = base.get_model()
    model.load_state_dict(torch.load(weights_path("fasterrcnn", seed), map_location="cpu", weights_only=True))
    model.eval()
    val_ds = base.CocoTeethDataset(val_json, base.train_transforms())

    rows = []
    with torch.no_grad():
        for idx, iid in enumerate(val_ds.image_ids):
            info = val_ds.images[iid]
            w_img, h_img = info["width"], info["height"]
            anns = val_ds.anns_by_image[iid]
            gt_cls = [a["category_id"] - 1 for a in anns]
            gt_boxes = [[a["bbox"][0] / w_img, a["bbox"][1] / h_img,
                         (a["bbox"][0] + a["bbox"][2]) / w_img, (a["bbox"][1] + a["bbox"][3]) / h_img]
                        for a in anns]

            img, _ = val_ds[idx]
            out = model([img])[0]
            pred_cls = (out["labels"].numpy() - 1).astype(int)
            pred_boxes = out["boxes"].numpy() / np.array([w_img, h_img, w_img, h_img])

            match = {}
            if len(gt_boxes) and len(pred_boxes):
                iou = ty.iou_xyxy(np.asarray(gt_boxes, float), np.asarray(pred_boxes, float))
                pairs = sorted(((iou[i, j], i, j) for i in range(len(gt_boxes))
                                for j in range(len(pred_boxes)) if iou[i, j] >= ty.MATCH_IOU_THRESHOLD),
                               key=lambda t: t[0], reverse=True)
                used_p = set()
                for _, i, j in pairs:
                    if i in match or j in used_p:
                        continue
                    match[i] = int(pred_cls[j])
                    used_p.add(j)

            for i, (c, b) in enumerate(zip(gt_cls, gt_boxes)):
                rows.append({"image_id": iid, "gt_idx": i, "true_class": c,
                             "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                             "pred_class": match.get(i, "")})
    return rows


def run_seed(model, seed):
    rows = run_fasterrcnn(seed) if model == "fasterrcnn" else run_ultralytics(model, seed)

    eval_dir = eval_dir_for(model, seed)
    summ = next(csv.DictReader(open(eval_dir / "summary.csv")))
    n_missed = sum(r["pred_class"] == "" for r in rows)
    n_matched = len(rows) - n_missed
    n_correct = sum(r["pred_class"] != "" and int(r["pred_class"]) == r["true_class"] for r in rows)
    exp_matched, exp_missed = int(summ["n_test"]), int(summ["n_unmatched_gt_missed_detections"])
    exp_correct = round(float(summ["top1_acc"]) * exp_matched)
    ok = (n_matched, n_missed, n_correct) == (exp_matched, exp_missed, exp_correct)

    dest = eval_dir / ("per_tooth.csv" if ok else "per_tooth.csv.mismatch")
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
    ap.add_argument("models", nargs="*", help="yolo, rtdetr, and/or fasterrcnn (default all three)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--check", action="store_true", help="verify inputs exist; run no inference")
    args = ap.parse_args()
    models = args.models or list(MODEL_SEEDS)
    bad = [m for m in models if m not in MODEL_SEEDS]
    if bad:
        ap.error(f"unknown model(s) {bad}; choose from {list(MODEL_SEEDS)}")

    jobs = []
    for m in models:
        seeds = args.seeds if args.seeds is not None else MODEL_SEEDS[m]
        for s in seeds:
            if s not in MODEL_SEEDS[m]:
                print(f"skip {m} seed {s}: no downloaded weights for this seed/model", flush=True)
                continue
            jobs.append((m, s))

    if args.check:
        failed = False
        for m, s in jobs:
            problems = check_inputs(m, s)
            print(f"{m} seed {s}: {'OK' if not problems else 'PROBLEM'}")
            for pr in problems:
                print("   ", pr)
            failed |= bool(problems)
        sys.exit(1 if failed else 0)

    results = [run_seed(m, s) for m, s in jobs]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
