"""COCO-style AP of the tooth-numbering detectors (cpu_repro/cv/AP_RULES.md). Reported only.

    python ap_eval.py --boxes <boxes.csv> --run name=<detections.csv> [--run ...] --out <dir>

--run files hold one row per raw detection (image_id, class_id, conf, x1, y1, x2, y2; boxes
normalized to the image, confidence at least 0.05), either one file per detector or a
dentex_detections.csv with a `detector` column (then name=<file>#<detector>).
AP is class-aware: a detection counts for its own number, so a misnumbered box is a false
positive of the number it gives and leaves the labeled tooth unmatched. Matching is COCO's
(per X-ray and class, detections by confidence, each takes the unmatched labeled tooth of
highest IoU at or above the threshold), AP is the 101-point interpolated area, averaged over
the numbers present, and AP50:95 over IoU 0.50, 0.55, ..., 0.95. 95% CIs resample X-rays.
Detections below the 0.05 confidence the files were saved at are missing, so the tail of the
precision-recall curve is cut there.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

IOU_THRS = np.round(np.arange(0.5, 0.951, 0.05), 2)
REC_THRS = np.linspace(0, 1, 101)
N_BOOT = 1_000


def box_iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def match_class(gt, det_xyxy, det_conf):
    """COCO matching of one class in one X-ray: tp flags [n_det, n_thr], detections in confidence order."""
    order = np.argsort(-det_conf, kind="stable")
    tp = np.zeros((len(order), len(IOU_THRS)), bool)
    if len(gt) == 0 or len(order) == 0:
        return order, tp
    m = box_iou(det_xyxy[order], gt)
    for k, thr in enumerate(IOU_THRS):
        taken = np.zeros(len(gt), bool)
        for i in range(len(order)):
            cand = np.where(taken, -1.0, m[i])
            j = int(cand.argmax())
            if cand[j] >= thr:
                taken[j] = True
                tp[i, k] = True
    return order, tp


def collect(dets, boxes):
    """Per class: detection confidences, X-ray index, tp flags; per class and X-ray: labeled teeth."""
    ids = sorted(boxes["image_id"].unique())
    ix = {i: k for k, i in enumerate(ids)}
    gt_xyxy = np.stack([boxes.x_center - boxes.width / 2, boxes.y_center - boxes.height / 2,
                        boxes.x_center + boxes.width / 2, boxes.y_center + boxes.height / 2], 1)
    per = {c: dict(conf=[], img=[], tp=[], n_gt=np.zeros(len(ids))) for c in range(32)}
    gb = {(i, c): g for (i, c), g in boxes.assign(_k=np.arange(len(boxes))).groupby(["image_id", "class_id"])}
    db = {(i, c): g for (i, c), g in dets.groupby(["image_id", "class_id"])}
    for key in set(gb) | set(db):
        image_id, c = key
        if image_id not in ix:
            continue
        gt = gt_xyxy[gb[key]["_k"].to_numpy()] if key in gb else np.zeros((0, 4))
        per[c]["n_gt"][ix[image_id]] += len(gt)
        if key in db:
            d = db[key]
            order, tp = match_class(gt, d[["x1", "y1", "x2", "y2"]].to_numpy(float), d["conf"].to_numpy(float))
            per[c]["conf"].append(d["conf"].to_numpy(float)[order])
            per[c]["img"].append(np.full(len(order), ix[image_id]))
            per[c]["tp"].append(tp)
    out = {}
    for c, v in per.items():
        if not v["conf"]:
            out[c] = dict(conf=np.zeros(0), img=np.zeros(0, int), tp=np.zeros((0, len(IOU_THRS)), bool), n_gt=v["n_gt"])
            continue
        conf, img, tp = np.concatenate(v["conf"]), np.concatenate(v["img"]), np.concatenate(v["tp"])
        o = np.argsort(-conf, kind="stable")
        out[c] = dict(conf=conf[o], img=img[o], tp=tp[o], n_gt=v["n_gt"])
    return out, len(ids)


def ap_curve(w_det, tp, n_gt):
    """101-point AP for each IoU threshold; w_det are per-detection weights (bootstrap counts)."""
    if n_gt <= 0:
        return np.full(len(IOU_THRS), np.nan)
    if len(tp) == 0:
        return np.zeros(len(IOU_THRS))
    ctp = np.cumsum(w_det[:, None] * tp, 0)
    cfp = np.cumsum(w_det[:, None] * ~tp, 0)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    ap = np.zeros(len(IOU_THRS))
    for k in range(len(IOU_THRS)):
        p = np.maximum.accumulate(prec[::-1, k])[::-1]
        idx = np.searchsorted(rec[:, k], REC_THRS, side="left")
        ok = idx < len(p)
        ap[k] = np.where(ok, p[np.minimum(idx, len(p) - 1)], 0).sum() / len(REC_THRS)
    return ap


def mean_ap(per, w_img):
    aps = []
    for c, v in per.items():
        n = float((w_img * v["n_gt"]).sum())
        if n > 0:
            aps.append(ap_curve(w_img[v["img"]] if len(v["img"]) else np.zeros(0), v["tp"], n))
    aps = np.array(aps)
    return float(100 * aps[:, 0].mean()), float(100 * aps.mean())


def evaluate(dets, boxes, rng):
    per, n_img = collect(dets, boxes)
    p50, p5095 = mean_ap(per, np.ones(n_img))
    draws = np.array([mean_ap(per, np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(float))
                      for _ in range(N_BOOT)])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return dict(ap50=p50, ap50_lo=lo[0], ap50_hi=hi[0], ap5095=p5095, ap5095_lo=lo[1], ap5095_hi=hi[1],
                n_xrays=n_img, n_teeth=len(boxes), n_dets=len(dets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", type=Path, required=True)
    ap.add_argument("--run", action="append", required=True, help="name=detections.csv[#detector]")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    boxes = pd.read_csv(a.boxes, dtype={"fdi": str})
    rng = np.random.default_rng(0)
    rows = []
    for spec in a.run:
        name, path = spec.split("=", 1)
        det = None
        if "#" in path:
            path, det = path.split("#")
        d = pd.read_csv(path)
        if det is not None:
            d = d[d["detector"] == det]
        d = d[d["image_id"].isin(set(boxes["image_id"]))]
        r = dict(run=name, **evaluate(d, boxes, rng))
        rows.append(r)
        print(f"{name}: AP50 {r['ap50']:.1f} ({r['ap50_lo']:.1f} to {r['ap50_hi']:.1f}), "
              f"AP50:95 {r['ap5095']:.1f} ({r['ap5095_lo']:.1f} to {r['ap5095_hi']:.1f})", flush=True)
    pd.DataFrame(rows).to_csv(a.out / "ap_summary.csv", index=False, float_format="%.3f")


if __name__ == "__main__":
    main()
