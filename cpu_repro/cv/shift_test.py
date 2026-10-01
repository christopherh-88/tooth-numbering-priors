"""Shift test (cpu_repro/cv/PHASE3_RULES.md item 1, frozen 2026-10-01).

Moves each test X-ray's content by dx (fraction of width) or dy (fraction
of height), fills the uncovered strip with black, moves the labeled boxes
with it, and runs each detector's model from the fold that tests that
X-ray. Runs in a Kaggle GPU kernel (see kaggle/make_phase3_kernels.py):

    python shift_test.py --repo <repo checkout> --models <dir with *_cvseed0/> --out <dir>

Writes shift_per_tooth.csv (one row per tooth, detector and condition) and
shift_summary.csv (per detector and condition: accuracy change with a
paired X-ray bootstrap CI, follow rate with its permutation null).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CONDITIONS = [(0.0, 0.0), (-0.10, 0.0), (-0.05, 0.0), (0.05, 0.0), (0.10, 0.0), (0.0, -0.05), (0.0, 0.05)]
CONF, MATCH_IOU, NMS_IOU, IMGSZ = 0.5, 0.5, 0.7, 640
FEATURES = ["x_center", "y_center", "width", "height", "area", "aspect_ratio"]
N_BOOT = N_PERM = 10_000
SECTION60_TOP1 = {"yolov8x": 94.8974, "rtdetr_l": 93.8287, "fasterrcnn": 92.5961}


def add_shape(df):
    return df.assign(area=df["width"] * df["height"], aspect_ratio=df["width"] / df["height"])


def shift_image(img, dx_px, dy_px):
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    xs, xd = (slice(0, w - dx_px), slice(dx_px, w)) if dx_px >= 0 else (slice(-dx_px, w), slice(0, w + dx_px))
    ys, yd = (slice(0, h - dy_px), slice(dy_px, h)) if dy_px >= 0 else (slice(-dy_px, h), slice(0, h + dy_px))
    out[yd, xd] = img[ys, xs]
    return out


def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def match(gt_xyxy, det_cls, det_conf, det_xyxy):
    """Greedy one-to-one IoU matching, class-agnostic, as score_cv.match."""
    pred = np.full(len(gt_xyxy), np.nan)
    keep = det_conf >= CONF
    det_cls, det_xyxy = det_cls[keep], det_xyxy[keep]
    if len(det_cls) == 0 or len(gt_xyxy) == 0:
        return pred
    m = iou(gt_xyxy, det_xyxy)
    used_g, used_d = set(), set()
    for _, i, j in sorted(((m[i, j], i, j) for i, j in zip(*np.nonzero(m >= MATCH_IOU))), reverse=True):
        if i in used_g or j in used_d:
            continue
        used_g.add(i)
        used_d.add(j)
        pred[i] = det_cls[j]
    return pred


class Detector:
    def __init__(self, name, fold_dir):
        self.name = name
        if name in ("yolov8x", "rtdetr_l"):
            from ultralytics import RTDETR, YOLO
            w = fold_dir / "ultralytics" / "weights" / "best.pt"
            self.model = (YOLO if name == "yolov8x" else RTDETR)(str(w))
        else:
            import torch
            from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            m = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, box_nms_thresh=NMS_IOU)
            m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 33)
            m.load_state_dict(torch.load(fold_dir / "best.pt", map_location="cuda", weights_only=True))
            m.roi_heads.score_thresh = 0.05
            self.model = m.to("cuda").eval()

    def __call__(self, bgr):
        """Returns class ids, confidences and normalized xyxy boxes."""
        h, w = bgr.shape[:2]
        if self.name != "fasterrcnn":
            r = self.model.predict(source=bgr, conf=0.05, iou=NMS_IOU, imgsz=IMGSZ, device=0, verbose=False)[0]
            return (r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(),
                    r.boxes.xyxyn.cpu().numpy())
        import torch
        x = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).permute(2, 0, 1).float().div(255)
        with torch.no_grad():
            o = self.model([x.to("cuda")])[0]
        return ((o["labels"].cpu().numpy() - 1).astype(int), o["scores"].cpu().numpy(),
                o["boxes"].cpu().numpy() / np.array([w, h, w, h]))


def boot_ci(df, a, b, rng):
    """Paired change (b - a, in pp) and 95% CI, resampling X-rays."""
    s = df.assign(n=1).groupby("image_id")[["n", a, b]].sum().to_numpy(float)
    stat = lambda t: 100 * (t[2] - t[1]) / t[0]
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    return stat(s.sum(0)), *np.percentile(stat(s[idx].sum(1).T), [2.5, 97.5])


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    boxes = pd.read_csv(a.repo / "cpu_repro/cv/boxes.csv")
    folds = pd.read_csv(a.repo / "cpu_repro/cv/folds.csv")
    boxes = add_shape(boxes.merge(folds[["image_id", "fold"]], on="image_id"))
    coord = {f: HistGradientBoostingClassifier(random_state=0).fit(
        boxes.loc[boxes["fold"] != f, FEATURES], boxes.loc[boxes["fold"] != f, "class_id"])
        for f in range(5)}
    img_dir = a.repo / "Dataset/bb_u_net_dataset/panoramic_x_rays"

    rows = []
    for det in ("yolov8x", "rtdetr_l", "fasterrcnn"):
        run = next(a.models.rglob(f"{det}_cvseed0"))
        for f in range(5):
            model = Detector(det, run / f"fold{f}")
            for image_id, g in boxes[boxes["fold"] == f].groupby("image_id"):
                img = cv2.imread(str(img_dir / f"{image_id}.jpg"))
                h, w = img.shape[:2]
                for dx, dy in CONDITIONS:
                    dx_px, dy_px = int(round(dx * w)), int(round(dy * h))
                    s = g.assign(x_center=g["x_center"] + dx_px / w, y_center=g["y_center"] + dy_px / h)
                    xyxy = np.stack([s.x_center - s.width / 2, s.y_center - s.height / 2,
                                     s.x_center + s.width / 2, s.y_center + s.height / 2], 1)
                    inside = ((xyxy >= 0) & (xyxy <= 1)).all(1)
                    s, xyxy = s[inside], xyxy[inside]
                    pred = match(xyxy, *model(shift_image(img, dx_px, dy_px)))
                    coord_pred = coord[f].predict(add_shape(s)[FEATURES]) if len(s) else []
                    rows += [dict(detector=det, dx=dx, dy=dy, image_id=image_id, fold=f, fdi=r.fdi,
                                  class_id=r.class_id, det_pred=p, coord_pred=c)
                             for r, p, c in zip(s.itertuples(), pred, coord_pred)]
            print(det, "fold", f, "done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(a.out / "shift_per_tooth.csv", index=False)

    summary = []
    for det, d in t.groupby("detector"):
        base = d[(d.dx == 0) & (d.dy == 0)].set_index(["image_id", "fdi"])
        top1_0 = 100 * (base["det_pred"] == base["class_id"]).mean()
        print(f"{det}: unshifted top-1 {top1_0:.4f} (Section 60: {SECTION60_TOP1[det]:.4f})", flush=True)
        for (dx, dy), c in d.groupby(["dx", "dy"]):
            if dx == 0 and dy == 0:
                continue
            c = c.set_index(["image_id", "fdi"])
            j = c.join(base[["det_pred"]], rsuffix="_0", how="inner").reset_index()
            j["ok0"] = (j["det_pred_0"] == j["class_id"]).astype(int)
            j["ok1"] = (j["det_pred"] == j["class_id"]).astype(int)
            change, lo, hi = boot_ci(j, "ok0", "ok1", rng)
            ch = j[j["det_pred"].notna() & j["det_pred_0"].notna() & (j["det_pred"] != j["det_pred_0"])]
            follow = float((ch["det_pred"] == ch["coord_pred"]).mean()) if len(ch) else np.nan
            null = np.array([np.mean(ch["det_pred"].to_numpy() == rng.permutation(ch["coord_pred"].to_numpy()))
                             for _ in range(N_PERM)]) if len(ch) else np.array([np.nan])
            summary.append(dict(
                detector=det, dx=dx, dy=dy, n_teeth=len(j), n_xrays=j["image_id"].nunique(),
                top1_unshifted=100 * j["ok0"].mean(), top1_shifted=100 * j["ok1"].mean(),
                change_pp=change, change_lo=lo, change_hi=hi,
                coord_top1_shifted=100 * (j["coord_pred"] == j["class_id"]).mean(),
                n_changed=len(ch), follow_pct=100 * follow, follow_null_mean_pct=100 * np.nanmean(null),
                follow_null_p95_pct=100 * np.nanpercentile(null, 95),
                follow_p=float(np.mean(null >= follow)) if len(ch) else np.nan,
                unshifted_full_top1=top1_0, section60_top1=SECTION60_TOP1[det]))
    pd.DataFrame(summary).to_csv(a.out / "shift_summary.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(summary).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
