"""DENTEX follow-up checks, reported only (DENTEX_EXTERNAL_RULES.md grades the frozen run).

    python dentex_sensitivity.py --res results/kaggle_dentex --ufba-boxes boxes.csv --ufba-folds <folds csv>

1. Thresholds: rescores the saved detections at confidence 0.25 and/or IoU 0.3
   (confidence-first, as scored), giving each detector's top-1, missed rate and
   margin over DENTEX position-only.
2. Framing: rescales each X-ray's box coordinates to the span its labeled teeth
   cover, then fits the UFBA fold position models on rescaled UFBA boxes and
   applies them to rescaled DENTEX boxes (and five-fold within DENTEX).
Writes dentex_thresholds.csv and dentex_framing.csv to --res.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_cv as sc  # noqa: E402
from closure_intervention import position_models  # noqa: E402

DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]
GRID = [(0.5, 0.5), (0.25, 0.5), (0.5, 0.3), (0.25, 0.3)]


def rescale(b):
    """Box coordinates relative to the span of that X-ray's labeled teeth."""
    b = b.copy()
    for c, s in (("x_center", "width"), ("y_center", "height")):
        lo = (b[c] - b[s] / 2).groupby(b["image_id"]).transform("min")
        hi = (b[c] + b[s] / 2).groupby(b["image_id"]).transform("max")
        b[c], b[s] = (b[c] - lo) / (hi - lo), b[s] / (hi - lo)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--ufba-boxes", type=Path, required=True)
    ap.add_argument("--ufba-folds", type=Path, required=True)
    a = ap.parse_args()
    sc.MATCH_ORDER = "conf"
    rng = np.random.default_rng(0)
    t = pd.read_csv(a.res / "dentex_teeth.csv")
    dets = pd.read_csv(a.res / "dentex_detections.csv")
    keep = ["image_id", "fdi", "class_id", "x_center", "y_center", "width", "height", "fold", "tid", "coord_correct"]
    base = t[keep].copy()
    base["_pos"] = base["coord_correct"].astype(int)

    rows = []
    for conf, iou in GRID:
        sc.CONF, sc.MATCH_IOU = conf, iou
        u = base.copy()
        for d in DETS:
            sc.add_detections(u, d, dets.loc[dets["detector"] == d].drop(columns="detector"))
            u[f"_{d}"] = u[f"{d}_correct"].astype(int)
            top1 = sc.boot(u, [f"_{d}"], lambda s: 100 * s[1] / s[0], rng)
            margin = sc.boot(u, [f"_{d}", "_pos"], lambda s: 100 * (s[1] - s[2]) / s[0], rng)
            rows.append(dict(conf=conf, iou=iou, detector=d, top1=top1[0], top1_lo=top1[1], top1_hi=top1[2],
                             missed_pct=100 * u[f"{d}_missed"].mean(), margin=margin[0], margin_lo=margin[1],
                             margin_hi=margin[2]))
        if (conf, iou) == (0.5, 0.5):  # must reproduce the frozen run
            for d in DETS:
                assert (u[f"{d}_correct"] == t[f"{d}_correct"]).all(), d
    T = pd.DataFrame(rows)
    T.to_csv(a.res / "dentex_thresholds.csv", index=False)

    ub = rescale(pd.read_csv(a.ufba_boxes))
    folds = pd.read_csv(a.ufba_folds)
    dx = rescale(base)
    feats = dx.assign(area=dx["width"] * dx["height"], aspect_ratio=dx["width"] / dx["height"])[sc.FEATURES]
    pred = np.full(len(dx), -1)
    for f, clf in position_models(ub, folds).items():
        m = (dx["fold"] == f).to_numpy()
        pred[m] = clf.predict(feats[m])
    dx["_ufba_rescaled"] = (pred == dx["class_id"]).astype(int)
    cv = sc.position_only(dx.drop(columns="fold"), dx[["image_id", "fold"]].drop_duplicates())
    dx = dx.merge(cv[["tid", "coord_pred"]], on="tid", validate="1:1")
    dx["_dentex_rescaled"] = (dx["coord_pred"] == dx["class_id"]).astype(int)
    dx["_ufba_raw"] = t.set_index("tid").loc[dx["tid"], "coord_ufba_correct"].astype(int).to_numpy()
    F = []
    for c in ["_ufba_raw", "_ufba_rescaled", "_pos", "_dentex_rescaled"]:
        p, lo, hi = sc.boot(dx, [c], lambda s: 100 * s[1] / s[0], rng)
        F.append(dict(model=c.strip("_").replace("pos", "dentex_raw"), top1=p, lo=lo, hi=hi))
    F = pd.DataFrame(F)
    F.to_csv(a.res / "dentex_framing.csv", index=False)
    pd.set_option("display.width", 200)
    print(T.round(2).to_string(index=False))
    print(F.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
