"""Score the Phase 2 CV runs (cpu_repro/cv/PHASE2_CV_RULES.md, frozen 2026-10-01).

    python cpu_repro/cv/score_cv.py --det yolov8x=<dir> [--det rtdetr_l=<dir> --det fasterrcnn=<dir>] \
        --out <dir> [--boxes cpu_repro/cv/boxes.csv --folds cpu_repro/cv/folds.csv] [--match-order conf]

Each <dir> holds fold{0..4}/test_detections.csv from train_cv.py. Uses only
numpy, pandas and scikit-learn, so it runs unchanged in a Kaggle CPU kernel.

1. Position-only model per fold: HistGradientBoostingClassifier(random_state=0)
   on the same 6 features as cpu_repro/coord_baseline/build_coord_baseline.py
   (raw, unscaled), trained on the fold's training and validation X-rays,
   tested on its test X-rays.
2. Detections with confidence >= 0.5 are matched to labeled teeth by greedy
   one-to-one IoU >= 0.5, class-agnostic, as in yolo_training/train_yolo.py's
   match_boxes. An unmatched tooth is missed and counts as wrong.
3. Every statistic in the rules: pooled gap with a 10,000-draw X-ray
   bootstrap CI, per-fold guard, missed vs. misnumbered, kappa and phi, and
   (with all three detectors) joint-failure agreement with a 10,000-shuffle
   permutation null, defined as in yolo_training/cross_architecture_agreement.py.

Outputs in --out: per_tooth_predictions.csv, cv_fold_summary.csv,
cv_pooled_summary.csv, cv_joint_failures.csv (when two or more detectors are given; all of them
must be wrong, unmissed and agree).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES = ["x_center", "y_center", "width", "height", "area", "aspect_ratio"]
CONF = 0.5
MATCH_IOU = 0.5
MATCH_ORDER = "iou"  # "iou": highest IoU first; "conf": highest confidence first (Section 62 check)
N_BOOT = 10_000
N_PERM = 10_000
GUARD_MATCHED_TOP1 = 85.0
GUARD_MISSED = 10.0
FDI = [f"{q}{t}" for q in "1234" for t in "12345678"]


def position_only(boxes, folds):
    b = boxes.merge(folds[["image_id", "fold"]], on="image_id", validate="m:1")
    b["area"] = b["width"] * b["height"]
    b["aspect_ratio"] = b["width"] / b["height"]
    out = []
    for f in sorted(b["fold"].unique()):
        tr, te = b[b["fold"] != f], b[b["fold"] == f]
        clf = HistGradientBoostingClassifier(random_state=0).fit(tr[FEATURES], tr["class_id"])
        out.append(te.assign(coord_pred=clf.predict(te[FEATURES])))
    return pd.concat(out).sort_values(["image_id", "class_id"]).reset_index(drop=True)


def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def match(teeth, dets):
    """Per labeled tooth: matched predicted class (NaN if missed) and its confidence."""
    pred = np.full(len(teeth), np.nan)
    conf = np.full(len(teeth), np.nan)
    dets = dets[dets["conf"] >= CONF]
    if len(dets) == 0:
        return pred, conf
    g = teeth[["x_center", "y_center", "width", "height"]].to_numpy()
    g = np.stack([g[:, 0] - g[:, 2] / 2, g[:, 1] - g[:, 3] / 2,
                  g[:, 0] + g[:, 2] / 2, g[:, 1] + g[:, 3] / 2], 1)
    m = iou(g, dets[["x1", "y1", "x2", "y2"]].to_numpy())
    c = dets["conf"].to_numpy()
    key = (lambda i, j: (m[i, j],)) if MATCH_ORDER == "iou" else (lambda i, j: (c[j], m[i, j]))
    pairs = sorted((key(i, j) + (i, j) for i, j in zip(*np.nonzero(m >= MATCH_IOU))), reverse=True)
    used_g, used_d = set(), set()
    for *_, i, j in pairs:
        if i in used_g or j in used_d:
            continue
        used_g.add(i)
        used_d.add(j)
        pred[i] = dets["class_id"].iloc[j]
        conf[i] = dets["conf"].iloc[j]
    return pred, conf


def add_detector(teeth, name, run_dir):
    dets = pd.concat([pd.read_csv(p) for p in sorted(Path(run_dir).glob("fold*/test_detections.csv"))])
    n_folds = len(list(Path(run_dir).glob("fold*/test_detections.csv")))
    assert n_folds == 5, (name, n_folds)
    unknown = set(dets["image_id"]) - set(teeth["image_id"])
    assert not unknown, (name, sorted(unknown)[:5])
    by_img = dict(tuple(dets.groupby("image_id")))
    preds, confs = [], []
    for image_id, grp in teeth.groupby("image_id", sort=False):
        p, c = match(grp, by_img.get(image_id, dets.iloc[:0]))
        preds.append(pd.Series(p, index=grp.index))
        confs.append(pd.Series(c, index=grp.index))
    teeth[f"{name}_pred"] = pd.concat(preds)
    teeth[f"{name}_conf"] = pd.concat(confs)
    teeth[f"{name}_correct"] = teeth[f"{name}_pred"] == teeth["class_id"]
    teeth[f"{name}_missed"] = teeth[f"{name}_pred"].isna()


def boot(df, cols, stat, rng):
    """Point and 95% CI, resampling X-rays. cols are summed per X-ray."""
    s = df.assign(n=1).groupby("image_id")[["n"] + cols].sum().to_numpy(float)
    point = stat(s.sum(0))
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    vals = stat(s[idx].sum(1).T)
    return point, *np.percentile(vals, [2.5, 97.5])


def kappa_phi(t):
    n, a, b, bc, bw = t
    p1, p2 = a / n, b / n
    c_exp = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = ((bc + bw) / n - c_exp) / (1 - c_exp)
    kappa_max = ((1 - np.abs(p1 - p2)) - c_exp) / (1 - c_exp)
    phi = (bc * bw - (a - bc) * (b - bc)) / np.sqrt(a * (n - a) * b * (n - b))
    return kappa, kappa_max, phi


def summarize(teeth, dets, rng):
    t = teeth.assign(_c=teeth["coord_correct"].astype(int))
    fold_rows, pooled = [], []
    for d in dets:
        t[f"_{d}"] = t[f"{d}_correct"].astype(int)
        t[f"_{d}_m"] = (~t[f"{d}_missed"]).astype(int)
        t[f"_{d}_cm"] = (t["coord_correct"] & ~t[f"{d}_missed"]).astype(int)
        for f, g in t.groupby("fold"):
            matched = ~g[f"{d}_missed"]
            row = dict(detector=d, fold=f, n_xrays=g["image_id"].nunique(), n_teeth=len(g),
                       det_top1=100 * g[f"{d}_correct"].mean(),
                       coord_top1=100 * g["coord_correct"].mean(),
                       missed_pct=100 * g[f"{d}_missed"].mean(),
                       matched_only_top1=100 * g.loc[matched, f"{d}_correct"].mean())
            row["gap_pp"] = row["det_top1"] - row["coord_top1"]
            row["guard_failed"] = (row["matched_only_top1"] < GUARD_MATCHED_TOP1
                                   or row["missed_pct"] > GUARD_MISSED)
            fold_rows.append(row)

        cols = [f"_{d}", "_c", f"_{d}_m", f"_{d}_cm"]
        stats = {
            "det_top1": lambda s: 100 * s[1] / s[0],
            "coord_top1": lambda s: 100 * s[2] / s[0],
            "gap_pp": lambda s: 100 * (s[1] - s[2]) / s[0],
            "missed_pct": lambda s: 100 * (s[0] - s[3]) / s[0],
            "misnumbered_pct": lambda s: 100 * (s[3] - s[1]) / s[0],
            "matched_only_top1": lambda s: 100 * s[1] / s[3],
            "gap_matched_only_pp": lambda s: 100 * (s[1] - s[4]) / s[3],
        }
        for name, fn in stats.items():
            p, lo, hi = boot(t, cols, fn, rng)
            pooled.append(dict(detector=d, comparison="position_only", metric=name,
                               point=p, ci_lo=lo, ci_hi=hi))
    pairs = [(d, "coord") for d in dets] + [(a, b) for i, a in enumerate(dets) for b in dets[i + 1:]]
    for a, b in pairs:
        ca = t[f"_{a}"]
        cb = t["_c"] if b == "coord" else t[f"_{b}"]
        x = t.assign(_a=ca, _b=cb, _bc=ca & cb, _bw=(1 - ca) & (1 - cb))
        for k, name in enumerate(["kappa", "kappa_max", "phi"]):
            p, lo, hi = boot(x, ["_a", "_b", "_bc", "_bw"], lambda s, k=k: kappa_phi(s)[k], rng)
            pooled.append(dict(detector=a, comparison="position_only" if b == "coord" else b,
                               metric=name, point=p, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(fold_rows), pd.DataFrame(pooled)


def joint_failures(teeth, dets, rng):
    """Same definitions as yolo_training/cross_architecture_agreement.py's
    triple_agreement: every given detector and the position-only model wrong,
    no detector missed the tooth."""
    jf = teeth[~teeth["coord_correct"]]
    for d in dets:
        jf = jf[~jf[f"{d}_correct"] & jf[f"{d}_pred"].notna()]
    preds = np.stack([jf[f"{d}_pred"].astype(int).to_numpy() for d in dets])
    same = (preds == preds[0]).all(0)
    shared = preds[0][same]
    coord = jf.loc[same, "coord_pred"].astype(int).to_numpy()
    true = jf.loc[same, "class_id"].astype(int).to_numpy()
    n_match = int((shared == coord).sum())
    null = np.array([np.mean(shared == rng.permutation(coord)) for _ in range(N_PERM)]) if same.any() \
        else np.array([np.nan])
    neighbor = [FDI[a][0] == FDI[b][0] and abs(int(FDI[a][1]) - int(FDI[b][1])) == 1
                for a, b in zip(true, shared)]
    return pd.DataFrame([dict(
        n_teeth=len(teeth), n_joint=len(jf), joint_pct_of_teeth=100 * len(jf) / len(teeth),
        n_same_wrong=int(same.sum()), same_wrong_pct=100 * same.mean() if len(jf) else np.nan,
        n_position_match=n_match,
        position_match_pct=100 * n_match / same.sum() if same.any() else np.nan,
        null_mean_pct=100 * np.nanmean(null), null_p95_pct=100 * np.nanpercentile(null, 95),
        perm_p=float(np.mean(null >= n_match / max(same.sum(), 1))),
        neighbor_pct=100 * np.mean(neighbor) if neighbor else np.nan)])


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--det", action="append", required=True, help="name=run_dir")
    ap.add_argument("--boxes", type=Path, default=here / "boxes.csv")
    ap.add_argument("--folds", type=Path, default=here / "folds.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--match-order", choices=["iou", "conf"], default="iou")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    global MATCH_ORDER
    MATCH_ORDER = a.match_order
    rng = np.random.default_rng(0)

    teeth = position_only(pd.read_csv(a.boxes), pd.read_csv(a.folds))
    teeth["coord_correct"] = teeth["coord_pred"] == teeth["class_id"]
    dets = []
    for spec in a.det:
        name, run_dir = spec.split("=", 1)
        add_detector(teeth, name, run_dir)
        dets.append(name)

    keep = ["image_id", "category", "fold", "fdi", "class_id", "x_center", "y_center", "width",
            "height", "coord_pred"] + [f"{d}_{k}" for d in dets for k in ("pred", "conf")]
    teeth[keep].to_csv(a.out / "per_tooth_predictions.csv", index=False, float_format="%.6f")

    folds, pooled = summarize(teeth, dets, rng)
    folds.to_csv(a.out / "cv_fold_summary.csv", index=False, float_format="%.4f")
    pooled.to_csv(a.out / "cv_pooled_summary.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(folds.round(2).to_string(index=False))
    print(pooled[pooled["comparison"] == "position_only"].round(2).to_string(index=False))
    if len(dets) >= 2:
        jf = joint_failures(teeth, dets, rng).assign(detectors=" ".join(dets))
        jf.to_csv(a.out / "cv_joint_failures.csv", index=False, float_format="%.4f")
        print(jf.round(2).T.to_string())


if __name__ == "__main__":
    main()
