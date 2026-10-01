"""Calibration on joint failures and cutoff sensitivity
(cpu_repro/cv/PHASE3_RULES.md items 2 and 3, frozen 2026-10-01).

    python calib_sweep.py --per-tooth <per_tooth_predictions.csv> --repo <repo checkout> \
        --det yolov8x=<dir> --det rtdetr_l=<dir> --det fasterrcnn=<dir> --out <dir>

Item 2 uses the per-tooth file from score_cv.py. Item 3 rescores the raw
detections (confidence >= 0.05, from train_cv.py) at each cutoff pair with
score_cv.py's own position-only model and matcher, imported from the same
directory.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_cv as sc  # noqa: E402

DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]
CONFS = [0.25, 0.5, 0.75]
IOUS = [0.3, 0.5, 0.7]
N_BOOT = 10_000
SECTION60_GAP = {"yolov8x": 16.4023, "rtdetr_l": 15.3336, "fasterrcnn": 14.1010}


def calibration(t, rng):
    joint = t["coord_pred"] != t["class_id"]
    for d in DETS:
        joint &= (t[f"{d}_pred"] != t["class_id"]) & t[f"{d}_pred"].notna()
    preds = np.stack([t[f"{d}_pred"].to_numpy() for d in DETS])
    joint &= (preds == preds[0]).all(0)
    rows = []
    for d in DETS:
        ok = t[f"{d}_pred"] == t["class_id"]
        sub = t[ok | joint].assign(y=ok[ok | joint].astype(int))  # 1 = correct, 0 = joint failure
        groups = {k: v.index.to_numpy() for k, v in sub.groupby("image_id")}
        keys = list(groups)
        auc = roc_auc_score(sub["y"], sub[f"{d}_conf"])
        boots = []
        for _ in range(N_BOOT):
            idx = np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
            s = sub.loc[idx]
            if s["y"].nunique() == 2:
                boots.append(roc_auc_score(s["y"], s[f"{d}_conf"]))
        rows.append(dict(detector=d, n_joint=int(joint.sum()), n_correct=int(ok.sum()),
                         conf_mean_joint=sub.loc[sub.y == 0, f"{d}_conf"].mean(),
                         conf_mean_correct=sub.loc[sub.y == 1, f"{d}_conf"].mean(),
                         conf_median_joint=sub.loc[sub.y == 0, f"{d}_conf"].median(),
                         auroc=auc, auroc_lo=np.percentile(boots, 2.5), auroc_hi=np.percentile(boots, 97.5),
                         rule=1 if auc >= 0.8 else (2 if auc < 0.7 else 3)))
    return pd.DataFrame(rows)


def sweep(repo, det_dirs, order="iou"):
    teeth = sc.position_only(pd.read_csv(repo / "cpu_repro/cv/boxes.csv"),
                             pd.read_csv(repo / "cpu_repro/cv/folds.csv"))
    coord_top1 = 100 * (teeth["coord_pred"] == teeth["class_id"]).mean()
    rows = []
    sc.MATCH_ORDER = order
    for conf in CONFS:
        for iou_cut in IOUS:
            sc.CONF, sc.MATCH_IOU = conf, iou_cut
            for d, run_dir in det_dirs.items():
                t = teeth.copy()
                sc.add_detector(t, d, run_dir)
                top1 = 100 * t[f"{d}_correct"].mean()
                rows.append(dict(detector=d, match_order=order, conf=conf, iou=iou_cut, det_top1=top1, coord_top1=coord_top1,
                                 gap_pp=top1 - coord_top1, missed_pct=100 * t[f"{d}_missed"].mean(),
                                 section60_gap=SECTION60_GAP[d],
                                 diff_from_section60=top1 - coord_top1 - SECTION60_GAP[d]))
                print(rows[-1], flush=True)
    sc.CONF, sc.MATCH_IOU, sc.MATCH_ORDER = 0.5, 0.5, "iou"
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tooth", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--det", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--match-order-only", action="store_true",
                    help="skip item 2; run the sweep under both matcher orders (PHASE3_RULES.md note)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.match_order_only:
        return match_order(a)
    rng = np.random.default_rng(0)
    cal = calibration(pd.read_csv(a.per_tooth), rng)
    cal.to_csv(a.out / "calibration_joint_failures.csv", index=False, float_format="%.4f")
    print(cal.round(3).to_string(index=False), flush=True)
    sw = sweep(a.repo, dict(s.split("=", 1) for s in a.det))
    sw.to_csv(a.out / "cutoff_sweep.csv", index=False, float_format="%.4f")
    worst = sw.loc[sw["diff_from_section60"].abs().idxmax()]
    print(sw.round(2).to_string(index=False))
    print("largest |diff| from Section 60:", round(float(worst["diff_from_section60"]), 2),
          "at", dict(worst[["detector", "conf", "iou"]]))
    print("rule:", "within 2 pp everywhere" if sw["diff_from_section60"].abs().max() <= 2 else "range reported")


def match_order(a):
    dets = dict(s.split("=", 1) for s in a.det)
    sw = pd.concat([sweep(a.repo, dets, o) for o in ("iou", "conf")])
    sw.to_csv(a.out / "cutoff_sweep_match_order.csv", index=False, float_format="%.4f")
    w = sw.pivot_table(index=["detector", "conf", "iou"], columns="match_order", values="gap_pp")
    w["conf_minus_iou"] = w["conf"] - w["iou"]
    print(w.round(2).to_string())
    r = w.loc["rtdetr_l"]["conf"]
    print("RT-DETR-l conf-first, conf 0.25 minus conf 0.5 (per IoU):",
          {i: round(r[(0.25, i)] - r[(0.5, i)], 2) for i in IOUS})
    print("change at 0.5/0.5 per detector:",
          {d: round(w.loc[(d, 0.5, 0.5), "conf_minus_iou"], 3) for d in DETS})


if __name__ == "__main__":
    main()
