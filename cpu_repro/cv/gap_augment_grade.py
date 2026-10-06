"""Grade gap augmentation training (cpu_repro/cv/GAP_AUGMENT_RULES.md, frozen 2026-10-05).

    python gap_augment_grade.py --original <yolov8x_cvseed0 dir> --control <yolov8x_cvseed0_tseed1 dir> \
        --augmented <yolov8x_cvseed0_aug dir> --per-tooth <confidence-first per_tooth_predictions.csv> --out <dir>

Scores the three YOLOv8x runs on the seed 0 split with confidence-first
matching (score_cv.add_detector), then compares them tooth by tooth:
top-1 overall, top-1 on teeth next to a gap, and the share of the 161
joint failures (Section 65) numbered right. 95% CIs from 10,000 bootstrap
draws of X-rays. Writes augment_per_tooth.csv and augment_summary.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import score_cv
from gap_check import NEIGHBORS

DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]
N_BOOT = 10_000
RUNS = ["original", "control", "augmented"]


def boot_diff(df, a, b, mask, rng):
    """Top-1 of a, of b, and a minus b (points) on rows in mask; CIs by X-ray."""
    x = df[mask].assign(n=1)
    s = x.groupby("image_id")[["n", a, b]].sum().to_numpy(float)
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    bt = s[idx].sum(1)
    tot = s.sum(0)
    out = {}
    for name, num, k in ((a, bt[:, 1], 1), (b, bt[:, 2], 2)):
        out[name] = 100 * tot[k] / tot[0]
        out[f"{name}_lo"], out[f"{name}_hi"] = np.percentile(100 * num / bt[:, 0], [2.5, 97.5])
    d = 100 * (bt[:, 1] - bt[:, 2]) / bt[:, 0]
    out["diff"] = 100 * (tot[1] - tot[2]) / tot[0]
    out["diff_lo"], out["diff_hi"] = np.percentile(d, [2.5, 97.5])
    return out


def rule(gap, overall):
    if gap["diff_lo"] <= 0 <= gap["diff_hi"]:
        return 2
    if gap["diff"] >= 2 and gap["diff_lo"] > 0 and overall["diff"] > -0.5:
        return 1
    return 3


def main():
    ap = argparse.ArgumentParser()
    for r in RUNS:
        ap.add_argument(f"--{r}", type=Path, required=True)
    ap.add_argument("--per-tooth", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    score_cv.MATCH_ORDER = "conf"

    ref = pd.read_csv(a.per_tooth, dtype={"fdi": str})
    t = ref[["image_id", "fold", "fdi", "class_id", "x_center", "y_center", "width", "height"]].copy()
    for r in RUNS:
        score_cv.add_detector(t, r, getattr(a, r))
    present = t.groupby("image_id")["fdi"].agg(set).to_dict()
    t["next_to_gap"] = [any(n not in present[i] for n in NEIGHBORS[f]) for i, f in zip(t["image_id"], t["fdi"])]
    joint = ref["coord_pred"] != ref["class_id"]
    for d in DETS:
        joint &= (ref[f"{d}_pred"] != ref["class_id"]) & ref[f"{d}_pred"].notna()
    joint &= (ref["yolov8x_pred"] == ref["rtdetr_l_pred"]) & (ref["yolov8x_pred"] == ref["fasterrcnn_pred"])
    t["joint_failure"] = joint.to_numpy()
    for r in RUNS:
        t[f"ok_{r}"] = (t[f"{r}_pred"] == t["class_id"]).astype(int)
    t.to_csv(a.out / "augment_per_tooth.csv", index=False)

    # Check: the original run must reproduce Section 65's YOLOv8x top-1 (94.98%).
    print("original top-1:", round(100 * t["ok_original"].mean(), 4), "(Section 65: 94.9836)", flush=True)

    rng = np.random.default_rng(0)
    masks = {"overall": np.ones(len(t), bool), "next_to_gap": t["next_to_gap"].to_numpy(),
             "joint_failures": t["joint_failure"].to_numpy()}
    rows = []
    for a_run, b_run in (("augmented", "control"), ("augmented", "original"), ("control", "original")):
        res = {}
        for m, mask in masks.items():
            r = boot_diff(t, f"ok_{a_run}", f"ok_{b_run}", mask, rng)
            res[m] = r
            rows.append(dict(comparison=f"{a_run} minus {b_run}", subset=m, n=int(mask.sum()),
                             top1_a=r[f"ok_{a_run}"], top1_b=r[f"ok_{b_run}"],
                             diff=r["diff"], diff_lo=r["diff_lo"], diff_hi=r["diff_hi"]))
        if (a_run, b_run) == ("augmented", "control"):
            graded = rule(res["next_to_gap"], res["overall"])
    out = pd.DataFrame(rows)
    out["rule"] = np.where((out["comparison"] == "augmented minus control") & (out["subset"] == "next_to_gap"),
                           graded, np.nan)
    out.to_csv(a.out / "augment_summary.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(out.round(2).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
