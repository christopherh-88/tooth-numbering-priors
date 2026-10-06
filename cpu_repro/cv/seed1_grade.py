"""Grade CV seed 1 (cpu_repro/cv/PHASE3_BATCH2_RULES.md item 3).

    python seed1_grade.py --dir results/kaggle_seed1
    python seed1_grade.py --dir results/kaggle_seed1_3det --dets yolov8x rtdetr_l fasterrcnn
    python seed1_grade.py --dir results/kaggle_seed2_3det --dets yolov8x rtdetr_l fasterrcnn --other-seed 2

Reads seed{0,1}_{N}det/ from the tooth-numbering-cv-seed1-score(3) kernel
(--dets, default YOLOv8x and RT-DETR-l; confidence-first). Gaps and CIs come from
cv_pooled_summary.csv. The joint same-wrong position-match rate (both
detectors and the position-only model wrong, same answer, none missed;
share where the position-only model gives that answer) gets an X-ray
bootstrap CI here, since score_cv prints only the point. Writes
seed1_grade.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 10_000


def position_match(t, rng, DETS):
    p = [t[f"{d}_pred"] for d in DETS]
    sw = t["coord_pred"] != t["class_id"]
    for x in p:
        sw &= (x != t["class_id"]) & x.notna()
    for x in p[1:]:
        sw &= p[0] == x
    pm = sw & (t["coord_pred"] == p[0])
    s = pd.DataFrame(dict(image_id=t["image_id"], n=sw.astype(int), m=pm.astype(int)))
    v = s.groupby("image_id").sum().to_numpy(float)
    bt = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].sum(1)
    lo, hi = np.percentile(100 * bt[:, 1] / bt[:, 0], [2.5, 97.5])
    return int(sw.sum()), 100 * pm.sum() / sw.sum(), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--dets", nargs="+", default=["yolov8x", "rtdetr_l"])
    ap.add_argument("--other-seed", type=int, default=1, help="the split compared with seed 0 (1 or 2)")
    a = ap.parse_args()
    DETS = a.dets
    tag = f"{len(DETS)}det"
    rows = []
    other = a.other_seed
    for s in (0, other):
        d = a.dir / f"seed{s}_{tag}"
        pooled = pd.read_csv(d / "cv_pooled_summary.csv")
        for det in DETS:
            r = pooled[(pooled["detector"] == det) & (pooled["metric"] == "gap_pp")].iloc[0]
            rows.append(dict(seed=s, metric=f"gap_pp_{det}", n=np.nan, point=r["point"], lo=r["ci_lo"], hi=r["ci_hi"]))
        n, pt, lo, hi = position_match(pd.read_csv(d / "per_tooth_predictions.csv"), np.random.default_rng(0), DETS)
        rows.append(dict(seed=s, metric="position_match_pct", n=n, point=pt, lo=lo, hi=hi))
    out = pd.DataFrame(rows)
    s0 = out[out.seed == 0].set_index("metric")
    s1 = out[out.seed == other].set_index("metric")
    out[f"seed{other}_inside_seed0_ci"] = out["metric"].map(
        lambda m: bool(s0.loc[m, "lo"] <= s1.loc[m, "point"] <= s0.loc[m, "hi"]))
    out.to_csv(a.dir / f"seed{other}_grade.csv", index=False, float_format="%.4f")
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
