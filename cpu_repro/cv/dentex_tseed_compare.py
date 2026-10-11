"""DENTEX second training seed: side by side, pooled, overlap (cpu_repro/cv/DENTEX_TSEED_RULES.md). Reported only.

    python dentex_tseed_compare.py --seed0 <Section 86 dentex_teeth.csv> --seed1 <tseed1 dentex_teeth.csv> --out <dir>

Both tables come from dentex_external.py --saved, so they list the same labeled teeth in the same order.
Writes tseed_compare.csv. Definitions are the ones of dentex_external.py (rules 3 and 4):
position share = joint & pos_match over joint; gap ratio = (joint at gap / joint) / (all-right at gap / all-right).
Pooled estimates stack both seeds' rows and resample X-rays jointly (an X-ray brings both seeds' rows).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 10_000
FLAGS = ["joint", "pos_match", "all_right", "joint_gap", "right_gap"]
DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]


def share(u):  # u columns: n, joint, pos_match, all_right, joint_gap, right_gap
    return 100 * u[2] / u[1]


def ratio(u):
    return (u[4] / u[1]) / (u[5] / u[3])


def per_xray(t, cols):
    return t.assign(_n=1).groupby("image_id")[["_n"] + cols].sum()


def ci(per, stat, rng):
    s = per.to_numpy(float)
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    v = stat(s[idx].sum(1).T)
    return stat(s.sum(0)), *np.nanpercentile(v, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0", type=Path, required=True)
    ap.add_argument("--seed1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    t0, t1 = pd.read_csv(a.seed0), pd.read_csv(a.seed1)
    assert len(t0) == len(t1), (len(t0), len(t1))
    assert (t0["image_id"].to_numpy() == t1["image_id"].to_numpy()).all()
    assert (t0["class_id"].to_numpy() == t1["class_id"].to_numpy()).all()
    for t in (t0, t1):
        for c in FLAGS:
            t[c] = t[c].astype(int)
    rows = []

    def add(group, metric, per, stat, **kw):
        p, lo, hi = ci(per, stat, rng)
        rows.append(dict(group=group, metric=metric, point=p, lo=lo, hi=hi, **kw))

    for name, t in (("seed0", t0), ("seed1", t1)):
        per = per_xray(t, FLAGS)
        n = int(t["joint"].sum())
        add(name, "position_share_pct", per, share, n=n)
        add(name, "gap_ratio", per, ratio, n=n)
        for d in DETS:
            c = t[f"{d}_correct"].astype(int)
            m = t[f"{d}_missed"].astype(int)
            k = t.assign(_c=c, _m=m, _k=t["coord_correct"].astype(int))
            add(name, f"top1_{d}_pct", per_xray(k, ["_c"]), lambda u: 100 * u[1] / u[0])
            add(name, f"missed_{d}_pct", per_xray(k, ["_m"]), lambda u: 100 * u[1] / u[0])
            add(name, f"margin_{d}_pp", per_xray(k, ["_c", "_k"]), lambda u: 100 * (u[1] - u[2]) / u[0])

    pooled = pd.concat([t0, t1], ignore_index=True)
    n = int(pooled["joint"].sum())
    per = per_xray(pooled, FLAGS)
    add("pooled", "position_share_pct", per, share, n=n)
    add("pooled", "gap_ratio", per, ratio, n=n)

    j0 = set(map(tuple, t0.loc[t0["joint"] == 1, ["image_id", "tid"]].to_numpy()))
    j1 = set(map(tuple, t1.loc[t1["joint"] == 1, ["image_id", "tid"]].to_numpy()))
    inter, union = len(j0 & j1), len(j0 | j1)
    rows.append(dict(group="overlap", metric="shared_error_sets", point=inter / union, lo=np.nan, hi=np.nan,
                     n=union, seed0=len(j0), seed1=len(j1), both=inter,
                     in_other_given_seed0=inter / len(j0), in_other_given_seed1=inter / len(j1)))
    S = pd.DataFrame(rows)
    S.to_csv(a.out / "tseed_compare.csv", index=False)
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
