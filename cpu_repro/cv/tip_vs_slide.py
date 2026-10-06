"""Tip against slide on the same movers (TIPPING_RULES.md, "also reported").

    python tip_vs_slide.py --slide results/kaggle_closure/closure_intervention/closure_rows.csv \
        --tip results/kaggle_tip/closure_tip/closure_rows.csv

Movers scored in both runs (numbered right by all three detectors on the
intact image); share given the erased tooth's number by all three at full
closure in each, and slide minus tip with a 95% CI from 10,000 bootstrap
draws of X-rays.
"""
import argparse

import numpy as np
import pandas as pd


def all3(path):
    d = pd.read_csv(path)
    d = d[d["condition"] == "closed"]
    g = d.groupby(["image_id", "target"])
    ok = g.apply(lambda x: len(x) == 3 and (x["intact_pred"] == x["mover_class"]).all(), include_groups=False)
    fill = g.apply(lambda x: (x["pred"] == x["target_class"]).all(), include_groups=False)
    return pd.DataFrame(dict(ok=ok, fill=fill))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--tip", required=True)
    a = ap.parse_args()
    j = all3(a.slide).join(all3(a.tip), lsuffix="_s", rsuffix="_t", how="inner")
    j = j[j["ok_s"] & j["ok_t"]].reset_index()
    j["fs"], j["ft"] = j["fill_s"].astype(int), j["fill_t"].astype(int)
    v = j.groupby("image_id").agg(fs=("fs", "sum"), ft=("ft", "sum"), n=("fs", "size")).to_numpy(float)
    b = v[np.random.default_rng(0).integers(0, len(v), (10_000, len(v)))].sum(1)
    lo, hi = np.percentile(100 * (b[:, 0] - b[:, 1]) / b[:, 2], [2.5, 97.5])
    t = v.sum(0)
    print(f"movers {len(j)}: slide {100 * t[0] / t[2]:.2f}%, tip {100 * t[1] / t[2]:.2f}%, "
          f"slide minus tip {100 * (t[0] - t[1]) / t[2]:.2f} ({lo:.2f} to {hi:.2f}); "
          f"tip fills also filled by slide: {int((j['fill_t'] & j['fill_s']).sum())} of {int(j['fill_t'].sum())}")


if __name__ == "__main__":
    main()
