"""Missing teeth per X-ray in UFBA and DENTEX (RESULTS.md Section 94). Reported only.

    python missing_teeth.py --out results/missing_teeth

A position is missing when the X-ray has no labeled tooth there. Positions 1 to 7 of
each quadrant (28 positions, third molars left out), as in gap_strata.py (Section 87).
Reads boxes.csv and folds.csv (UFBA) and dentex_boxes.csv and dentex_folds.csv (DENTEX);
writes missing_teeth.csv, one row per dataset.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
POS17 = [f"{q}{t}" for q in "1234" for t in "1234567"]
STRATA = [(0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 99, "6+")]


def missing_per_xray(boxes, folds):
    present = boxes.assign(fdi=boxes["fdi"].astype(str)).groupby("image_id")["fdi"].agg(set).to_dict()
    ids = sorted(set(folds["image_id"]))
    return pd.Series({i: sum(f not in present.get(i, set()) for f in POS17) for i in ids})


def summarize(name, m):
    row = dict(dataset=name, xrays=len(m), mean_missing=m.mean(), median_missing=m.median(),
               missing_total=int(m.sum()), missing_pct_of_28=100 * m.sum() / (28 * len(m)))
    for lo, hi, s in STRATA:
        row[f"xrays_{s}"] = int(((m >= lo) & (m <= hi)).sum())
        row[f"share_{s}_pct"] = 100 * ((m >= lo) & (m <= hi)).mean()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = [summarize("UFBA", missing_per_xray(pd.read_csv(HERE / "boxes.csv"), pd.read_csv(HERE / "folds.csv"))),
            summarize("DENTEX", missing_per_xray(pd.read_csv(HERE / "dentex_boxes.csv"),
                                                 pd.read_csv(HERE / "dentex_folds.csv")))]
    out = pd.DataFrame(rows)
    out.to_csv(a.out / "missing_teeth.csv", index=False, float_format="%.4f")
    print(out.T.to_string())


if __name__ == "__main__":
    main()
