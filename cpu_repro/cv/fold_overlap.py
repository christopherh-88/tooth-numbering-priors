"""Overlap between the test folds of the three UFBA splits (RESULTS.md Section 96). Reported only.

    python fold_overlap.py --out results/fold_overlap

For each pair of splits, the largest share of one test fold's X-rays that also lie in a single
test fold of the other split (max over fold pairs of |A and B| / |A|), and the largest Jaccard
index. Reads folds.csv, folds_seed1.csv, folds_seed2.csv; writes fold_overlap.csv.
"""
import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    folds = [pd.read_csv(HERE / f) for f in ("folds.csv", "folds_seed1.csv", "folds_seed2.csv")]
    rows = []
    for x, y in ((0, 1), (0, 2), (1, 2)):
        share = jac = 0.0
        for i in range(5):
            A = set(folds[x][folds[x].fold == i].image_id)
            for j in range(5):
                B = set(folds[y][folds[y].fold == j].image_id)
                share = max(share, len(A & B) / len(A))
                jac = max(jac, len(A & B) / len(A | B))
        rows.append(dict(split_a=x, split_b=y, max_share_of_fold_pct=100 * share, max_jaccard=jac))
    out = pd.DataFrame(rows)
    out.to_csv(a.out / "fold_overlap.csv", index=False, float_format="%.4f")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
