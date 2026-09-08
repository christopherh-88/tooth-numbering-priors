"""Does the coordinate-only result depend on which tooth-numbering notation
labels the same 32 physical tooth positions?

FDI_CODES (build_coord_baseline.py) already IS the canonical partition of
box geometry into 32 classes, one per physical tooth position. Universal
and Palmer numbering are both bijective RELABELINGS of that exact same
32-way partition - same teeth, same quadrant/position structure, different
symbols. This script (a) builds and verifies those two relabelings are
clean bijections against FDI, and (b) empirically confirms - rather than
assumes - how label-agnostic multiclass classifiers (LogReg, GBT) behave
across the three notations, on both UFBA-425 and DENTEX, all 5 seeds.

Because this is a bijective relabeling of the same classes, it does not
test the generalization-boundary hypothesis in RESULTS.md Section 11.4
(same label-space structure, same acquisition protocol) - it confirms an
expected label-invariance, not a new data point on the hypothesis. See
Section 11.4 for the full interpretation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "cpu_repro" / "coord_baseline"))
from build_coord_baseline import (  # noqa: E402
    FDI_CODES, FEATURE_COLS, SEEDS, TEST_FRACTION, grouped_split,
    load_instances as load_ufba,
)

sys.path.insert(0, str(REPO_ROOT / "cpu_repro" / "coord_baseline" / "dentex"))
from build_coord_baseline_dentex import load_instances as load_dentex  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent


def fdi_to_universal(code: str) -> int:
    q, p = int(code[0]), int(code[1])
    return {1: 9 - p, 2: 8 + p, 3: 25 - p, 4: 24 + p}[q]


QUADRANT_LETTER = {1: "UR", 2: "UL", 3: "LL", 4: "LR"}


def fdi_to_palmer(code: str) -> str:
    q, p = int(code[0]), int(code[1])
    return f"{QUADRANT_LETTER[q]}{p}"


universal_map = {c: fdi_to_universal(c) for c in FDI_CODES}
palmer_map = {c: fdi_to_palmer(c) for c in FDI_CODES}
universal_relabel = np.array([universal_map[c] for c in FDI_CODES])
palmer_relabel = np.array([palmer_map[c] for c in FDI_CODES], dtype=object)


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    sem = values.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean, t_crit * sem


def run_dataset(name, df):
    print("\n" + "=" * 90)
    print(f"{name}: 5 seeds x 3 conventions x 2 classifiers")
    print("=" * 90)
    per_convention = {c: {"gbt": [], "logreg": []} for c in ["FDI", "Universal", "Palmer"]}

    for seed in SEEDS:
        train_df, test_df = grouped_split(df, seed, TEST_FRACTION)
        x_train = train_df[FEATURE_COLS].to_numpy()
        x_test = test_df[FEATURE_COLS].to_numpy()
        y_train_fdi = train_df["class_id"].to_numpy()
        y_test_fdi = test_df["class_id"].to_numpy()

        for convention, relabel in [("FDI", np.arange(32)), ("Universal", universal_relabel), ("Palmer", palmer_relabel)]:
            y_train = relabel[y_train_fdi]
            y_test = relabel[y_test_fdi]

            clf = HistGradientBoostingClassifier(random_state=0)
            clf.fit(x_train, y_train)
            gbt_acc = float(np.mean(clf.predict(x_test) == y_test))

            scaler = StandardScaler().fit(x_train)
            clf2 = LogisticRegression(max_iter=2000)
            clf2.fit(scaler.transform(x_train), y_train)
            logreg_acc = float(np.mean(clf2.predict(scaler.transform(x_test)) == y_test))

            per_convention[convention]["gbt"].append(gbt_acc)
            per_convention[convention]["logreg"].append(logreg_acc)
        print(f"  seed {seed} done")

    print(f"\n{name} SUMMARY (mean +/- 95% CI, 5 seeds):")
    rows = []
    for convention in ["FDI", "Universal", "Palmer"]:
        gbt_mean, gbt_ci = mean_ci95(per_convention[convention]["gbt"])
        lr_mean, lr_ci = mean_ci95(per_convention[convention]["logreg"])
        print(f"  {convention:10s}  GBT={gbt_mean:.4f}+/-{gbt_ci:.4f}   LogReg={lr_mean:.4f}+/-{lr_ci:.4f}")
        rows.append({"dataset": name, "convention": convention,
                      "gbt_top1_mean": gbt_mean, "gbt_top1_ci95": gbt_ci,
                      "logreg_top1_mean": lr_mean, "logreg_top1_ci95": lr_ci})
    return rows


def main():
    rows = run_dataset("UFBA-425", load_ufba()) + run_dataset("DENTEX", load_dentex())
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "numbering_convention_summary.csv", index=False)
    print(f"\nSaved numbering_convention_summary.csv to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
