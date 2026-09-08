"""Coordinate-only baseline (see ../build_coord_baseline.py), applied to
DENTEX instead of UFBA-425 - same feature set, same image-level grouped
split, same 5 seeds, same evaluate() function, imported directly (not
reimplemented) from build_coord_baseline.py so results land in the exact
same columns as cpu_repro/coord_baseline/per_seed_results.csv and are
directly comparable.

Data: cpu_repro/anomaly_scan/dentex_raw/dentex_full_fdi_table.csv, built by
anomaly_scan/dentex_raw/build_table.py from DENTEX's quadrant_enumeration
(634 images) and quadrant_enumeration_disease (705 train + 50 validation)
tiers - the two tiers that carry a full FDI code (see
anomaly_scan/dentex_findings.md). 21,806 tooth instances, 1358 images.

DENTEX only covers permanent-dentition quadrants (1-4), so FDI_CODES here
is the identical 32-code list UFBA-425 uses - imported, not retyped, so a
class_id means the same FDI code in both baselines.

Leakage check done before writing this script (see
../../CONVENTIONS.md-adjacent verification, logged in dentex/README.md):
the quadrant_enumeration and quadrant_enumeration_disease tiers were
hash-compared on a 30-image sample and found to be effectively disjoint
physical images (0/30 matches) despite overlapping filename indices - safe
to pool and split by image_id as done here.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_coord_baseline import (  # noqa: E402
    FDI_CODES,
    FEATURE_COLS,
    SEEDS,
    TEST_FRACTION,
    CONFUSION_MATRIX_SEED,
    evaluate,
    grouped_split,
    mean_ci95,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TABLE_PATH = REPO_ROOT / "cpu_repro" / "anomaly_scan" / "dentex_raw" / "dentex_full_fdi_table.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

CODE_TO_IDX = {code: i for i, code in enumerate(FDI_CODES)}


def load_instances() -> pd.DataFrame:
    df = pd.read_csv(TABLE_PATH, dtype={"quadrant": str, "position": str, "fdi": str})
    df["class_id"] = df["fdi"].map(CODE_TO_IDX)
    assert df["class_id"].notna().all(), "unmapped FDI code found - DENTEX table has a code outside FDI_CODES"
    df["class_id"] = df["class_id"].astype(int)
    df["label_file"] = df["image_id"]  # one DENTEX image == one atomic annotation unit, no augmentation crops
    df["area"] = df["width"] * df["height"]
    df["aspect_ratio"] = df["width"] / df["height"].replace(0, np.nan)
    df["aspect_ratio"] = df["aspect_ratio"].fillna(df["aspect_ratio"].median())
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_instances()
    print(f"Loaded {len(df)} tooth instances from {df['image_id'].nunique()} DENTEX images "
          f"(full FDI code, quadrant_enumeration + quadrant_enumeration_disease tiers)")

    classifiers = {
        "logistic_regression": lambda: LogisticRegression(max_iter=2000),
        "gradient_boosted_tree": lambda: HistGradientBoostingClassifier(random_state=0),
    }

    per_seed_results = {name: [] for name in classifiers}
    confusion_data = {}

    for seed in SEEDS:
        train_df, test_df = grouped_split(df, seed, TEST_FRACTION)
        x_train = train_df[FEATURE_COLS].to_numpy()
        y_train = train_df["class_id"].to_numpy()
        x_test = test_df[FEATURE_COLS].to_numpy()
        y_test = test_df["class_id"].to_numpy()

        scaler = StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        for name, make_clf in classifiers.items():
            clf = make_clf()
            if name == "logistic_regression":
                clf.fit(x_train_scaled, y_train)
                y_pred = clf.predict(x_test_scaled)
            else:
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)

            metrics = evaluate(y_test, y_pred, y_train)
            metrics["seed"] = seed
            per_seed_results[name].append(metrics)

            if seed == CONFUSION_MATRIX_SEED:
                confusion_data[name] = (y_test, y_pred)

        print(f"seed {seed}: train_images={train_df['image_id'].nunique()} "
              f"test_images={test_df['image_id'].nunique()} "
              f"train_instances={len(train_df)} test_instances={len(test_df)}")

    print()
    print("=" * 88)
    print("SUMMARY (mean +/- 95% CI over 5 seeds, image-level grouped split) - DENTEX")
    print("=" * 88)

    summary_rows = []
    metric_keys = [
        "top1_acc", "quadrant_acc", "tooth_type_acc",
        "majority_baseline_acc", "mirror_quadrant_error_frac", "neighbor_error_frac",
    ]
    for name, results in per_seed_results.items():
        print(f"\n{name}")
        row = {"classifier": name}
        for key in metric_keys:
            values = [r[key] for r in results]
            mean, ci = mean_ci95(values)
            print(f"  {key:28s} {mean:.4f} +/- {ci:.4f}")
            row[f"{key}_mean"] = mean
            row[f"{key}_ci95"] = ci
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "summary.csv", index=False)

    per_seed_rows = [{"classifier": name, **r} for name, results in per_seed_results.items() for r in results]
    pd.DataFrame(per_seed_rows).to_csv(OUTPUT_DIR / "per_seed_results.csv", index=False)

    for name, (y_test, y_pred) in confusion_data.items():
        cm = confusion_matrix(y_test, y_pred, labels=list(range(32)))
        pd.DataFrame(cm, index=FDI_CODES, columns=FDI_CODES).to_csv(
            OUTPUT_DIR / f"confusion_matrix_{name}_seed{CONFUSION_MATRIX_SEED}.csv"
        )
        fig, ax = plt.subplots(figsize=(11, 10))
        im = ax.imshow(cm, cmap="viridis")
        ax.set_xticks(range(32)); ax.set_yticks(range(32))
        ax.set_xticklabels(FDI_CODES, rotation=90, fontsize=7)
        ax.set_yticklabels(FDI_CODES, fontsize=7)
        ax.set_xlabel("Predicted FDI code"); ax.set_ylabel("True FDI code")
        ax.set_title(f"{name} - DENTEX confusion matrix (seed {CONFUSION_MATRIX_SEED})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"confusion_matrix_{name}_seed{CONFUSION_MATRIX_SEED}.png", dpi=150)
        plt.close(fig)

    print(f"\nSaved summary.csv, per_seed_results.csv, and confusion matrices to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
