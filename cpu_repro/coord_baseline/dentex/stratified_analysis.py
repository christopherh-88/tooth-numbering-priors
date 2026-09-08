"""Does coordinate-only accuracy drop on DENTEX's known-anomalous teeth?

Two strata, evaluated separately against a canonical remainder, using the
exact same trained models and evaluate() function as
build_coord_baseline_dentex.py (same 5 seeds, same image-level grouped
split - these are NOT new models, they are the same per-seed classifiers,
sliced by which stratum a test instance falls into):

1. "dissociation" - all test-fold instances belonging to one of the 101
   images flagged by scan_dentex.py (duplicate FDI code and/or a
   position-vs-quadrant violation - see anomaly_scan/dentex_findings.md).
   This is an IMAGE-level flag: every tooth in a flagged image is in this
   stratum, not just the specific anomalous box.
2. "impacted" - all test-fold instances whose diagnosis is "Impacted"
   (644 total in the full dataset, before splitting). This is an
   INSTANCE-level flag.

"canonical" is every other test-fold instance (not in a flagged image, not
impacted-diagnosed) - the reference point both strata are compared against.

These two strata overlap a little (an impacted tooth can also live in a
flagged image) - reported, not hidden.

Because both strata are a small slice of 1358 images, and each seed's test
fold is itself only ~20% of the data, per-seed stratum sample sizes can be
small. Sizes are reported explicitly and any stratum below the size
thresholds set out below is flagged as too small to support a reliable
comparison, rather than reporting a confident-looking number over few
instances.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_coord_baseline import FEATURE_COLS, SEEDS, TEST_FRACTION, evaluate, grouped_split, mean_ci95  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_coord_baseline_dentex import load_instances  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SCAN_PATH = REPO_ROOT / "cpu_repro" / "anomaly_scan" / "dentex_results" / "full_scan_dentex.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

# Flag a stratum as too small to trust if either bound is crossed.
MIN_POOLED_N = 100        # summed across all 5 seeds' test folds
MIN_PER_SEED_N = 20       # smallest acceptable single-seed test-fold count


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_instances()
    scan = pd.read_csv(SCAN_PATH)
    flagged_images = set(scan.loc[(scan["n_duplicate_codes"] > 0) | (scan["n_position_violations"] > 0), "image_id"])
    print(f"Flagged (dissociation) images: {len(flagged_images)}")

    df["is_flagged_image"] = df["image_id"].isin(flagged_images)
    df["is_impacted"] = df["diagnosis"] == "Impacted"
    print(f"Total instances in flagged images: {df['is_flagged_image'].sum()}")
    print(f"Total impacted-diagnosis instances: {df['is_impacted'].sum()}")
    print(f"Instances that are both: {(df['is_flagged_image'] & df['is_impacted']).sum()}")

    classifiers = {
        "logistic_regression": lambda: LogisticRegression(max_iter=2000),
        "gradient_boosted_tree": lambda: HistGradientBoostingClassifier(random_state=0),
    }

    strata_names = ["canonical", "dissociation", "impacted"]
    per_seed_rows = []

    for seed in SEEDS:
        train_df, test_df = grouped_split(df, seed, TEST_FRACTION)
        x_train = train_df[FEATURE_COLS].to_numpy()
        y_train = train_df["class_id"].to_numpy()
        x_test = test_df[FEATURE_COLS].to_numpy()
        y_test = test_df["class_id"].to_numpy()

        is_flagged = test_df["is_flagged_image"].to_numpy()
        is_impacted = test_df["is_impacted"].to_numpy()
        is_canonical = ~is_flagged & ~is_impacted

        masks = {"canonical": is_canonical, "dissociation": is_flagged, "impacted": is_impacted}

        scaler = StandardScaler().fit(x_train)
        x_train_scaled, x_test_scaled = scaler.transform(x_train), scaler.transform(x_test)

        for clf_name, make_clf in classifiers.items():
            clf = make_clf()
            if clf_name == "logistic_regression":
                clf.fit(x_train_scaled, y_train)
                y_pred = clf.predict(x_test_scaled)
            else:
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)

            for stratum in strata_names:
                m = masks[stratum]
                n = int(m.sum())
                if n == 0:
                    metrics = {"top1_acc": float("nan"), "quadrant_acc": float("nan"),
                               "tooth_type_acc": float("nan"), "majority_baseline_acc": float("nan"),
                               "mirror_quadrant_error_frac": float("nan"), "neighbor_error_frac": float("nan"),
                               "n_test": 0}
                else:
                    metrics = evaluate(y_test[m], y_pred[m], y_train)
                    metrics["n_test"] = n
                metrics.update({"seed": seed, "classifier": clf_name, "stratum": stratum})
                per_seed_rows.append(metrics)

        print(f"seed {seed}: test_instances total={len(test_df)} "
              f"canonical={is_canonical.sum()} dissociation={is_flagged.sum()} impacted={is_impacted.sum()}")

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(OUTPUT_DIR / "stratified_per_seed_results.csv", index=False)

    metric_keys = ["top1_acc", "quadrant_acc", "tooth_type_acc"]
    summary_rows = []
    for (stratum, clf_name), g in per_seed_df.groupby(["stratum", "classifier"]):
        row = {"stratum": stratum, "classifier": clf_name,
               "pooled_n": int(g["n_test"].sum()), "min_seed_n": int(g["n_test"].min())}
        for key in metric_keys:
            valid = g.dropna(subset=[key])
            if len(valid) == 0:
                row[f"{key}_mean"], row[f"{key}_ci95"] = float("nan"), float("nan")
            else:
                mean, ci = mean_ci95(valid[key].to_numpy())
                row[f"{key}_mean"], row[f"{key}_ci95"] = mean, ci
        row["too_small"] = row["pooled_n"] < MIN_POOLED_N or row["min_seed_n"] < MIN_PER_SEED_N
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    order = {"canonical": 0, "dissociation": 1, "impacted": 2}
    summary_df["_order"] = summary_df["stratum"].map(order)
    summary_df = summary_df.sort_values(["_order", "classifier"]).drop(columns="_order")
    summary_df.to_csv(OUTPUT_DIR / "stratified_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("STRATIFIED COMPARISON (mean +/- 95% CI over 5 seeds; same trained models as the main baseline)")
    print("=" * 100)
    for stratum in ["canonical", "dissociation", "impacted"]:
        print(f"\n{stratum}")
        sub = summary_df[summary_df["stratum"] == stratum]
        for _, row in sub.iterrows():
            flag = "  <<< TOO SMALL TO TRUST" if row["too_small"] else ""
            print(f"  {row['classifier']:24s} top1={row['top1_acc_mean']:.4f}+/-{row['top1_acc_ci95']:.4f}  "
                  f"quadrant={row['quadrant_acc_mean']:.4f}+/-{row['quadrant_acc_ci95']:.4f}  "
                  f"type={row['tooth_type_acc_mean']:.4f}+/-{row['tooth_type_acc_ci95']:.4f}  "
                  f"pooled_n={row['pooled_n']} min_seed_n={row['min_seed_n']}{flag}")

    print("\nDrop vs. canonical (top-1 accuracy, gradient_boosted_tree):")
    canon = summary_df[(summary_df["stratum"] == "canonical") & (summary_df["classifier"] == "gradient_boosted_tree")]["top1_acc_mean"].iloc[0]
    for stratum in ["dissociation", "impacted"]:
        row = summary_df[(summary_df["stratum"] == stratum) & (summary_df["classifier"] == "gradient_boosted_tree")].iloc[0]
        drop = canon - row["top1_acc_mean"]
        flag = " (too small to trust)" if row["too_small"] else ""
        print(f"  {stratum}: {row['top1_acc_mean']:.4f} vs canonical {canon:.4f} -> drop of {drop:+.4f}{flag}")

    print(f"\nSaved stratified_summary.csv and stratified_per_seed_results.csv to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
