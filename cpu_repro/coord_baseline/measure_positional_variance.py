"""Measure the dataset's natural per-class positional variance.

Purpose: the mitigation experiment (mitigation/README.md) needs a
geometry-jitter magnitude that is deliberately LARGER than the positional
variance already naturally present in the data - otherwise the jitter
augmentation wouldn't meaningfully perturb anything the model doesn't
already see. This script measures that natural variance directly instead
of guessing a jitter size.

For each FDI class, per-class spread of (x_center, y_center) across all
instances of that class in the pooled dataset is the natural "how much
does this tooth's position wander across different images/patients"
signal. Reuses the same instance loader as build_coord_baseline.py so
this is measured on the identical data/convention (normalized [0,1]
YOLO coordinates), not a re-derived or re-scaled version of it.

No training, no model fitting - purely descriptive statistics.
"""
import numpy as np
import pandas as pd

from build_coord_baseline import load_instances, FDI_CODES

pd.set_option("display.width", 120)


def main():
    df = load_instances()
    print(f"Loaded {len(df)} instances across {df['image_id'].nunique()} "
          f"source images.\n")

    rows = []
    for class_id, code in enumerate(FDI_CODES):
        sub = df[df["class_id"] == class_id]
        if len(sub) < 2:
            continue
        rows.append({
            "fdi_code": code,
            "n": len(sub),
            "x_std": sub["x_center"].std(),
            "y_std": sub["y_center"].std(),
            "x_iqr": sub["x_center"].quantile(0.75) - sub["x_center"].quantile(0.25),
            "y_iqr": sub["y_center"].quantile(0.75) - sub["y_center"].quantile(0.25),
        })
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv("positional_variance_by_class.csv", index=False)

    print("Per-class positional spread (normalized [0,1] coordinates):")
    print(stats_df.to_string(index=False))

    print("\n=== Summary across all 32 classes ===")
    print(f"mean x_std across classes: {stats_df['x_std'].mean():.4f}")
    print(f"mean y_std across classes: {stats_df['y_std'].mean():.4f}")
    print(f"median x_std across classes: {stats_df['x_std'].median():.4f}")
    print(f"median y_std across classes: {stats_df['y_std'].median():.4f}")
    print(f"mean x_iqr across classes: {stats_df['x_iqr'].mean():.4f}")
    print(f"mean y_iqr across classes: {stats_df['y_iqr'].mean():.4f}")

    # image-level canvas fraction -> pixel terms is dataset/model-resolution
    # dependent, so this is reported in normalized units; convert at
    # implementation time using whatever input resolution the training
    # notebook actually uses (currently 512x512 after resize_img).
    print("\nSuggested jitter magnitude (normalized units): at least "
          f"{2 * stats_df['x_std'].mean():.4f} in x and "
          f"{2 * stats_df['y_std'].mean():.4f} in y (2x mean per-class std) "
          "to deliberately exceed natural variance rather than mimic it. "
          "This is a starting point, not a tuned value - revisit once the "
          "mitigation experiment is actually run.")


if __name__ == "__main__":
    main()
