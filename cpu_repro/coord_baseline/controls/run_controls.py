"""Control conditions for the coordinate-only baseline
(../build_coord_baseline.py).

Four conditions, same 5 seeds, same image-level grouped split, same two
classifiers:

1. full (real)               - x_center, y_center, width, height, area,
                                aspect_ratio, as in the main baseline.
2. full (shuffled per image) - the same six features, but for each image
                                (each YOLO label file) the feature ROWS are
                                randomly permuted among that image's teeth
                                before the class labels are attached. This
                                destroys the box-geometry <-> tooth-identity
                                correspondence while preserving exactly the
                                same distribution of box shapes/positions per
                                image (same boxes, wrong owners). If the main
                                baseline's accuracy came from a genuine
                                geometry-identity relationship rather than
                                some artifact of how the data is laid out,
                                this condition should collapse to roughly the
                                majority-class baseline.
3. position only (x,y)       - x_center, y_center only. Tests how much
                                signal is in "where" alone.
4. size only (w,h)           - width, height only. Tests how much signal is
                                in "how big/what shape" alone, with no
                                positional information at all.

Train/test membership (which images are in the held-out fold) is identical
across all four conditions for a given seed - only the feature values differ
- so the four accuracy numbers are a fair side-by-side comparison.
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_coord_baseline import (  # noqa: E402
    FEATURE_COLS,
    SEEDS,
    TEST_FRACTION,
    evaluate,
    grouped_split,
    load_instances,
    mean_ci95,
)

OUTPUT_DIR = Path(__file__).resolve().parent

POSITION_COLS = ["x_center", "y_center"]
SIZE_COLS = ["width", "height"]

CONDITIONS = [
    "full (real)",
    "full (shuffled per image)",
    "position only (x,y)",
    "size only (w,h)",
]


def shuffle_features_within_image(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return a copy of df where, within each label_file (= one image's set
    of annotated teeth), the FEATURE_COLS values are randomly permuted
    across rows. class_id, image_id, label_file stay attached to their
    original row - only the box geometry describing each row is scrambled."""
    rng = np.random.RandomState(seed)
    shuffled = df.copy()
    feature_matrix = df[FEATURE_COLS].to_numpy()
    result = feature_matrix.copy()
    for _, idx in df.groupby("label_file").indices.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        perm = rng.permutation(len(idx))
        result[idx] = feature_matrix[idx][perm]
    shuffled[FEATURE_COLS] = result
    return shuffled


def features_for_condition(condition, train_df, test_df, shuffled_train_df, shuffled_test_df):
    if condition == "full (real)":
        return train_df[FEATURE_COLS].to_numpy(), test_df[FEATURE_COLS].to_numpy()
    if condition == "full (shuffled per image)":
        return (
            shuffled_train_df[FEATURE_COLS].to_numpy(),
            shuffled_test_df[FEATURE_COLS].to_numpy(),
        )
    if condition == "position only (x,y)":
        return train_df[POSITION_COLS].to_numpy(), test_df[POSITION_COLS].to_numpy()
    if condition == "size only (w,h)":
        return train_df[SIZE_COLS].to_numpy(), test_df[SIZE_COLS].to_numpy()
    raise ValueError(condition)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_instances()
    print(f"Loaded {len(df)} instances from {df['image_id'].nunique()} images / "
          f"{df['label_file'].nunique()} label files")

    classifiers = {
        "logistic_regression": lambda: LogisticRegression(max_iter=2000),
        "gradient_boosted_tree": lambda: HistGradientBoostingClassifier(random_state=0),
    }

    per_seed_rows = []

    for seed in SEEDS:
        train_df, test_df = grouped_split(df, seed, TEST_FRACTION)
        shuffled_df = shuffle_features_within_image(df, seed)
        shuffled_train_df = shuffled_df.loc[train_df.index]
        shuffled_test_df = shuffled_df.loc[test_df.index]

        y_train = train_df["class_id"].to_numpy()
        y_test = test_df["class_id"].to_numpy()

        for condition in CONDITIONS:
            x_train, x_test = features_for_condition(
                condition, train_df, test_df, shuffled_train_df, shuffled_test_df
            )

            for clf_name, make_clf in classifiers.items():
                clf = make_clf()
                if clf_name == "logistic_regression":
                    scaler = StandardScaler().fit(x_train)
                    clf.fit(scaler.transform(x_train), y_train)
                    y_pred = clf.predict(scaler.transform(x_test))
                else:
                    clf.fit(x_train, y_train)
                    y_pred = clf.predict(x_test)

                metrics = evaluate(y_test, y_pred, y_train)
                metrics.update({"seed": seed, "condition": condition, "classifier": clf_name})
                per_seed_rows.append(metrics)

        print(f"seed {seed} done "
              f"(train_images={train_df['image_id'].nunique()} test_images={test_df['image_id'].nunique()})")

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(OUTPUT_DIR / "per_seed_results.csv", index=False)

    metric_keys = ["top1_acc", "quadrant_acc", "tooth_type_acc", "majority_baseline_acc"]
    summary_rows = []
    for (condition, clf_name), g in per_seed_df.groupby(["condition", "classifier"]):
        row = {"condition": condition, "classifier": clf_name}
        for key in metric_keys:
            mean, ci = mean_ci95(g[key].to_numpy())
            row[f"{key}_mean"] = mean
            row[f"{key}_ci95"] = ci
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df["condition"] = pd.Categorical(summary_df["condition"], categories=CONDITIONS, ordered=True)
    summary_df = summary_df.sort_values(["condition", "classifier"])
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print("\n" + "=" * 100)
    print("SUMMARY (mean +/- 95% CI over 5 seeds, image-level grouped split, same split across conditions)")
    print("=" * 100)
    for condition in CONDITIONS:
        print(f"\n{condition}")
        sub = summary_df[summary_df["condition"] == condition]
        for _, row in sub.iterrows():
            print(f"  {row['classifier']:24s} "
                  f"top1={row['top1_acc_mean']:.4f}+/-{row['top1_acc_ci95']:.4f}  "
                  f"quadrant={row['quadrant_acc_mean']:.4f}+/-{row['quadrant_acc_ci95']:.4f}  "
                  f"type={row['tooth_type_acc_mean']:.4f}+/-{row['tooth_type_acc_ci95']:.4f}  "
                  f"majority_baseline={row['majority_baseline_acc_mean']:.4f}")

    # bar chart: top-1 accuracy across the four conditions, one bar pair per condition
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(CONDITIONS))
    width = 0.35
    for i, clf_name in enumerate(classifiers):
        sub = summary_df[summary_df["classifier"] == clf_name].set_index("condition").reindex(CONDITIONS)
        ax.bar(
            x + i * width, sub["top1_acc_mean"], width,
            yerr=sub["top1_acc_ci95"], label=clf_name, capsize=4,
        )
    majority = summary_df["majority_baseline_acc_mean"].mean()
    ax.axhline(majority, color="gray", linestyle="--", linewidth=1, label=f"majority-class baseline ({majority:.3f})")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(CONDITIONS, rotation=15, ha="right")
    ax.set_ylabel("Top-1 accuracy (32-way)")
    ax.set_title("Coordinate-signal ablation: real vs shuffled vs position-only vs size-only")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "controls_comparison.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved summary.csv, per_seed_results.csv, and controls_comparison.png to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
