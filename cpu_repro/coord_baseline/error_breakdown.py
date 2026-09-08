"""Follow-up analysis on top of build_coord_baseline.py:

1. Do errors concentrate in particular tooth types? (Hypothesis: premolars
   and molars are more self-similar in size/shape than incisors, so the
   coordinate-only model should do worse on them.)
2. Does accuracy vary with how many teeth are annotated in the image?
   (Hypothesis: fewer teeth = weaker positional context, e.g. no neighbors
   to triangulate against, so accuracy should drop.)

Reuses load_instances/grouped_split/FDI_CODES/FEATURE_COLS from
build_coord_baseline.py so the leakage-safe grouped split and feature
definitions stay identical to the main baseline - only the reporting is new.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_coord_baseline import (  # noqa: E402
    FDI_CODES,
    FEATURE_COLS,
    SEEDS,
    TEST_FRACTION,
    grouped_split,
    load_instances,
)

OUTPUT_DIR = Path(__file__).resolve().parent

TOOTH_TYPE_GROUPS = {
    "1": "Incisor",   # FDI 2nd digit 1 (central incisor)
    "2": "Incisor",   # 2nd digit 2 (lateral incisor)
    "3": "Canine",
    "4": "Premolar",
    "5": "Premolar",
    "6": "Molar",
    "7": "Molar",
    "8": "Molar",     # third molar / wisdom tooth
}

# Bin edges chosen from the observed distribution of teeth-per-crop
# (min 1, 25th pct 25, median 30, 75th pct 32, max 34): a handful of crops
# have very few annotated teeth (partial dentition / implants / edge crops),
# most have close to a full arch.
TEETH_COUNT_BINS = [0, 15, 23, 28, 32, 100]
TEETH_COUNT_LABELS = ["1-15", "16-23", "24-28", "29-32", "33+"]


def tooth_group_of(class_id: np.ndarray) -> np.ndarray:
    codes = np.array(FDI_CODES)[class_id]
    return np.array([TOOTH_TYPE_GROUPS[c[1]] for c in codes])


def run_all_seeds(df):
    """Train both classifiers across all seeds, return one dataframe of
    every test-set prediction (pooled across seeds) with metadata attached."""
    all_preds = []

    for seed in SEEDS:
        train_df, test_df = grouped_split(df, seed, TEST_FRACTION)
        x_train = train_df[FEATURE_COLS].to_numpy()
        y_train = train_df["class_id"].to_numpy()
        x_test = test_df[FEATURE_COLS].to_numpy()
        y_test = test_df["class_id"].to_numpy()

        # teeth-count-in-crop context feature, computed from train/test
        # membership only (no leakage: count is intrinsic to the label file)
        test_teeth_count = test_df.groupby("label_file")["class_id"].transform("size").to_numpy()

        scaler = StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        logreg = LogisticRegression(max_iter=2000).fit(x_train_scaled, y_train)
        gbt = HistGradientBoostingClassifier(random_state=0).fit(x_train, y_train)

        preds = pd.DataFrame(
            {
                "seed": seed,
                "true_class": y_test,
                "pred_logreg": logreg.predict(x_test_scaled),
                "pred_gbt": gbt.predict(x_test),
                "teeth_in_crop": test_teeth_count,
                "label_file": test_df["label_file"].to_numpy(),
            }
        )
        all_preds.append(preds)
        print(f"seed {seed} done ({len(test_df)} test instances)")

    return pd.concat(all_preds, ignore_index=True)


def per_tooth_type_accuracy(preds: pd.DataFrame):
    rows = []
    preds = preds.copy()
    preds["true_group"] = tooth_group_of(preds["true_class"].to_numpy())

    for clf_col, clf_name in [("pred_logreg", "logistic_regression"), ("pred_gbt", "gradient_boosted_tree")]:
        preds[f"{clf_col}_correct"] = preds["true_class"] == preds[clf_col]
        # per-seed accuracy per group, then mean/CI over seeds so this is
        # comparable in spirit to the main baseline's mean+-95% CI numbers.
        per_seed = (
            preds.groupby(["seed", "true_group"])[f"{clf_col}_correct"]
            .mean()
            .reset_index()
        )
        for group, g in per_seed.groupby("true_group"):
            values = g[f"{clf_col}_correct"].to_numpy()
            mean = values.mean()
            ci = (
                values.std(ddof=1) / np.sqrt(len(values)) * 2.776  # t(0.975, df=4)
                if len(values) > 1
                else 0.0
            )
            rows.append(
                {
                    "classifier": clf_name,
                    "tooth_group": group,
                    "accuracy_mean": mean,
                    "accuracy_ci95": ci,
                    "n_instances": int(preds[preds["true_group"] == group].shape[0]),
                }
            )

    # also per-individual-FDI-class accuracy (finer grain than the 4 groups)
    per_class_rows = []
    for clf_col, clf_name in [("pred_logreg", "logistic_regression"), ("pred_gbt", "gradient_boosted_tree")]:
        preds[f"{clf_col}_correct"] = preds["true_class"] == preds[clf_col]
        per_seed = preds.groupby(["seed", "true_class"])[f"{clf_col}_correct"].mean().reset_index()
        for cls, g in per_seed.groupby("true_class"):
            values = g[f"{clf_col}_correct"].to_numpy()
            per_class_rows.append(
                {
                    "classifier": clf_name,
                    "fdi_code": FDI_CODES[int(cls)],
                    "tooth_group": TOOTH_TYPE_GROUPS[FDI_CODES[int(cls)][1]],
                    "accuracy_mean": values.mean(),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(per_class_rows)


def accuracy_vs_teeth_count(preds: pd.DataFrame):
    preds = preds.copy()
    preds["teeth_count_bin"] = pd.cut(
        preds["teeth_in_crop"], bins=TEETH_COUNT_BINS, labels=TEETH_COUNT_LABELS
    )

    rows = []
    for clf_col, clf_name in [("pred_logreg", "logistic_regression"), ("pred_gbt", "gradient_boosted_tree")]:
        preds[f"{clf_col}_correct"] = preds["true_class"] == preds[clf_col]
        per_seed = (
            preds.groupby(["seed", "teeth_count_bin"], observed=True)[f"{clf_col}_correct"]
            .agg(["mean", "count"])
            .reset_index()
        )
        for bin_label, g in per_seed.groupby("teeth_count_bin", observed=True):
            values = g["mean"].to_numpy()
            if len(values) == 0:
                continue
            mean = values.mean()
            ci = values.std(ddof=1) / np.sqrt(len(values)) * 2.776 if len(values) > 1 else 0.0
            rows.append(
                {
                    "classifier": clf_name,
                    "teeth_count_bin": bin_label,
                    "accuracy_mean": mean,
                    "accuracy_ci95": ci,
                    "n_instances_pooled": int(g["count"].sum()),
                }
            )

    # correlation between raw teeth-in-crop count and per-instance correctness
    corr_rows = []
    for clf_col, clf_name in [("pred_logreg", "logistic_regression"), ("pred_gbt", "gradient_boosted_tree")]:
        preds[f"{clf_col}_correct"] = (preds["true_class"] == preds[clf_col]).astype(float)
        corr = preds["teeth_in_crop"].corr(preds[f"{clf_col}_correct"])
        corr_rows.append({"classifier": clf_name, "pearson_r_teeth_count_vs_correct": corr})

    return pd.DataFrame(rows), pd.DataFrame(corr_rows)


def plot_tooth_group(group_df):
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = ["Incisor", "Canine", "Premolar", "Molar"]
    width = 0.35
    x = np.arange(len(groups))
    for i, clf_name in enumerate(["logistic_regression", "gradient_boosted_tree"]):
        sub = group_df[group_df["classifier"] == clf_name].set_index("tooth_group").reindex(groups)
        ax.bar(x + i * width, sub["accuracy_mean"], width, yerr=sub["accuracy_ci95"], label=clf_name, capsize=4)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Coordinate-only accuracy by tooth type")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_by_tooth_type.png", dpi=150)
    plt.close(fig)


def plot_teeth_count(count_df):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = TEETH_COUNT_LABELS
    width = 0.35
    x = np.arange(len(bins))
    for i, clf_name in enumerate(["logistic_regression", "gradient_boosted_tree"]):
        sub = count_df[count_df["classifier"] == clf_name].set_index("teeth_count_bin").reindex(bins)
        ax.bar(x + i * width, sub["accuracy_mean"], width, yerr=sub["accuracy_ci95"], label=clf_name, capsize=4)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(bins)
    ax.set_xlabel("Teeth annotated in crop")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Coordinate-only accuracy vs. teeth-in-crop count")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_by_teeth_count.png", dpi=150)
    plt.close(fig)


def main():
    df = load_instances()
    print(f"Loaded {len(df)} instances from {df['image_id'].nunique()} images / "
          f"{df['label_file'].nunique()} label files")

    preds = run_all_seeds(df)
    preds.to_csv(OUTPUT_DIR / "pooled_predictions.csv", index=False)

    group_df, per_class_df = per_tooth_type_accuracy(preds)
    group_df.to_csv(OUTPUT_DIR / "accuracy_by_tooth_type.csv", index=False)
    per_class_df.to_csv(OUTPUT_DIR / "accuracy_by_fdi_class.csv", index=False)

    print("\n" + "=" * 70)
    print("ACCURACY BY TOOTH TYPE (mean +/- 95% CI over 5 seeds)")
    print("=" * 70)
    for clf_name, g in group_df.groupby("classifier"):
        print(f"\n{clf_name}")
        for _, row in g.sort_values("accuracy_mean", ascending=False).iterrows():
            print(f"  {row['tooth_group']:10s} {row['accuracy_mean']:.4f} +/- {row['accuracy_ci95']:.4f}  (n={row['n_instances']})")

    count_df, corr_df = accuracy_vs_teeth_count(preds)
    count_df.to_csv(OUTPUT_DIR / "accuracy_by_teeth_count.csv", index=False)
    corr_df.to_csv(OUTPUT_DIR / "teeth_count_correlation.csv", index=False)

    print("\n" + "=" * 70)
    print("ACCURACY VS TEETH-IN-CROP COUNT (mean +/- 95% CI over 5 seeds)")
    print("=" * 70)
    for clf_name, g in count_df.groupby("classifier"):
        print(f"\n{clf_name}")
        for _, row in g.iterrows():
            print(f"  {row['teeth_count_bin']:8s} {row['accuracy_mean']:.4f} +/- {row['accuracy_ci95']:.4f}  (n={row['n_instances_pooled']})")

    print("\nPearson correlation (teeth-in-crop count vs. per-instance correctness):")
    for _, row in corr_df.iterrows():
        print(f"  {row['classifier']:24s} r={row['pearson_r_teeth_count_vs_correct']:.4f}")

    plot_tooth_group(group_df)
    plot_teeth_count(count_df)

    print(f"\nSaved CSVs and plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
