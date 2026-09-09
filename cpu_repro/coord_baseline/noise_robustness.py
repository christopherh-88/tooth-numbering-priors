"""How sharp is the coordinate-only signal? Add Gaussian noise to box
coordinates at test time (train stays clean) and measure how top-1 accuracy
degrades toward the majority baseline.

Motivation: a real trained detector's predicted boxes are never pixel-perfect
copies of ground truth - they carry localization error. This curve tells us
how much coordinate noise the shortcut signal can tolerate before it
disappears, which is the right lens for interpreting Task 2 (comparing a real
detector's error pattern against this coordinate-only diagnostic) once GPU
access resumes. It also stands alone as a sensitivity characterization of
Claim A, distinct from the mitigation experiment (which perturbs *training*
data to unlearn the shortcut, not test-time noise to characterize it).

Noise model: independent Gaussian jitter added to x_center/y_center/width/
height (each in normalized 0-1 image coordinates) at a range of std levels,
then area/aspect_ratio are recomputed from the jittered width/height so the
feature vector stays internally consistent. Width/height are clipped to a
small positive floor to avoid degenerate boxes.

Reuses load_instances/grouped_split/evaluate/FEATURE_COLS/SEEDS/TEST_FRACTION
from build_coord_baseline.py - only the test-time perturbation is new.
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
    FEATURE_COLS,
    SEEDS,
    TEST_FRACTION,
    evaluate,
    grouped_split,
    load_instances,
)

OUTPUT_DIR = Path(__file__).resolve().parent

# Grounded in RESULTS.md Section 13's measured natural per-class positional
# std (x: 0.0267, y: 0.0490) - the grid brackets that scale so we can see
# where accuracy sits relative to "noise comparable to natural class spread."
NOISE_STDS = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]


def add_noise(df: pd.DataFrame, std: float, rng: np.random.RandomState) -> pd.DataFrame:
    df = df.copy()
    if std > 0:
        df["x_center"] = df["x_center"] + rng.normal(0, std, size=len(df))
        df["y_center"] = df["y_center"] + rng.normal(0, std, size=len(df))
        df["width"] = np.clip(df["width"] + rng.normal(0, std, size=len(df)), 1e-3, None)
        df["height"] = np.clip(df["height"] + rng.normal(0, std, size=len(df)), 1e-3, None)
    df["area"] = df["width"] * df["height"]
    df["aspect_ratio"] = df["width"] / df["height"]
    return df


def main():
    df = load_instances()
    print(f"Loaded {len(df)} instances from {df['image_id'].nunique()} images")

    rows = []
    for std in NOISE_STDS:
        for seed in SEEDS:
            train_df, test_df = grouped_split(df, seed, TEST_FRACTION)
            noisy_test = add_noise(test_df, std, np.random.RandomState(1000 + seed))

            x_train = train_df[FEATURE_COLS].to_numpy()
            y_train = train_df["class_id"].to_numpy()
            x_test = noisy_test[FEATURE_COLS].to_numpy()
            y_test = noisy_test["class_id"].to_numpy()

            scaler = StandardScaler().fit(x_train)
            logreg = LogisticRegression(max_iter=2000).fit(scaler.transform(x_train), y_train)
            gbt = HistGradientBoostingClassifier(random_state=0).fit(x_train, y_train)

            for clf_name, y_pred in [
                ("logistic_regression", logreg.predict(scaler.transform(x_test))),
                ("gradient_boosted_tree", gbt.predict(x_test)),
            ]:
                metrics = evaluate(y_test, y_pred, y_train)
                rows.append({"noise_std": std, "seed": seed, "classifier": clf_name, **metrics})
        print(f"noise_std={std} done")

    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT_DIR / "noise_robustness_raw.csv", index=False)

    summary_rows = []
    for (std, clf_name), g in raw.groupby(["noise_std", "classifier"]):
        values = g["top1_acc"].to_numpy()
        mean = values.mean()
        ci = (
            values.std(ddof=1) / np.sqrt(len(values)) * 2.776
            if len(values) > 1
            else 0.0
        )
        summary_rows.append(
            {
                "noise_std": std,
                "classifier": clf_name,
                "top1_acc_mean": mean,
                "top1_acc_ci95": ci,
                "majority_baseline_acc_mean": g["majority_baseline_acc"].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["classifier", "noise_std"])
    summary.to_csv(OUTPUT_DIR / "noise_robustness_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("TOP-1 ACCURACY VS. TEST-TIME COORDINATE NOISE (mean +/- 95% CI, 5 seeds)")
    print("=" * 70)
    for clf_name, g in summary.groupby("classifier"):
        print(f"\n{clf_name} (majority baseline ~{g['majority_baseline_acc_mean'].iloc[0]:.4f})")
        for _, row in g.iterrows():
            print(f"  std={row['noise_std']:.2f}  acc={row['top1_acc_mean']:.4f} +/- {row['top1_acc_ci95']:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for clf_name, g in summary.groupby("classifier"):
        ax.errorbar(g["noise_std"], g["top1_acc_mean"], yerr=g["top1_acc_ci95"], marker="o", label=clf_name, capsize=4)
    ax.axhline(summary["majority_baseline_acc_mean"].iloc[0], color="gray", linestyle="--", label="majority baseline")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Gaussian noise std added to box coordinates (normalized units)")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Coordinate-only accuracy vs. test-time coordinate noise")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "noise_robustness.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved CSVs and plot to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
