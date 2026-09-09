"""Which geometric feature actually carries the FDI-class signal?

The main baseline (build_coord_baseline.py) uses all six features
(x_center, y_center, width, height, area, aspect_ratio) together and reports
one accuracy number. This doesn't say whether the ~69% top-1 accuracy comes
mainly from *position* (x/y - "where on the jaw is this box"), from *size/
shape* (width/height/area/aspect_ratio - which could leak identity via
crowding or systematic annotation-size differences per tooth type), or needs
all of them jointly. Single-feature and leave-one-out accuracy turns the
black-box number into an interpretable mechanistic claim for the paper.

Reuses load_instances/grouped_split/evaluate/SEEDS/TEST_FRACTION from
build_coord_baseline.py; only GBT is used here (matches the main baseline's
better-performing classifier, and avoids doubling runtime with logreg).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_coord_baseline import (  # noqa: E402
    SEEDS,
    TEST_FRACTION,
    evaluate,
    grouped_split,
    load_instances,
)

OUTPUT_DIR = Path(__file__).resolve().parent

ALL_FEATURES = ["x_center", "y_center", "width", "height", "area", "aspect_ratio"]
POSITION_FEATURES = ["x_center", "y_center"]
SHAPE_FEATURES = ["width", "height", "area", "aspect_ratio"]

SINGLE_FEATURE_SETS = {f: [f] for f in ALL_FEATURES}
GROUPED_SETS = {
    "position_only (x,y)": POSITION_FEATURES,
    "shape_only (w,h,area,aspect)": SHAPE_FEATURES,
    "all_six (main baseline)": ALL_FEATURES,
}
LEAVE_ONE_OUT_SETS = {
    f"all_minus_{f}": [c for c in ALL_FEATURES if c != f] for f in ALL_FEATURES
}


def run_feature_set(df, cols, label):
    accs = []
    for seed in SEEDS:
        train_df, test_df = grouped_split(df, seed, TEST_FRACTION)
        x_train = train_df[cols].to_numpy()
        y_train = train_df["class_id"].to_numpy()
        x_test = test_df[cols].to_numpy()
        y_test = test_df["class_id"].to_numpy()

        gbt = HistGradientBoostingClassifier(random_state=0).fit(x_train, y_train)
        y_pred = gbt.predict(x_test)
        metrics = evaluate(y_test, y_pred, y_train)
        accs.append(metrics["top1_acc"])

    accs = np.array(accs)
    mean = accs.mean()
    ci = accs.std(ddof=1) / np.sqrt(len(accs)) * 2.776 if len(accs) > 1 else 0.0
    return {"feature_set": label, "features": ",".join(cols), "top1_acc_mean": mean, "top1_acc_ci95": ci}


def main():
    df = load_instances()
    print(f"Loaded {len(df)} instances from {df['image_id'].nunique()} images")
    print("Classifier: gradient_boosted_tree (matches main baseline's stronger classifier)\n")

    rows = []

    print("=" * 70)
    print("SINGLE-FEATURE ACCURACY")
    print("=" * 70)
    for name, cols in SINGLE_FEATURE_SETS.items():
        r = run_feature_set(df, cols, f"single:{name}")
        rows.append(r)
        print(f"  {name:14s} acc={r['top1_acc_mean']:.4f} +/- {r['top1_acc_ci95']:.4f}")

    print("\n" + "=" * 70)
    print("GROUPED FEATURE SETS")
    print("=" * 70)
    for name, cols in GROUPED_SETS.items():
        r = run_feature_set(df, cols, name)
        rows.append(r)
        print(f"  {name:32s} acc={r['top1_acc_mean']:.4f} +/- {r['top1_acc_ci95']:.4f}")

    print("\n" + "=" * 70)
    print("LEAVE-ONE-OUT (all six minus one)")
    print("=" * 70)
    for name, cols in LEAVE_ONE_OUT_SETS.items():
        r = run_feature_set(df, cols, name)
        rows.append(r)
        print(f"  {name:20s} acc={r['top1_acc_mean']:.4f} +/- {r['top1_acc_ci95']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "feature_ablation.csv", index=False)

    # plot: single features + grouped sets, sorted by accuracy
    plot_rows = out[out["feature_set"].str.startswith("single:") | out["feature_set"].isin(GROUPED_SETS.keys())]
    plot_rows = plot_rows.sort_values("top1_acc_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [s.replace("single:", "") for s in plot_rows["feature_set"]]
    ax.barh(labels, plot_rows["top1_acc_mean"], xerr=plot_rows["top1_acc_ci95"], capsize=4)
    ax.set_xlabel("Top-1 accuracy (GBT, 5-seed mean)")
    ax.set_title("Feature ablation: which geometry features predict FDI class?")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_ablation.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved CSV and plot to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
