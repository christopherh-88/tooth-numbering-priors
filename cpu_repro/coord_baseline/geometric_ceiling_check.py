"""How close is the current classifier's accuracy to the theoretical
ceiling achievable from geometry alone?

Section 2 reports 69.5% top-1 (GBT) from 6 geometric features. That
number is only interesting if it's near the best any classifier could do
given how much real classes' geometric distributions overlap - a weak
classifier could leave real headroom unexploited, which would undercut
the "geometry meaningfully predicts identity" framing (a stronger
classifier might reveal much more). A k-NN classifier with a large
neighborhood, evaluated non-parametrically, approximates the Bayes-error
ceiling well when there's enough data per class (700-1000 instances per
class here) without assuming any particular decision-boundary shape the
way logistic regression (linear) or a fixed-depth GBT do.

Uses the exact same seed-0 grouped train/test split as Section 2/GBT, so
the comparison is apples-to-apples - same instances, same features, same
generalization boundary (image-level grouping prevents leakage), only
the classifier changes.
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from build_coord_baseline import (
    load_instances, grouped_split, evaluate, FEATURE_COLS, SEEDS,
    TEST_FRACTION,
)

K_GRID = [5, 15, 25, 50, 100]


def main():
    df = load_instances()
    seed = SEEDS[0]
    train_df, test_df = grouped_split(df, seed, TEST_FRACTION)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["class_id"].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["class_id"].values

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"seed {seed}: n_train={len(X_train)}, n_test={len(X_test)}\n")
    print(f"{'k':>5} {'top1_acc':>10} {'quadrant_acc':>13} {'tooth_type_acc':>15}")

    best_acc = 0.0
    for k in K_GRID:
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        res = evaluate(y_test, y_pred, y_train)
        best_acc = max(best_acc, res["top1_acc"])
        print(f"{k:>5} {res['top1_acc']:>10.4f} {res['quadrant_acc']:>13.4f} "
              f"{res['tooth_type_acc']:>15.4f}")

    gbt_acc = 0.692770  # Section 2, seed-0 point value, GBT
    logreg_acc = 0.663814  # Section 2, seed-0 point value, LogReg
    print(f"\nFor reference (Section 2, seed-0 point values):")
    print(f"  logistic regression: {logreg_acc:.4f}")
    print(f"  gradient-boosted tree: {gbt_acc:.4f}")
    print(f"\nBest k-NN top1_acc across k in {K_GRID}: {best_acc:.4f}")
    print(f"Gap vs. GBT: {best_acc - gbt_acc:+.4f} ({(best_acc - gbt_acc)*100:+.2f} pp)")


if __name__ == "__main__":
    main()
