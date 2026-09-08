"""Leave-one-image-out sensitivity for the naive instance-weighted
dissociation-vs-canonical estimate (the -6.1/-6.3pp figure superseded by
dissociation_mixed_effects.py's crossed-model estimate, ~-3.6 to -3.9pp).

For each of the 70 dissociation images with at least one pooled test
instance across the 5 seeds, recompute the instance-weighted estimate with
that image's instances excluded, to check whether the naive estimate is
driven by a small number of outlier images (it is not, individually) or by
a small cluster of large, low-accuracy images considered together (it is -
see RESULTS.md Section 10 for the interpretation).
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "cpu_repro" / "coord_baseline"))
from build_coord_baseline import FEATURE_COLS, SEEDS, TEST_FRACTION, grouped_split  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "cpu_repro" / "coord_baseline" / "dentex"))
from build_coord_baseline_dentex import load_instances  # noqa: E402

SCAN_PATH = REPO_ROOT / "cpu_repro" / "anomaly_scan" / "dentex_results" / "full_scan_dentex.csv"
OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    df = load_instances()
    scan = pd.read_csv(SCAN_PATH)
    flagged_images = set(scan.loc[(scan["n_duplicate_codes"] > 0) | (scan["n_position_violations"] > 0), "image_id"])
    df["is_flagged_image"] = df["image_id"].isin(flagged_images)
    df["is_impacted"] = df["diagnosis"] == "Impacted"
    df["is_canonical"] = ~df["is_flagged_image"] & ~df["is_impacted"]
    teeth_count = df.groupby("image_id").size().rename("teeth_count")

    rows = []
    for seed in SEEDS:
        train_df, test_df = grouped_split(df, seed, TEST_FRACTION)
        x_train = train_df[FEATURE_COLS].to_numpy()
        y_train = train_df["class_id"].to_numpy()
        x_test = test_df[FEATURE_COLS].to_numpy()
        y_test = test_df["class_id"].to_numpy()
        clf = HistGradientBoostingClassifier(random_state=0)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        tdf = test_df.copy()
        tdf["correct"] = (y_test == y_pred).astype(float)
        tdf["seed"] = seed
        rows.append(tdf)
    pooled = pd.concat(rows, ignore_index=True)

    canon = pooled[pooled["is_canonical"]]
    dissoc = pooled[pooled["is_flagged_image"]]
    canon_acc = canon["correct"].mean()
    full_estimate = (dissoc["correct"].mean() - canon_acc) * 100
    print(f"Full instance-weighted estimate: {full_estimate:+.2f}pp "
          f"(canonical n={len(canon)}, dissociation n={len(dissoc)})")

    dissoc_images = sorted(dissoc["image_id"].unique())
    loio_rows = []
    for img in dissoc_images:
        sub = dissoc[dissoc["image_id"] != img]
        est = (sub["correct"].mean() - canon_acc) * 100
        loio_rows.append({"image_id": img, "n_excluded": int((dissoc["image_id"] == img).sum()),
                           "loio_estimate_pp": est, "shift_from_full_pp": est - full_estimate})
    loio_df = pd.DataFrame(loio_rows).sort_values("loio_estimate_pp")
    loio_df.to_csv(OUTPUT_DIR / "dissociation_loio_results.csv", index=False)

    print(f"LOIO range: [{loio_df['loio_estimate_pp'].min():+.2f}, {loio_df['loio_estimate_pp'].max():+.2f}]pp")
    print(f"images shifting estimate >1pp when excluded individually: "
          f"{(loio_df['shift_from_full_pp'].abs() > 1).sum()} / {len(loio_df)}")
    print(f"images shifting estimate >2pp when excluded individually: "
          f"{(loio_df['shift_from_full_pp'].abs() > 2).sum()} / {len(loio_df)}")

    per_image_acc = dissoc.groupby("image_id").agg(n_instances=("correct", "size"), acc=("correct", "mean")).reset_index()
    per_image_acc["teeth_count_full_image"] = per_image_acc["image_id"].map(teeth_count)
    per_image_acc = per_image_acc.sort_values("acc")
    per_image_acc.to_csv(OUTPUT_DIR / "dissociation_per_image_accuracy.csv", index=False)
    print("\nBottom 5 dissociation images by per-image accuracy:")
    print(per_image_acc.head(5).to_string(index=False))

    print("\nCumulative removal of worst-K images:")
    for k in [1, 2, 3, 5, 10, 15, 20]:
        worst = per_image_acc.head(k)["image_id"].tolist()
        sub = dissoc[~dissoc["image_id"].isin(worst)]
        est = (sub["correct"].mean() - canon_acc) * 100
        print(f"  drop worst {k:2d}: new estimate={est:+.2f}pp (n_dropped_instances={len(dissoc) - len(sub)})")

    print(f"\nSaved dissociation_loio_results.csv, dissociation_per_image_accuracy.csv to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
