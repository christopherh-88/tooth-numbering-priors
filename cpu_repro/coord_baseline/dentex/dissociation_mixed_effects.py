"""Mixed-effects model on the dissociation stratum, full instance-level data
(not the 5-seed mean used in stratified_analysis.py).

Reuses the exact same trained models / pooled test-fold predictions as
stratified_analysis.py and composition_check.py (5 seeds, GBT, same
splits). Builds one row per (instance, seed-it-was-tested-in) with a
binary `correct` outcome, restricted to canonical + dissociation instances
(impacted excluded, matching the existing dissociation-vs-canonical
comparison), and fits two linear probability mixed-effects models:

  1. correct ~ is_dissociation + (1 | image_id) + (1 | seed)   [primary]
  2. correct ~ is_dissociation + log(teeth_count) + (1 | image_id) + (1 | seed)

A linear (not logistic) mixed model is used deliberately: its fixed-effect
coefficient on is_dissociation is directly a percentage-point difference
in accuracy, the same unit already used elsewhere in this repo, so the
numbers are directly comparable without a log-odds conversion. image_id
and seed are both modeled as random effects (crossed, via a variance
component for seed) because either source of non-independence - multiple
teeth in the same image, and the same image re-tested by differently
trained models across seeds - would make a naive pooled-instance CI
anti-conservative if left unmodeled. Model 2 tests whether teeth-in-image
("crowding") is a confound for the stratum effect; see RESULTS.md Section
10 for why it was ruled out (not significant, wrong direction, closes
~12% of the gap to the naive pooled estimate).

See dissociation_loio.py for why the primary estimate here (~-3.6 to
-3.9pp) is smaller than the naive pooled/paired-seed estimate (~-6.1 to
-6.3pp).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingClassifier

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "cpu_repro" / "coord_baseline"))
from build_coord_baseline import FEATURE_COLS, SEEDS, TEST_FRACTION, grouped_split  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "cpu_repro" / "coord_baseline" / "dentex"))
from build_coord_baseline_dentex import load_instances  # noqa: E402

SCAN_PATH = REPO_ROOT / "cpu_repro" / "anomaly_scan" / "dentex_results" / "full_scan_dentex.csv"
OUTPUT_DIR = Path(__file__).resolve().parent


def build_model_df():
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
    model_df = pooled[pooled["is_canonical"] | pooled["is_flagged_image"]].copy()
    model_df["is_dissociation"] = model_df["is_flagged_image"].astype(int)
    model_df = model_df.join(teeth_count, on="image_id")
    model_df["log_teeth_count"] = np.log(model_df["teeth_count"])
    return model_df


def fit_crossed(model_df, formula):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm(formula, model_df, groups=model_df["image_id"], vc_formula={"seed": "0 + C(seed)"})
        return md.fit(reml=True)


def main():
    model_df = build_model_df()
    print(f"model_df: n={len(model_df)} "
          f"(canonical={int((model_df['is_dissociation'] == 0).sum())}, "
          f"dissociation={int(model_df['is_dissociation'].sum())}), "
          f"unique images={model_df['image_id'].nunique()}")

    print("\n" + "=" * 90)
    print("MODEL 1 (primary): correct ~ is_dissociation + (1 | image_id) + (1 | seed)")
    print("=" * 90)
    mdf1 = fit_crossed(model_df, "correct ~ is_dissociation")
    print(mdf1.summary())
    c1, p1 = mdf1.params["is_dissociation"], mdf1.pvalues["is_dissociation"]
    ci1 = mdf1.conf_int().loc["is_dissociation"]

    print("\n" + "=" * 90)
    print("MODEL 2 (+ log_teeth_count covariate)")
    print("=" * 90)
    mdf2 = fit_crossed(model_df, "correct ~ is_dissociation + log_teeth_count")
    print(mdf2.summary())
    c2, p2 = mdf2.params["is_dissociation"], mdf2.pvalues["is_dissociation"]
    ci2 = mdf2.conf_int().loc["is_dissociation"]
    ct, pt = mdf2.params["log_teeth_count"], mdf2.pvalues["log_teeth_count"]
    cit = mdf2.conf_int().loc["log_teeth_count"]

    naive_canon = model_df.loc[model_df["is_dissociation"] == 0, "correct"].mean()
    naive_dissoc = model_df.loc[model_df["is_dissociation"] == 1, "correct"].mean()
    naive_diff = (naive_dissoc - naive_canon) * 100

    summary = pd.DataFrame([
        {"model": "naive_pooled_instances", "estimate_pp": naive_diff, "ci_low_pp": None, "ci_high_pp": None, "p_value": None},
        {"model": "crossed_image_seed_no_covariate", "estimate_pp": c1 * 100, "ci_low_pp": ci1[0] * 100, "ci_high_pp": ci1[1] * 100, "p_value": p1},
        {"model": "crossed_image_seed_plus_log_teeth_count", "estimate_pp": c2 * 100, "ci_low_pp": ci2[0] * 100, "ci_high_pp": ci2[1] * 100, "p_value": p2},
        {"model": "log_teeth_count_coefficient", "estimate_pp": ct * 100, "ci_low_pp": cit[0] * 100, "ci_high_pp": cit[1] * 100, "p_value": pt},
    ])
    summary.to_csv(OUTPUT_DIR / "dissociation_mixed_effects_summary.csv", index=False)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(summary.to_string(index=False))
    gap_before = naive_diff - c1 * 100
    gap_after = naive_diff - c2 * 100
    print(f"\nGap closed by log_teeth_count covariate: "
          f"{(1 - abs(gap_after) / abs(gap_before)) * 100:.1f}%")
    print(f"\nSaved dissociation_mixed_effects_summary.csv to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
