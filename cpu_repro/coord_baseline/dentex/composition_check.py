"""Is the impacted-stratum accuracy rise (and the dissociation-stratum drop)
in stratified_analysis.py a real effect, or a tooth-type composition
artifact?

Motivating fact, checked here first: DENTEX's "Impacted" diagnosis applies
exclusively to third molars in this dataset (FDI 18/28/38/48) - the impacted
stratum is not a general anomaly sample, it is a third-molar sample. Since
third molars were already the easiest tooth-type group to place from
geometry alone (`../README.md`, tooth-type breakdown), any accuracy
comparison against the canonical remainder (which is only ~9% third molars)
is partly comparing different tooth-type mixes, not just "impacted vs. not."

This script controls for that in two ways, reusing the exact same trained
models and test-fold predictions as `stratified_analysis.py` (not a new
experiment - same 5 seeds, same GBT classifier, same splits):

1. Restrict all three strata to third-molar instances only (18/28/38/48) and
   compare top-1 accuracy on that matched subset.
2. Composition-standardize: reweight canonical's per-FDI-code accuracy by
   each stratum's own FDI-code mix, to get "canonical's expected accuracy if
   it had this stratum's tooth-type composition," and compare that to the
   stratum's actual accuracy - the gap that survives is the part composition
   does not explain.

Also reports the dissociation stratum with third molars excluded entirely,
since that stratum's mix is only mildly skewed (14.2% vs. canonical's 8.7%
third-molar share) and the question there is whether any of its drop
depends on that mild skew.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_coord_baseline import FEATURE_COLS, SEEDS, TEST_FRACTION, grouped_split  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_coord_baseline_dentex import load_instances  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SCAN_PATH = REPO_ROOT / "cpu_repro" / "anomaly_scan" / "dentex_results" / "full_scan_dentex.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

THIRD_MOLARS = {"18", "28", "38", "48"}


def main():
    df = load_instances()
    scan = pd.read_csv(SCAN_PATH)
    flagged_images = set(scan.loc[(scan["n_duplicate_codes"] > 0) | (scan["n_position_violations"] > 0), "image_id"])
    df["is_flagged_image"] = df["image_id"].isin(flagged_images)
    df["is_impacted"] = df["diagnosis"] == "Impacted"
    df["is_canonical"] = ~df["is_flagged_image"] & ~df["is_impacted"]

    print("=" * 100)
    print("STEP 1: FDI code / third-molar composition per stratum (full dataset, pre-split)")
    print("=" * 100)

    composition_rows = []
    for stratum_name, mask in [("canonical", df["is_canonical"]), ("dissociation", df["is_flagged_image"]), ("impacted", df["is_impacted"])]:
        sub = df[mask]
        n = len(sub)
        third_molar_n = int(sub["fdi"].isin(THIRD_MOLARS).sum())
        print(f"\n{stratum_name}: n={n}, third-molar share={third_molar_n}/{n} = {third_molar_n / n:.4f}")
        composition_rows.append({"stratum": stratum_name, "n": n, "third_molar_n": third_molar_n,
                                  "third_molar_share": third_molar_n / n})

    pd.DataFrame(composition_rows).to_csv(OUTPUT_DIR / "composition_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("STEP 2: composition-controlled comparison (gradient-boosted tree, pooled over 5 seeds)")
    print("=" * 100)

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
        tdf["correct"] = y_test == y_pred
        tdf["seed"] = seed
        rows.append(tdf)

    pooled = pd.concat(rows, ignore_index=True)
    mask_by_stratum = {
        "canonical": pooled["is_canonical"],
        "dissociation": pooled["is_flagged_image"],
        "impacted": pooled["is_impacted"],
    }

    print("\n--- third-molars-only (FDI 18/28/38/48), matched subset across strata ---")
    matched_rows = []
    for stratum_name, mask in mask_by_stratum.items():
        sub = pooled[mask & pooled["fdi"].isin(THIRD_MOLARS)]
        acc = sub["correct"].mean() if len(sub) else float("nan")
        print(f"  {stratum_name}: n={len(sub)}, third-molar-only top1_acc={acc:.4f}")
        matched_rows.append({"stratum": stratum_name, "n": len(sub), "third_molar_only_top1_acc": acc})
    pd.DataFrame(matched_rows).to_csv(OUTPUT_DIR / "composition_matched_subset.csv", index=False)

    print("\n--- dissociation, non-third-molars only (excludes the mild third-molar skew) ---")
    canonical_pool = pooled[pooled["is_canonical"]]
    canonical_acc = canonical_pool["correct"].mean()
    dissoc_non3m = pooled[mask_by_stratum["dissociation"] & ~pooled["fdi"].isin(THIRD_MOLARS)]
    canonical_non3m = canonical_pool[~canonical_pool["fdi"].isin(THIRD_MOLARS)]
    print(f"  canonical (non-3rd-molar): n={len(canonical_non3m)}, top1_acc={canonical_non3m['correct'].mean():.4f}")
    print(f"  dissociation (non-3rd-molar): n={len(dissoc_non3m)}, top1_acc={dissoc_non3m['correct'].mean():.4f}")

    print("\n--- composition-standardized: canonical's accuracy if it had the stratum's own FDI-code mix ---")
    canonical_per_code_acc = canonical_pool.groupby("fdi")["correct"].mean()
    standardized_rows = []
    for stratum_name in ["dissociation", "impacted"]:
        sub = pooled[mask_by_stratum[stratum_name]]
        actual_acc = sub["correct"].mean()
        code_weights = sub["fdi"].value_counts(normalize=True)
        common_codes = [c for c in code_weights.index if c in canonical_per_code_acc.index]
        coverage = code_weights[common_codes].sum()
        expected_canonical_acc = (code_weights[common_codes] * canonical_per_code_acc[common_codes]).sum() / coverage

        raw_gap = actual_acc - canonical_acc
        controlled_gap = actual_acc - expected_canonical_acc
        print(f"\n  {stratum_name}: actual={actual_acc:.4f} (n={len(sub)}), "
              f"canonical-with-this-mix={expected_canonical_acc:.4f} (code coverage={coverage:.3f})")
        print(f"  raw gap vs. canonical: {raw_gap:+.4f}   composition-controlled residual: {controlled_gap:+.4f}")
        standardized_rows.append({
            "stratum": stratum_name, "n": len(sub), "actual_top1_acc": actual_acc,
            "canonical_top1_acc": canonical_acc,
            "canonical_reweighted_to_stratum_mix": expected_canonical_acc,
            "raw_gap": raw_gap, "composition_controlled_residual": controlled_gap,
            "fdi_code_coverage_in_canonical": coverage,
        })
    pd.DataFrame(standardized_rows).to_csv(OUTPUT_DIR / "composition_standardized.csv", index=False)

    print(f"\nSaved composition_summary.csv, composition_matched_subset.csv, "
          f"composition_standardized.csv to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
