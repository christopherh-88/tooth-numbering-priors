"""Do YOLO and the coordinate-only baseline fail on the *same* samples,
not just at similar rates? Sections 21/22 compare error rates and error
taxonomy but never joined the two models' predictions into one per-sample
table. Exploratory/diagnostic - no GO/NO-GO rule applies here.

Reuses case_study_yolo_vs_coord.py's build_coord_predictions() and
run_yolo_predictions() unchanged (same deterministic seed-0 refit + CPU
inference on the existing best.pt checkpoint - no new training or GPU run),
but does NOT filter down to one quadrant of the 2x2 table the way the case
study did - this script keeps the full joined set.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from case_study_yolo_vs_coord import build_coord_predictions, run_yolo_predictions  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coord_baseline"))
from build_coord_baseline import (  # noqa: E402
    is_mirror_quadrant_error,
    is_neighbor_error,
)

OUT_DIR = Path(__file__).resolve().parent / "eval_results"


def phi_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation applied to two binary (0/1) vectors - the
    standard association measure for a 2x2 table of binary outcomes.
    Chosen over Cohen's kappa: kappa is built for inter-rater agreement
    on nominal categories and corrects for chance *agreement* (which
    would count "both models happened to be right" the same way as
    "both models happened to be wrong"); phi is the direct correlation
    between the two correctness indicators, which is what "are these two
    models' errors correlated" actually asks, and it is numerically
    identical to the Matthews correlation coefficient for a 2x2 table."""
    a = a.astype(float)
    b = b.astype(float)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    print("Refitting coordinate-only baseline (seed 0, GBT) with instance identity...")
    coord_df = build_coord_predictions()
    print("Running YOLOv8 (best.pt) on the val split for per-instance predictions...")
    yolo_df = run_yolo_predictions()

    merged = coord_df.merge(yolo_df, on=["label_file", "line_idx"], how="inner")
    n_total = len(merged)
    print(f"Joined on (label_file, line_idx): {n_total} instances.")

    # Undetected ground-truth boxes (yolo_pred is NaN - YOLO never found this
    # tooth at all) count as YOLO-wrong, not dropped: the coordinate-only
    # baseline is always given a box to classify, so treating a missed
    # detection as anything other than "YOLO failed on this instance" would
    # understate YOLO's real miss rate on this joined set.
    merged["yolo_correct"] = (
        merged["yolo_pred"].notna()
        & (merged["yolo_pred"].astype("Int64") == merged["true_class"])
    )
    merged["coord_correct"] = merged["coord_pred"] == merged["true_class"]

    both_correct = merged[merged["yolo_correct"] & merged["coord_correct"]]
    both_wrong = merged[~merged["yolo_correct"] & ~merged["coord_correct"]]
    yolo_only = merged[merged["yolo_correct"] & ~merged["coord_correct"]]
    coord_only = merged[~merged["yolo_correct"] & merged["coord_correct"]]

    print("\n2x2 breakdown:")
    for name, subset in [("both_correct", both_correct), ("both_wrong", both_wrong),
                          ("yolo_only_correct", yolo_only), ("coord_only_correct", coord_only)]:
        print(f"  {name:22s} {len(subset):5d}  ({100 * len(subset) / n_total:.1f}%)")

    p_yolo_wrong = 1 - merged["yolo_correct"].mean()
    p_coord_wrong = 1 - merged["coord_correct"].mean()
    expected_both_wrong_frac = p_yolo_wrong * p_coord_wrong
    expected_both_wrong_n = expected_both_wrong_frac * n_total
    observed_both_wrong_n = len(both_wrong)

    print(f"\nP(YOLO wrong) = {p_yolo_wrong:.4f}, P(coord wrong) = {p_coord_wrong:.4f}")
    print(f"Expected P(both wrong) under independence = {expected_both_wrong_frac:.4f} "
          f"({expected_both_wrong_n:.1f} instances)")
    print(f"Observed both-wrong = {observed_both_wrong_n} instances "
          f"({100 * observed_both_wrong_n / n_total:.2f}%)")
    print(f"Observed / expected ratio = {observed_both_wrong_n / expected_both_wrong_n:.2f}x")

    phi = phi_coefficient(
        (~merged["yolo_correct"].to_numpy(dtype=bool)).astype(int),
        (~merged["coord_correct"].to_numpy(dtype=bool)).astype(int),
    )
    print(f"\nPhi coefficient (correlation between the two wrong/right indicators) = {phi:.4f}")

    # Error taxonomy on the both-wrong subset, for both models' own predictions.
    def taxonomy_breakdown(df, pred_col):
        types = [
            "mirror_quadrant" if is_mirror_quadrant_error(t, p)
            else "neighbor" if is_neighbor_error(t, p)
            else "other"
            for t, p in zip(df["true_class"], df[pred_col].astype(int))
        ]
        return pd.Series(types).value_counts()

    print("\nBoth-wrong subset - coordinate model's own error-type breakdown:")
    coord_tax = taxonomy_breakdown(both_wrong, "coord_pred")
    print(coord_tax.to_string())

    print("\nBoth-wrong subset - YOLO's own error-type breakdown:")
    yolo_tax = taxonomy_breakdown(both_wrong[both_wrong["yolo_pred"].notna()], "yolo_pred")
    print(yolo_tax.to_string())

    merged.to_csv(OUT_DIR / "error_correlation_joined.csv", index=False)
    summary = {
        "n_total": n_total,
        "n_both_correct": len(both_correct),
        "n_both_wrong": len(both_wrong),
        "n_yolo_only_correct": len(yolo_only),
        "n_coord_only_correct": len(coord_only),
        "p_yolo_wrong": p_yolo_wrong,
        "p_coord_wrong": p_coord_wrong,
        "expected_both_wrong_frac": expected_both_wrong_frac,
        "expected_both_wrong_n": expected_both_wrong_n,
        "observed_both_wrong_n": observed_both_wrong_n,
        "observed_over_expected_ratio": observed_both_wrong_n / expected_both_wrong_n,
        "phi_coefficient": phi,
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "error_correlation_summary.csv", index=False)
    print(f"\nSaved error_correlation_joined.csv and error_correlation_summary.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
