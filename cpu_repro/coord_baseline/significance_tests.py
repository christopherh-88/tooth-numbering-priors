"""Paired significance tests for the coordinate-only baseline results.

STATUS: run 2026-09-08, results in RESULTS.md Section 12. Consumes only
files that already exist in this directory - no new data collection, no
retraining. Run with: python3 significance_tests.py

Two tests, both paired on seed (same 5 seeds, same image-level splits,
so pairing removes seed-to-seed split variance rather than treating the
two conditions as independent samples):

1. real vs. shuffled-per-image control (per_seed_results.csv here vs.
   controls/per_seed_results.csv), per classifier - is the drop from the
   real condition to the shuffled negative control significant, seed by
   seed, rather than just "the CIs don't overlap"?
2. top1_acc vs. majority_baseline_acc, per classifier - same idea,
   directly on the headline claim.

Uses a paired permutation test (sign-flip on the seed-level differences,
10000 resamples, fixed seed for reproducibility) rather than a paired
t-test, since n=5 seeds is too small to trust normality assumptions.
"""
import itertools
import numpy as np
import pandas as pd

RNG_SEED = 0
N_RESAMPLES = 10000

MAIN_CSV = "per_seed_results.csv"
CONTROLS_CSV = "controls/per_seed_results.csv"


def paired_permutation_test(a, b, n_resamples=N_RESAMPLES, rng_seed=RNG_SEED):
    """Two-sided paired permutation (sign-flip) test on a - b.

    a, b: equal-length 1D arrays, paired by index (here: by seed).
    Returns (observed_mean_diff, p_value).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape
    diffs = a - b
    observed = diffs.mean()
    n = len(diffs)
    rng = np.random.default_rng(rng_seed)
    # exact enumeration if small enough (2**n <= n_resamples), else Monte Carlo
    if 2 ** n <= n_resamples:
        count = 0
        total = 0
        for signs in itertools.product([1, -1], repeat=n):
            resampled_mean = (diffs * np.array(signs)).mean()
            if abs(resampled_mean) >= abs(observed) - 1e-12:
                count += 1
            total += 1
        p = count / total
    else:
        signs = rng.choice([1, -1], size=(n_resamples, n))
        resampled_means = (diffs[None, :] * signs).mean(axis=1)
        p = (np.abs(resampled_means) >= abs(observed) - 1e-12).mean()
    return observed, p


def main():
    main_df = pd.read_csv(MAIN_CSV)
    controls_df = pd.read_csv(CONTROLS_CSV)

    print("=== Test 1: real vs. shuffled-per-image control, per classifier ===")
    for clf in main_df["classifier"].unique():
        real = main_df[main_df["classifier"] == clf].sort_values("seed")
        shuf = controls_df[
            (controls_df["classifier"] == clf)
            & (controls_df["condition"] == "full (shuffled per image)")
        ].sort_values("seed")
        assert list(real["seed"]) == list(shuf["seed"]), "seed alignment mismatch"
        diff, p = paired_permutation_test(
            real["top1_acc"].values, shuf["top1_acc"].values
        )
        print(f"  {clf}: mean top1_acc diff (real - shuffled) = {diff:.4f}, "
              f"paired permutation p = {p:.5f}")

    print("\n=== Test 2: top1_acc vs. majority_baseline_acc, per classifier ===")
    for clf in main_df["classifier"].unique():
        sub = main_df[main_df["classifier"] == clf].sort_values("seed")
        diff, p = paired_permutation_test(
            sub["top1_acc"].values, sub["majority_baseline_acc"].values
        )
        print(f"  {clf}: mean diff (top1_acc - majority_baseline) = {diff:.4f}, "
              f"paired permutation p = {p:.5f}")

    print("\ndone")


if __name__ == "__main__":
    main()
