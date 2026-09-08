"""Pre-registered detectability check for Task 2 (error-pattern correlation),
run BEFORE spending GPU time training a real detector.

Question: given how few "wrong" predictions a real, reasonably-accurate
detector will actually produce on this test set, is there enough
statistical power to detect a meaningful difference between the real
detector's error-type distribution and the coordinate-only model's -
or would Task 2 be comparing noise to noise?

Comparison used (matches mitigation/README.md's evaluation protocol,
metric 3): neighbor-error-fraction (the dominant error type for the
coordinate-only model - 0.8079 logreg / 0.8287 GBT, RESULTS.md Section 2)
vs. whatever fraction the real detector's wrong predictions turn out to
be neighbor errors. This is treated as a one-sample proportion test
against p0 = the coordinate-only model's neighbor-error-fraction, since
that reference value is estimated on a much larger sample (n=5491, wrong
subset ~1800) than any plausible detector-error subsample, so its own
sampling error is comparatively negligible.

Detector accuracy is NOT YET KNOWN (no detector trained yet) - this
sweeps a plausible range (75-95%) rather than assuming a specific number,
since the point is to check whether the test is well-powered across
realistic scenarios, not to predict the actual result.
"""
import numpy as np
from scipy import stats

N_TEST = 5491  # seed-0 test size, RESULTS.md Section 2
P0 = 0.8287    # coordinate-only GBT neighbor-error-fraction, Section 2
ALPHA = 0.05   # two-sided
POWER = 0.80

ACCURACY_GRID = [0.75, 0.80, 0.85, 0.90, 0.95]

z_alpha = stats.norm.ppf(1 - ALPHA / 2)
z_beta = stats.norm.ppf(POWER)


def min_detectable_effect(n, p0, z_alpha, z_beta):
    """One-sample proportion test: smallest |p_hat - p0| detectable at
    the given n, alpha, power, using the normal approximation."""
    return (z_alpha + z_beta) * np.sqrt(p0 * (1 - p0) / n)


def main():
    print(f"Reference (coordinate-only GBT) neighbor-error-fraction, "
          f"P0 = {P0}")
    print(f"alpha = {ALPHA} (two-sided), power = {POWER}, "
          f"z_alpha={z_alpha:.4f}, z_beta={z_beta:.4f}\n")
    print(f"{'detector acc':>13} {'n_wrong':>8} {'min detectable |delta|':>24}")
    for acc in ACCURACY_GRID:
        n_wrong = int(round(N_TEST * (1 - acc)))
        mde = min_detectable_effect(n_wrong, P0, z_alpha, z_beta)
        print(f"{acc:>12.0%} {n_wrong:>8d} {mde:>23.4f}  "
              f"({mde*100:.1f} percentage points)")

    print("\nInterpretation: even under a conservative assumption of 95%")
    print("detector accuracy (only ~275 wrong predictions to work with),")
    print("Task 2 would still detect a difference in neighbor-error-")
    print("fraction as small as ~6 percentage points from the coordinate-")
    print("only model's 82.9%. A real detector relying on genuinely")
    print("different cues than geometry would be expected to produce a")
    print("much larger gap than that (its errors would plausibly look far")
    print("less neighbor-dominated), so Task 2 is well-powered across the")
    print("full plausible accuracy range - not an underpowered comparison")
    print("that risks reading noise as a null result.")
    print("\nCaveat: this assumes a normal approximation to the binomial")
    print("and treats P0 as a fixed population value (reasonable given")
    print("its much larger source sample). It does not model any")
    print("clustering by image (multiple teeth per image are not")
    print("independent draws) - a more careful analysis would use a")
    print("clustered/robust variance estimate once real detector output")
    print("exists, the same way Section 10's image+seed random-effects")
    print("model corrected an overstated paired-seed estimate there.")


if __name__ == "__main__":
    main()
