"""Full 32-class correctness proof for the MIRROR_MAP fix applied in
notebooks/yolov8+unet/yolov8+unet_training.ipynb cell 21.

The bug-fix verification run on 2026-09-08 (mitigation/README.md) only
spot-checked one pair (FDI 11 <-> 21). This proves the property holds
for all 32 classes, not just that one example, before it's relied on in
Friday's GPU training run.

Uses the exact same construction as the notebook fix (copied verbatim,
not re-derived differently here - a divergent re-implementation would
defeat the point of a verification script).
"""
import numpy as np

FDI_CODES = [
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
]
MIRROR_QUADRANTS = {("1", "2"), ("2", "1"), ("3", "4"), ("4", "3")}
MIRROR_MAP = [
    next(j for j, c2 in enumerate(FDI_CODES)
         if c2[1] == c[1] and (c[0], c2[0]) in MIRROR_QUADRANTS)
    for c in FDI_CODES
]


def main():
    print(f"{'i':>3} {'FDI':>5} -> {'MIRROR_MAP[i]':>13} {'FDI':>5}")
    for i, code in enumerate(FDI_CODES):
        j = MIRROR_MAP[i]
        print(f"{i:>3} {code:>5} -> {j:>13} {FDI_CODES[j]:>5}")

    checks = {}

    # 1. length and range
    checks["length == 32"] = len(MIRROR_MAP) == 32
    checks["all indices in [0, 31]"] = all(0 <= j < 32 for j in MIRROR_MAP)

    # 2. involution: applying twice returns the identity
    checks["is an involution (MIRROR_MAP[MIRROR_MAP[i]] == i for all i)"] = all(
        MIRROR_MAP[MIRROR_MAP[i]] == i for i in range(32)
    )

    # 3. no fixed points - every class has a DIFFERENT mirror partner
    checks["no fixed points (MIRROR_MAP[i] != i for all i)"] = all(
        MIRROR_MAP[i] != i for i in range(32)
    )

    # 4. every mapped pair preserves tooth-type digit and swaps quadrant
    #    via a valid MIRROR_QUADRANTS relation
    def pair_is_valid(i):
        j = MIRROR_MAP[i]
        ci, cj = FDI_CODES[i], FDI_CODES[j]
        return ci[1] == cj[1] and (ci[0], cj[0]) in MIRROR_QUADRANTS

    checks["every pair preserves tooth-type and swaps to a valid mirror quadrant"] = all(
        pair_is_valid(i) for i in range(32)
    )

    # 5. it's a genuine permutation (bijective - no two classes map to the
    #    same target, which would silently drop a class's box on flip)
    checks["is a bijection (32 distinct targets, none dropped)"] = len(set(MIRROR_MAP)) == 32

    # 6. the four expected quadrant blocks map to each other exactly
    #    (1<->2, 3<->4), not some other mixture
    q1 = [i for i, c in enumerate(FDI_CODES) if c[0] == "1"]
    q2 = [i for i, c in enumerate(FDI_CODES) if c[0] == "2"]
    q3 = [i for i, c in enumerate(FDI_CODES) if c[0] == "3"]
    q4 = [i for i, c in enumerate(FDI_CODES) if c[0] == "4"]
    checks["quadrant 1 maps entirely into quadrant 2"] = all(MIRROR_MAP[i] in q2 for i in q1)
    checks["quadrant 2 maps entirely into quadrant 1"] = all(MIRROR_MAP[i] in q1 for i in q2)
    checks["quadrant 3 maps entirely into quadrant 4"] = all(MIRROR_MAP[i] in q4 for i in q3)
    checks["quadrant 4 maps entirely into quadrant 3"] = all(MIRROR_MAP[i] in q3 for i in q4)

    print("\n=== Correctness checks ===")
    all_passed = True
    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
        all_passed = all_passed and result

    print(f"\n{'ALL CHECKS PASSED' if all_passed else 'AT LEAST ONE CHECK FAILED'} "
          f"- {'MIRROR_MAP is a correct, full 32-class mirror-quadrant involution' if all_passed else 'DO NOT rely on this fix as-is'}")


if __name__ == "__main__":
    main()
