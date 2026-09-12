"""Resume-note item 1: does excluding the images flagged inconsistent by
Task 4's programmatic Arch/Site check (verify_expanded.py) materially move
Section 25's headline top1/quadrant accuracy numbers (0.2535/0.4467, per
bootstrap_ci_section25.csv)?

Reuses load_instances()/grouped_split() unchanged (same seed-0 GBT fit as
Section 25 and bootstrap_ci.py). Re-derives the flagged image_id set by
running the same Arch/Site check as verify_expanded.py, then reports
accuracy on the full seed-0 test set vs. the same test set with flagged
images dropped.
"""

import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_coord_baseline_denpar import load_instances, grouped_split, FEATURE_COLS, RAW_DIR

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "coord_baseline"))
from build_coord_baseline import quadrant_of

SEED = 0


def flagged_image_ids():
    wb = openpyxl.load_workbook(RAW_DIR / "Characteristics.xlsx")
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))[1:]

    flagged = set()
    for r in rows:
        if r[0] is None:
            continue
        image_id, arch, site, code_str = r[0], r[1], r[2], r[3]
        try:
            codes = [str(int(float(c.strip()))) for c in str(code_str).split(",") if c.strip()]
        except Exception:
            continue
        if not codes:
            continue
        quadrants = set(c[0] for c in codes if len(c) == 2)
        if not quadrants:
            continue

        expected_arch_quads = {"1", "2", "5", "6"} if arch == "Upper" else \
                               {"3", "4", "7", "8"} if arch == "Lower" else None
        arch_fail = expected_arch_quads is not None and not quadrants.issubset(expected_arch_quads)

        site_fail = False
        if site in ("Right", "Left"):
            expected_site_quads = {"1", "4", "5", "8"} if site == "Right" else {"2", "3", "6", "7"}
            site_fail = not quadrants.issubset(expected_site_quads)

        if arch_fail or site_fail:
            flagged.add(str(int(image_id)))
    return flagged


def top1_metric(true, pred):
    return float(np.mean(true == pred))


def quadrant_metric(true, pred):
    return float(np.mean(quadrant_of(true) == quadrant_of(pred)))


def main():
    df = load_instances()
    train_df, test_df = grouped_split(df, SEED, 0.2)
    x_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df["class_id"].to_numpy()
    x_test = test_df[FEATURE_COLS].to_numpy()
    y_test = test_df["class_id"].to_numpy()

    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    result = test_df[["image_id", "class_id"]].copy()
    result["pred"] = y_pred
    result = result.rename(columns={"class_id": "true_class"})

    flagged = flagged_image_ids()
    n_test_images = result["image_id"].nunique()
    n_flagged_in_test = result[result["image_id"].isin(flagged)]["image_id"].nunique()

    full_top1 = top1_metric(result["true_class"].to_numpy(), result["pred"].to_numpy())
    full_quad = quadrant_metric(result["true_class"].to_numpy(), result["pred"].to_numpy())

    clean = result[~result["image_id"].isin(flagged)]
    clean_top1 = top1_metric(clean["true_class"].to_numpy(), clean["pred"].to_numpy())
    clean_quad = quadrant_metric(clean["true_class"].to_numpy(), clean["pred"].to_numpy())

    print("=" * 70)
    print("Resume-item 1: sensitivity of Section 25 headline numbers to")
    print("Task 4's flagged Arch/Site-inconsistent DenPAR rows")
    print("=" * 70)
    print(f"Full seed-0 test set: {len(result)} instances, {n_test_images} images")
    print(f"  Flagged (Arch or Site inconsistent) images present in this test set: {n_flagged_in_test}")
    print(f"  top1_acc      full={full_top1:.4f}   excl.flagged={clean_top1:.4f}   "
          f"delta={clean_top1 - full_top1:+.4f}")
    print(f"  quadrant_acc  full={full_quad:.4f}   excl.flagged={clean_quad:.4f}   "
          f"delta={clean_quad - full_quad:+.4f}")
    print(f"  (excl.flagged set: {len(clean)} instances, {clean['image_id'].nunique()} images)")

    pd.DataFrame([
        {"variant": "full_test_set", "n_instances": len(result), "n_images": n_test_images,
         "top1_acc": full_top1, "quadrant_acc": full_quad},
        {"variant": "excl_flagged", "n_instances": len(clean), "n_images": clean["image_id"].nunique(),
         "top1_acc": clean_top1, "quadrant_acc": clean_quad},
    ]).to_csv(Path(__file__).resolve().parent / "sensitivity_check_section25.csv", index=False)
    print(f"\nSaved sensitivity_check_section25.csv")


if __name__ == "__main__":
    main()
