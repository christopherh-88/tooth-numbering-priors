"""Does the coordinate-only model make more errors on standard-FDI teeth
in images that ALSO contain a supernumerary tooth, vs. images that don't?

This is the follow-up flagged in RESULTS.md Section 16 / paper/DRAFT.md
Section 5's clinical-stakes paragraph: Section 16 established that 24
real supernumerary instances exist across 23 images (existence-proof),
but never measured whether the coordinate-only model actually performs
worse near them. This script closes that gap - or reports honestly that
it doesn't, whichever the data shows.

Design: train the coordinate-only GBT on ALL of UFBA-425 (no held-out
split needed - evaluation happens entirely on the separate Dual-Labeled
Dataset, so there's no leakage concern the way there would be evaluating
on UFBA-425 itself). Evaluate on the Dual-Labeled Dataset's own
standard-FDI-labeled teeth (excluding the "91" instances themselves,
which have no valid target in the 32-class scheme), split into:
  - "supernumerary-present" group: teeth in the 23 images that also
    contain >=1 label-91 instance (RESULTS.md Section 16).
  - "control" group: teeth in the other paired-image label files.
Compares per-instance accuracy and per-image mean accuracy between the
two groups, with the same coordinate-convention check protocol used
before combining any two datasets elsewhere in this project
(cpu_repro/CONVENTIONS.md) run first, since a convention mismatch would
corrupt this comparison entirely (large "errors" that are just a mirrored
coordinate system, not a real effect).

Requires the Dual-Labeled Dataset downloaded locally (see
inspect_labels.py's docstring) - path below must be set to the extracted
labels/ and images1/ directories, not redistributed in this repo.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coord_baseline"))
from build_coord_baseline import (
    load_instances, FEATURE_COLS, FDI_CODES, CODE_TO_IDX,
    quadrant_of, tooth_type_of, is_mirror_quadrant_error, is_neighbor_error,
)

LABELS_DIR = "extracted/labels"
IMAGES_DIR = "extracted/images1"

# Section 16's 23 usable supernumerary-present image stems (re-derived
# below from the label files directly, not hardcoded from memory - this
# list is printed and should match RESULTS.md Section 16's count of 23).


def load_dual_labeled_instances():
    image_ids = set(os.path.splitext(f)[0] for f in os.listdir(IMAGES_DIR))
    rows = []
    supernumerary_stems = set()
    supernumerary_centroids = {}  # stem -> list of (x_center, y_center), normalized
    for lf in sorted(os.listdir(LABELS_DIR)):
        stem = os.path.splitext(lf)[0]
        if stem not in image_ids:
            continue  # only the 500 paired label+image files, per Section 16
        data = json.load(open(os.path.join(LABELS_DIR, lf)))
        img_w, img_h = data["imageWidth"], data["imageHeight"]
        for shape in data["shapes"]:
            label = str(shape["label"])
            pts = np.array(shape["points"], dtype=float)
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            x_center = (x1 + x2) / 2 / img_w
            y_center = (y1 + y2) / 2 / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            if label == "91":
                supernumerary_stems.add(stem)
                supernumerary_centroids.setdefault(stem, []).append((x_center, y_center))
                continue
            if label not in CODE_TO_IDX:
                continue  # skip any non-FDI label defensively
            rows.append({
                "image_id": stem,
                "class_id": CODE_TO_IDX[label],
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            })
    df = pd.DataFrame(rows)
    df["area"] = df["width"] * df["height"]
    df["aspect_ratio"] = df["width"] / df["height"].replace(0, np.nan)
    df["aspect_ratio"] = df["aspect_ratio"].fillna(df["aspect_ratio"].median())
    return df, supernumerary_stems, supernumerary_centroids


def convention_check(df, label):
    """Same protocol as cpu_repro/CONVENTIONS.md: quadrant-1 classes
    should have a lower mean x_center than quadrant-2 classes if the
    convention matches UFBA-425 (quadrants 1,4 -> image-left)."""
    q1_ids = [CODE_TO_IDX[c] for c in FDI_CODES if c[0] == "1"]
    q2_ids = [CODE_TO_IDX[c] for c in FDI_CODES if c[0] == "2"]
    q1_x = df[df["class_id"].isin(q1_ids)]["x_center"].mean()
    q2_x = df[df["class_id"].isin(q2_ids)]["x_center"].mean()
    print(f"[{label}] mean x_center: quadrant 1 = {q1_x:.4f}, "
          f"quadrant 2 = {q2_x:.4f} -> "
          f"{'MATCHES UFBA-425 convention (1 < 2)' if q1_x < q2_x else 'MISMATCH - DO NOT PROCEED'}")
    return q1_x < q2_x


def main():
    print("=== Step 1: coordinate-convention check (must pass before proceeding) ===")
    ufba_df = load_instances()
    ufba_ok = convention_check(ufba_df, "UFBA-425 (reference)")

    dual_df, supernumerary_stems, supernumerary_centroids = load_dual_labeled_instances()
    print(f"\nLoaded {len(dual_df)} standard-FDI instances across "
          f"{dual_df['image_id'].nunique()} images from the Dual-Labeled Dataset "
          f"(paired-image tranche).")
    print(f"Supernumerary-present images found: {len(supernumerary_stems)} "
          f"(should be 23, per RESULTS.md Section 16)")
    dual_ok = convention_check(dual_df, "Dual-Labeled Dataset")

    if not (ufba_ok and dual_ok):
        print("\nABORTING: coordinate convention mismatch detected - "
              "proceeding would produce a meaningless comparison.")
        return

    print("\n=== Step 2: train GBT on all of UFBA-425, evaluate on Dual-Labeled Dataset ===")
    X_train = ufba_df[FEATURE_COLS].values
    y_train = ufba_df["class_id"].values
    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(X_train, y_train)

    X_eval = dual_df[FEATURE_COLS].values
    y_true = dual_df["class_id"].values
    y_pred = clf.predict(X_eval)
    dual_df["correct"] = (y_true == y_pred)
    dual_df["supernumerary_present"] = dual_df["image_id"].isin(supernumerary_stems)

    print("\n=== Step 3: accuracy comparison ===")
    grp = dual_df.groupby("supernumerary_present")["correct"].agg(["mean", "count"])
    print(grp)

    sn = dual_df[dual_df["supernumerary_present"]]
    ctrl = dual_df[~dual_df["supernumerary_present"]]

    # per-instance two-proportion z-test
    p1, n1 = sn["correct"].mean(), len(sn)
    p2, n2 = ctrl["correct"].mean(), len(ctrl)
    p_pool = (sn["correct"].sum() + ctrl["correct"].sum()) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    print(f"\nPer-instance: supernumerary-present acc = {p1:.4f} (n={n1}), "
          f"control acc = {p2:.4f} (n={n2})")
    print(f"Two-proportion z-test: z = {z:.3f}, p = {p_value:.4f}")

    # per-image mean accuracy comparison (accounts for clustering better
    # than the per-instance test above, which treats every tooth as an
    # independent draw - it isn't, teeth in the same image share an
    # image-level accuracy)
    per_image = dual_df.groupby(["image_id", "supernumerary_present"])["correct"].mean().reset_index()
    sn_img = per_image[per_image["supernumerary_present"]]["correct"]
    ctrl_img = per_image[~per_image["supernumerary_present"]]["correct"]
    u_stat, u_p = stats.mannwhitneyu(sn_img, ctrl_img, alternative="two-sided")
    print(f"\nPer-image mean accuracy: supernumerary-present images "
          f"mean={sn_img.mean():.4f} (n={len(sn_img)} images), "
          f"control images mean={ctrl_img.mean():.4f} (n={len(ctrl_img)} images)")
    print(f"Mann-Whitney U test (image-level, accounts for clustering): "
          f"U = {u_stat:.1f}, p = {u_p:.4f}")

    print("\n=== Step 4: localized adjacency test ===")
    print("Whole-image averaging dilutes the signal with teeth far from the")
    print("supernumerary tooth that plausibly aren't displaced at all. For")
    print("each supernumerary instance, take its K=3 nearest standard teeth")
    print("(by centroid distance, same image) as the 'near' set; everything")
    print("else in the 23 supernumerary-present images is 'far (same-image)'.")
    K_NEAREST = 3
    near_ids = set()
    for stem, centroids in supernumerary_centroids.items():
        img_teeth = dual_df[dual_df["image_id"] == stem]
        if len(img_teeth) == 0:
            continue
        for cx, cy in centroids:
            dists = np.hypot(img_teeth["x_center"] - cx, img_teeth["y_center"] - cy)
            nearest = dists.nsmallest(K_NEAREST).index
            near_ids.update(nearest)

    sn_all = dual_df[dual_df["supernumerary_present"]]
    near_df = dual_df.loc[dual_df.index.isin(near_ids)]
    far_same_image_df = sn_all.loc[~sn_all.index.isin(near_ids)]

    print(f"\n{'group':>28} {'accuracy':>10} {'n':>6}")
    print(f"{'near (K=3) supernumerary':>28} {near_df['correct'].mean():>10.4f} {len(near_df):>6}")
    print(f"{'far, same image':>28} {far_same_image_df['correct'].mean():>10.4f} {len(far_same_image_df):>6}")
    print(f"{'control (all 477 images)':>28} {ctrl['correct'].mean():>10.4f} {len(ctrl):>6}")

    p_near, n_near = near_df["correct"].mean(), len(near_df)
    p_far, n_far = far_same_image_df["correct"].mean(), len(far_same_image_df)
    p_pool2 = (near_df["correct"].sum() + far_same_image_df["correct"].sum()) / (n_near + n_far)
    se2 = np.sqrt(p_pool2 * (1 - p_pool2) * (1 / n_near + 1 / n_far))
    z2 = (p_near - p_far) / se2 if se2 > 0 else float("nan")
    p2 = 2 * (1 - stats.norm.cdf(abs(z2)))
    print(f"\nnear vs. far-same-image: z = {z2:.3f}, p = {p2:.4f}")

    p_pool3 = (near_df["correct"].sum() + ctrl["correct"].sum()) / (n_near + n2)
    se3 = np.sqrt(p_pool3 * (1 - p_pool3) * (1 / n_near + 1 / n2))
    z3 = (p_near - ctrl["correct"].mean()) / se3
    p3 = 2 * (1 - stats.norm.cdf(abs(z3)))
    print(f"near vs. control: z = {z3:.3f}, p = {p3:.4f}")

    print("\n=== Step 5: control for teeth-count/crowding confound ===")
    print("More teeth in an image could hurt geometry-based accuracy on its")
    print("own, independent of whether one of them is supernumerary. Fit")
    print("per-image accuracy ~ supernumerary_present + n_teeth (OLS) to see")
    print("whether the effect survives controlling for crowding.")
    n_teeth = dual_df.groupby("image_id").size().rename("n_teeth")
    per_image_full = dual_df.groupby("image_id").agg(
        accuracy=("correct", "mean"),
        supernumerary_present=("supernumerary_present", "first"),
    ).join(n_teeth)

    r, r_p = stats.pearsonr(per_image_full["n_teeth"], per_image_full["accuracy"])
    print(f"\nCorrelation, n_teeth vs. accuracy (all 500 images): "
          f"r = {r:.4f}, p = {r_p:.4f}")

    import statsmodels.api as sm
    X = per_image_full[["supernumerary_present", "n_teeth"]].astype(float)
    X = sm.add_constant(X)
    y = per_image_full["accuracy"]
    ols = sm.OLS(y, X).fit()
    print("\nOLS: accuracy ~ supernumerary_present + n_teeth")
    print(ols.summary().tables[1])

    print("\ndone")


if __name__ == "__main__":
    main()
