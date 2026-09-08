"""Screening pass over the UFBA-425 YOLO annotations
(Dataset/yolo_train_dataset/) for cases where "position implies FDI code"
should break: duplicate codes, missing codes, and boxes whose position
contradicts their FDI quadrant.

This is a data-quality screen, not a clinical read. It flags candidates for
a human to look at; it does not diagnose anything about the patient.

Unit of analysis: one row per YOLO label FILE (each .txt file is one
detector-visible image - including each Roboflow-augmented crop of a source
X-ray as its own entry, since that's what a model actually sees and what
"total tooth count in this image" naturally means here). See README.md for
why we did not collapse augmented copies of the same source X-ray into one
row.

Checks
------
1. Total tooth count - how many boxes are annotated in this image.
2. Duplicate FDI codes - the same class_id (FDI code) annotated more than
   once in one image. Two teeth cannot share one FDI number.
3. Missing FDI codes - codes absent from the image, relative to the full
   32-code adult dentition. This is expected/common in this dataset (many
   UFBA-425 categories are explicitly partial dentition, implants, etc. -
   see the repo's top-level README) so it is reported for every image but
   weighted lightly in the anomaly score; it's context, not on its own
   evidence of a labeling error.
4. Position-vs-code inconsistency - does a box's (x, y) sit in the image
   region its FDI code implies? FDI quadrant convention, in *radiographic*
   (mirrored) image coordinates, confirmed empirically from the dataset's
   own bulk statistics (see module docstring below and
   `reference_quadrant_stats.csv`):
     quadrant 1 (upper right) -> image-left,  image-upper
     quadrant 2 (upper left)  -> image-right, image-upper
     quadrant 3 (lower left)  -> image-right, image-lower
     quadrant 4 (lower right) -> image-left,  image-lower
   A box is flagged if it sits clearly on the wrong side (x) and/or wrong
   level (y) for its code's quadrant, using thresholds set with a buffer
   past the dataset's own 5th/95th-percentile natural range per quadrant
   (see CONFIG) - so ordinary anatomical variation near the midline or
   occlusal plane is not flagged, only boxes that land solidly in the
   opposite region.

Anomaly score
-------------
A simple, documented, additive score per image (see WEIGHTS below) -
duplicates and position contradictions dominate the ranking; missing-code
count and total-count extremes contribute lightly. This is a heuristic for
sorting candidates for review, not a statistical test.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coord_baseline"))
from build_coord_baseline import FDI_CODES, load_instances  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent

ALL_CODES = set(FDI_CODES)

# Position-consistency thresholds. The dataset's own 5th/95th percentile
# natural ranges per quadrant (computed in the module docstring / dev
# notes) place the boundary right around x=0.50 / y=0.50 with almost no
# overlap; these thresholds add a ~0.05 buffer past that boundary so normal
# midline/occlusal-plane variation isn't flagged, only boxes clearly on the
# wrong side.
X_LEFT_MAX = 0.55   # quadrants 1,4 (expected image-left) flagged if x_center exceeds this
X_RIGHT_MIN = 0.45  # quadrants 2,3 (expected image-right) flagged if x_center is below this
Y_UPPER_MAX = 0.55  # quadrants 1,2 (expected image-upper) flagged if y_center exceeds this
Y_LOWER_MIN = 0.45  # quadrants 3,4 (expected image-lower) flagged if y_center is below this

EXPECTED_X_SIDE = {"1": "left", "2": "right", "3": "right", "4": "left"}
EXPECTED_Y_LEVEL = {"1": "upper", "2": "upper", "3": "lower", "4": "lower"}

# Anomaly-score weights. Duplicates and full quadrant contradictions (wrong
# on BOTH axes - the clearest anatomically-impossible case) dominate;
# single-axis position anomalies are weighted less since they're closer to
# the natural-variation boundary; missing-code count and count extremes are
# informational nudges, not primary drivers, because partial dentition is
# common and expected in this dataset by design.
WEIGHT_DUPLICATE_EXTRA_BOX = 3.0     # per redundant box beyond the first, per duplicated code
WEIGHT_POSITION_SINGLE_AXIS = 2.0    # per box wrong on exactly one axis (x or y)
WEIGHT_POSITION_BOTH_AXES = 5.0      # per box wrong on both axes (opposite-quadrant contradiction)
WEIGHT_MISSING_CODE = 0.05           # per missing FDI code (light, informational)
WEIGHT_SPARSE_PENALTY = 5.0          # flat penalty if total_tooth_count < SPARSE_THRESHOLD
SPARSE_THRESHOLD = 8
WEIGHT_OVERCOUNT_PER_EXTRA = 2.0     # per tooth beyond 32 (anatomically impossible for permanent dentition)


def check_position(row):
    quadrant = FDI_CODES[row["class_id"]][0]
    x, y = row["x_center"], row["y_center"]

    x_bad = None
    expected_x = EXPECTED_X_SIDE[quadrant]
    if expected_x == "left" and x > X_LEFT_MAX:
        x_bad = (f"expected image-left (x<={X_LEFT_MAX}), got x={x:.3f}", "x", x)
    elif expected_x == "right" and x < X_RIGHT_MIN:
        x_bad = (f"expected image-right (x>={X_RIGHT_MIN}), got x={x:.3f}", "x", x)

    y_bad = None
    expected_y = EXPECTED_Y_LEVEL[quadrant]
    if expected_y == "upper" and y > Y_UPPER_MAX:
        y_bad = (f"expected image-upper (y<={Y_UPPER_MAX}), got y={y:.3f}", "y", y)
    elif expected_y == "lower" and y < Y_LOWER_MIN:
        y_bad = (f"expected image-lower (y>={Y_LOWER_MIN}), got y={y:.3f}", "y", y)

    return x_bad, y_bad


def scan_image(label_file, group):
    codes_present = [FDI_CODES[c] for c in group["class_id"]]
    code_counts = pd.Series(codes_present).value_counts()
    duplicate_codes = code_counts[code_counts > 1]

    present_set = set(codes_present)
    missing_codes = sorted(ALL_CODES - present_set)

    position_notes = []
    single_axis_values = {"x": [], "y": []}  # for the clustering check below
    n_single_axis = 0
    n_both_axis = 0
    for _, row in group.iterrows():
        x_bad, y_bad = check_position(row)
        if x_bad or y_bad:
            code = FDI_CODES[row["class_id"]]
            if x_bad and y_bad:
                n_both_axis += 1
                position_notes.append(f"{code}: WRONG QUADRANT (both axes) - {x_bad[0]}; {y_bad[0]}")
            else:
                n_single_axis += 1
                bad = x_bad or y_bad
                msg, axis, value = bad
                single_axis_values[axis].append(value)
                position_notes.append(f"{code}: {msg}")

    # Heuristic: if several boxes in the SAME image all cross the threshold
    # on the same axis by a similarly small margin, that pattern looks more
    # like a rotated/cropped/tilted source X-ray shifting the whole
    # upper-lower or left-right boundary for this image, than N independent
    # per-tooth mislabeling errors. Flag it separately so it can be
    # deprioritized on manual review relative to duplicate_codes/
    # wrong_quadrant or an isolated single large-magnitude deviation.
    likely_global_framing = False
    for axis_values in single_axis_values.values():
        if len(axis_values) >= 3 and (max(axis_values) - min(axis_values)) <= 0.15:
            likely_global_framing = True

    total = len(group)

    score = 0.0
    reasons = []

    if len(duplicate_codes) > 0:
        extra_boxes = int((duplicate_codes - 1).sum())
        score += extra_boxes * WEIGHT_DUPLICATE_EXTRA_BOX
        reasons.append(f"duplicate_codes(+{extra_boxes * WEIGHT_DUPLICATE_EXTRA_BOX:.1f})")

    if n_both_axis > 0:
        score += n_both_axis * WEIGHT_POSITION_BOTH_AXES
        reasons.append(f"wrong_quadrant(+{n_both_axis * WEIGHT_POSITION_BOTH_AXES:.1f})")

    if n_single_axis > 0:
        score += n_single_axis * WEIGHT_POSITION_SINGLE_AXIS
        reasons.append(f"single_axis_position(+{n_single_axis * WEIGHT_POSITION_SINGLE_AXIS:.1f})")

    if len(missing_codes) > 0:
        missing_contribution = len(missing_codes) * WEIGHT_MISSING_CODE
        score += missing_contribution
        reasons.append(f"missing_codes(+{missing_contribution:.2f})")

    if total < SPARSE_THRESHOLD:
        score += WEIGHT_SPARSE_PENALTY
        reasons.append(f"sparse_annotation(+{WEIGHT_SPARSE_PENALTY:.1f})")

    if total > 32:
        overcount = total - 32
        score += overcount * WEIGHT_OVERCOUNT_PER_EXTRA
        reasons.append(f"overcount(+{overcount * WEIGHT_OVERCOUNT_PER_EXTRA:.1f})")

    primary_reason = "none"
    if reasons:
        # pick the single largest contributor as the headline reason
        primary_reason = max(
            reasons, key=lambda r: float(r.rsplit("(+", 1)[1].rstrip(")"))
        ).split("(")[0]

    return {
        "label_file": label_file,
        "image_id": group["image_id"].iloc[0],
        "total_tooth_count": total,
        "n_duplicate_codes": len(duplicate_codes),
        "duplicate_codes": ", ".join(f"{code}x{cnt}" for code, cnt in duplicate_codes.items()) or "-",
        "n_missing_codes": len(missing_codes),
        "missing_codes": ", ".join(missing_codes) if missing_codes else "-",
        "n_position_violations": n_single_axis + n_both_axis,
        "n_wrong_quadrant_both_axes": n_both_axis,
        "likely_global_framing": likely_global_framing,
        "position_violation_detail": "; ".join(position_notes) if position_notes else "-",
        "anomaly_score": round(score, 2),
        "primary_reason": primary_reason,
        "reason_breakdown": ", ".join(reasons) if reasons else "-",
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_instances()
    print(f"Loaded {len(df)} tooth instances from {df['label_file'].nunique()} label files "
          f"({df['image_id'].nunique()} unique source X-rays).")

    # reference stats, saved for transparency about the convention used above
    df["quadrant"] = df["class_id"].apply(lambda c: FDI_CODES[c][0])
    ref_stats = df.groupby("quadrant")[["x_center", "y_center"]].agg(["median", "mean", "std"])
    ref_stats.to_csv(OUTPUT_DIR / "reference_quadrant_stats.csv")

    rows = [scan_image(lf, g) for lf, g in df.groupby("label_file")]
    result_df = pd.DataFrame(rows).sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    result_df.insert(0, "rank", np.arange(1, len(result_df) + 1))

    result_df.to_csv(OUTPUT_DIR / "full_scan.csv", index=False)

    n_flagged = int((result_df["anomaly_score"] > 0).sum())
    print(f"{n_flagged} / {len(result_df)} images have a nonzero anomaly score.")

    top_n = 100
    candidates = result_df.head(top_n)
    candidates.to_csv(OUTPUT_DIR / "candidates.csv", index=False)

    print(f"\nTop 20 most anomalous images:")
    cols = ["rank", "label_file", "total_tooth_count", "n_duplicate_codes",
            "n_position_violations", "likely_global_framing", "n_missing_codes",
            "anomaly_score", "primary_reason"]
    print(candidates[cols].head(20).to_string(index=False))

    print("\nAnomaly reason category counts (among flagged images):")
    flagged = result_df[result_df["anomaly_score"] > 0]
    print(flagged["primary_reason"].value_counts().to_string())

    print(f"\nSaved full_scan.csv ({len(result_df)} rows) and candidates.csv (top {top_n}) to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
