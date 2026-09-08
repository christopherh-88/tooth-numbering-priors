"""Same annotation screening pass as scan_annotations.py (UFBA-425), applied
to DENTEX (arXiv 2305.19112, MICCAI 2023 challenge, Hamamci et al.).

DENTEX ships three annotation tiers per image, not one: quadrant-only
(693 images, no per-tooth FDI code - see NOTE below), quadrant+enumeration
(634 images, full FDI code, no diagnosis), and quadrant+enumeration+
diagnosis (705 train + 50 validation = 755 images, full FDI code AND a
diagnosis joined on the same box). Only the last two tiers carry a full
FDI code and are scanned here; quadrant-only annotations have no
enumeration digit and are structurally excluded (see dentex_findings.md).

NOTE on a real footgun found in DENTEX's own files, not assumed away: the
quadrant-only tier's own categories list is NOT in id-order (category id 0
maps to quadrant name '2', not '1' - see dentex_findings.md). The
quadrant-enumeration and quadrant-enumeration-disease tiers ARE in aligned
id order, but this script (via build_table.py) always resolves FDI codes
through each file's own categories_1/categories_2 id->name lookup rather
than assuming id+1 anywhere, specifically because of that inconsistency.

CONVENTION: DENTEX's own quadrant-to-image-region layout was verified
empirically against its own bulk statistics (median/percentile x,y per
quadrant - see dentex_findings.md) BEFORE reusing UFBA-425's position
thresholds here. It matches UFBA-425's convention (quadrants 1,4 ->
image-left; 2,3 -> image-right; 1,2 -> image-upper; 3,4 -> image-lower)
and has a similar-or-tighter natural spread, so the same buffered
thresholds are used - this was checked, not assumed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TABLE_PATH = HERE / "dentex_raw" / "dentex_full_fdi_table.csv"
OUTPUT_DIR = HERE / "dentex_results"

FULL_DENTITION_PERMANENT = {f"{q}{p}" for q in "1234" for p in "12345678"}

# Same buffered thresholds as scan_annotations.py (UFBA-425) - verified
# defensible for DENTEX's own natural range in dentex_findings.md, not
# reused blindly.
X_LEFT_MAX = 0.55
X_RIGHT_MIN = 0.45
Y_UPPER_MAX = 0.55
Y_LOWER_MIN = 0.45
EXPECTED_X_SIDE = {"1": "left", "2": "right", "3": "right", "4": "left"}
EXPECTED_Y_LEVEL = {"1": "upper", "2": "upper", "3": "lower", "4": "lower"}

WEIGHT_DUPLICATE_EXTRA_BOX = 3.0
WEIGHT_POSITION_BOTH_AXES = 5.0
WEIGHT_POSITION_SINGLE_AXIS = 2.0
WEIGHT_MISSING_CODE = 0.05
WEIGHT_SPARSE_PENALTY = 5.0
SPARSE_THRESHOLD = 8
WEIGHT_OVERCOUNT_PER_EXTRA = 2.0


def check_position(quadrant, x, y):
    x_bad = None
    if EXPECTED_X_SIDE[quadrant] == "left" and x > X_LEFT_MAX:
        x_bad = (f"expected image-left (x<={X_LEFT_MAX}), got x={x:.3f}", "x", x)
    elif EXPECTED_X_SIDE[quadrant] == "right" and x < X_RIGHT_MIN:
        x_bad = (f"expected image-right (x>={X_RIGHT_MIN}), got x={x:.3f}", "x", x)

    y_bad = None
    if EXPECTED_Y_LEVEL[quadrant] == "upper" and y > Y_UPPER_MAX:
        y_bad = (f"expected image-upper (y<={Y_UPPER_MAX}), got y={y:.3f}", "y", y)
    elif EXPECTED_Y_LEVEL[quadrant] == "lower" and y < Y_LOWER_MIN:
        y_bad = (f"expected image-lower (y>={Y_LOWER_MIN}), got y={y:.3f}", "y", y)

    return x_bad, y_bad


def scan_image(image_id, group):
    fdi_counts = group["fdi"].value_counts()
    duplicate_codes = fdi_counts[fdi_counts > 1]
    missing_codes = sorted(FULL_DENTITION_PERMANENT - set(group["fdi"]))

    position_notes = []
    single_axis_values = {"x": [], "y": []}
    n_single_axis, n_both_axis = 0, 0
    for _, row in group.iterrows():
        x_bad, y_bad = check_position(row["quadrant"], row["x_center"], row["y_center"])
        if x_bad or y_bad:
            if x_bad and y_bad:
                n_both_axis += 1
                position_notes.append(f"{row['fdi']}: WRONG QUADRANT (both axes) - {x_bad[0]}; {y_bad[0]}")
            else:
                n_single_axis += 1
                msg, axis, value = x_bad or y_bad
                single_axis_values[axis].append(value)
                position_notes.append(f"{row['fdi']}: {msg}")

    likely_global_framing = any(
        len(v) >= 3 and (max(v) - min(v)) <= 0.15 for v in single_axis_values.values()
    )

    total = len(group)
    score, reasons = 0.0, []

    if len(duplicate_codes) > 0:
        extra = int((duplicate_codes - 1).sum())
        score += extra * WEIGHT_DUPLICATE_EXTRA_BOX
        reasons.append(f"duplicate_codes(+{extra * WEIGHT_DUPLICATE_EXTRA_BOX:.1f})")
    if n_both_axis > 0:
        score += n_both_axis * WEIGHT_POSITION_BOTH_AXES
        reasons.append(f"wrong_quadrant(+{n_both_axis * WEIGHT_POSITION_BOTH_AXES:.1f})")
    if n_single_axis > 0:
        score += n_single_axis * WEIGHT_POSITION_SINGLE_AXIS
        reasons.append(f"single_axis_position(+{n_single_axis * WEIGHT_POSITION_SINGLE_AXIS:.1f})")
    if len(missing_codes) > 0:
        m = len(missing_codes) * WEIGHT_MISSING_CODE
        score += m
        reasons.append(f"missing_codes(+{m:.2f})")
    if total < SPARSE_THRESHOLD:
        score += WEIGHT_SPARSE_PENALTY
        reasons.append(f"sparse_annotation(+{WEIGHT_SPARSE_PENALTY:.1f})")
    if total > 32:
        over = (total - 32) * WEIGHT_OVERCOUNT_PER_EXTRA
        score += over
        reasons.append(f"overcount(+{over:.1f})")

    primary_reason = "none"
    if reasons:
        primary_reason = max(reasons, key=lambda r: float(r.rsplit("(+", 1)[1].rstrip(")"))).split("(")[0]

    return {
        "image_id": image_id,
        "file_name": group["file_name"].iloc[0],
        "split": group["split"].iloc[0],
        "total_tooth_count": total,
        "n_duplicate_codes": len(duplicate_codes),
        "duplicate_codes": ", ".join(f"{c}x{n}" for c, n in duplicate_codes.items()) or "-",
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
    df = pd.read_csv(TABLE_PATH, dtype={"quadrant": str, "position": str, "fdi": str})
    print(f"Loaded {len(df)} tooth instances (full FDI code) from {df['image_id'].nunique()} images")

    rows = [scan_image(img_id, g) for img_id, g in df.groupby("image_id")]
    result = pd.DataFrame(rows).sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result.to_csv(OUTPUT_DIR / "full_scan_dentex.csv", index=False)
    result.head(100).to_csv(OUTPUT_DIR / "candidates_dentex.csv", index=False)

    n_flagged = int((result["anomaly_score"] > 0).sum())
    print(f"{n_flagged} / {len(result)} images have a nonzero anomaly score.")

    cols = ["rank", "file_name", "split", "total_tooth_count", "n_duplicate_codes",
            "n_position_violations", "likely_global_framing", "n_missing_codes",
            "anomaly_score", "primary_reason"]
    print("\nTop 20 most anomalous images:")
    print(result[cols].head(20).to_string(index=False))

    print("\nAnomaly reason category counts (among flagged images):")
    print(result[result["anomaly_score"] > 0]["primary_reason"].value_counts().to_string())

    print(f"\nSaved full_scan_dentex.csv ({len(result)} rows) and candidates_dentex.csv to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
