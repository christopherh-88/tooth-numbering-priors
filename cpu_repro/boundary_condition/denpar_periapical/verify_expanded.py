"""Task 4 (external-review-driven): expand Section 25's box-to-FDI
verification from a 2-image hand-check to (a) a real programmatic
consistency check across all 1000 spreadsheet rows, and (b) a visual
spot-check on a random (not hand-picked) sample of the images actually
used in Section 25's analysis.

(a) Programmatic check: the spreadsheet's own Arch (Upper/Lower) and Site
(Right/Left/Anterior) metadata is independently checked against the FDI
quadrant digit each row's code list implies - FDI quadrant 1/2 = upper,
3/4 = lower; quadrant 1/4 = right side, 2/3 = left side. This catches
FDI-code data-entry errors in the spreadsheet itself, at 100% coverage
(not a sample) - a real automated check, not a proxy for the box-order
verification, which has no analogous full-coverage check available (no
independent second source gives box-to-tooth identity).

(b) Visual spot-check: since no full-coverage programmatic check exists
for the box-order-to-FDI-code correspondence itself, a genuinely random
sample (numpy RandomState(1), distinct from the seed used elsewhere in
this project, and not the same 2 images originally spot-checked) of
images actually used in Section 25's 633-image analysis set is rendered
with boxes and their assigned FDI labels drawn directly on the
radiograph, for visual review and a reported pass rate - not another
hand-picked pair.
"""

import sys
from pathlib import Path

import numpy as np
import openpyxl
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_coord_baseline_denpar import load_fdi_lists, find_image_file, RAW_DIR
import json

OUT_DIR = Path(__file__).resolve().parent
REVIEW_DIR = OUT_DIR / "verification_review"
N_SAMPLE = 20


def programmatic_arch_site_check():
    wb = openpyxl.load_workbook(RAW_DIR / "Characteristics.xlsx")
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))[1:]

    n_checked, n_arch_fail, n_site_fail = 0, 0, 0
    arch_failures, site_failures = [], []

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
        n_checked += 1

        # FDI quadrant digits: 1/2 = upper permanent, 3/4 = lower permanent,
        # 5/6 = upper primary (deciduous), 7/8 = lower primary - primary-tooth
        # quadrants must be included here or every primary-tooth row (which
        # this dataset does contain, per build_coord_baseline_denpar.py's own
        # exclusion count) would be miscounted as an Arch/Site mismatch that
        # isn't real. A first version of this check omitted 5-8 and reported
        # an inflated failure count - fixed before reporting any number.
        expected_arch_quads = {"1", "2", "5", "6"} if arch == "Upper" else \
                               {"3", "4", "7", "8"} if arch == "Lower" else None
        if expected_arch_quads is not None and not quadrants.issubset(expected_arch_quads):
            n_arch_fail += 1
            arch_failures.append((int(image_id), arch, code_str))

        if site in ("Right", "Left"):
            expected_site_quads = {"1", "4", "5", "8"} if site == "Right" else {"2", "3", "6", "7"}
            if not quadrants.issubset(expected_site_quads):
                n_site_fail += 1
                site_failures.append((int(image_id), site, code_str))

    print("=" * 70)
    print("TASK 4a: programmatic Arch/Site vs. FDI-quadrant consistency check")
    print(f"(full coverage: all {n_checked} spreadsheet rows with parseable codes)")
    print("=" * 70)
    print(f"Arch (Upper/Lower) vs. quadrant digit: {n_checked - n_arch_fail}/{n_checked} consistent "
          f"({100*(n_checked-n_arch_fail)/n_checked:.2f}%)")
    if arch_failures:
        print(f"  Failures: {arch_failures[:10]}")
    print(f"Site (Right/Left, excl. Anterior) vs. quadrant digit: "
          f"{n_checked - n_site_fail}/{n_checked} consistent")
    if site_failures:
        print(f"  Failures: {site_failures[:10]}")
    return n_checked, n_arch_fail, n_site_fail, arch_failures, site_failures


def render_review_image(image_id, fdi_codes, out_path):
    split, img_path = find_image_file(image_id)
    json_path = RAW_DIR / "Dataset" / split / "Key Points Annotations" / f"{image_id}.json"
    data = json.loads(json_path.read_text())
    boxes = data.get("bboxes", [])
    if len(boxes) != len(fdi_codes):
        return False, f"box/label count mismatch ({len(boxes)} boxes vs {len(fdi_codes)} codes)"

    boxes_sorted = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2)

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except Exception:
        font = ImageFont.load_default()

    for box, code in zip(boxes_sorted, fdi_codes):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 32)), code, fill="red", font=font)

    img.save(out_path)
    return True, None


def main():
    programmatic_arch_site_check()

    print("\n" + "=" * 70)
    print(f"TASK 4b: random visual spot-check, n={N_SAMPLE} images "
          f"(from Section 25's actual 633-image used set)")
    print("=" * 70)

    fdi_lists = load_fdi_lists()
    all_ids = sorted(fdi_lists.keys())
    rng = np.random.RandomState(1)  # distinct seed from the project's usual seed 0
    sample_ids = rng.choice(all_ids, size=min(N_SAMPLE, len(all_ids)), replace=False)

    REVIEW_DIR.mkdir(exist_ok=True)
    rendered = []
    for image_id in sample_ids:
        out_path = REVIEW_DIR / f"review_{image_id}.jpg"
        ok, err = render_review_image(image_id, fdi_lists[image_id], out_path)
        rendered.append((image_id, ok, err, str(out_path) if ok else None))
        print(f"  {image_id:>6s}  {'rendered' if ok else 'SKIPPED: ' + err}")

    n_rendered = sum(1 for _, ok, _, _ in rendered if ok)
    print(f"\n{n_rendered}/{len(rendered)} sampled images rendered for visual review "
          f"(others skipped - box/label count mismatch, same exclusion rule as the main pipeline).")
    print(f"Review images saved to {REVIEW_DIR}")

    import csv
    with open(REVIEW_DIR / "sample_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "rendered", "error", "path"])
        w.writerows(rendered)


if __name__ == "__main__":
    main()
