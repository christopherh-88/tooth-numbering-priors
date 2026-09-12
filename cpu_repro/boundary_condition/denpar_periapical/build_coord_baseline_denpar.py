"""Coordinate-only baseline, replicated on DenPAR (Rasnayaka et al., Scientific
Data 12:1615, 2025) - a periapical (IOPA) radiograph dataset, CC BY 4.0,
https://doi.org/10.5281/zenodo.16645076 - as the first real test of the
falsifiable boundary-condition hypothesis in RESULTS.md Section 11.4.

UFBA-425 and DENTEX (Sections 2, 10) are both panoramic radiographs: full
jaw in one canonically-framed image, satisfying both stated preconditions
(a near-deterministic spatial slot per FDI class, and a canonicalized
acquisition protocol pinning that slot to a consistent pixel location).
Periapical radiographs break precondition 2 on purpose: each image shows a
small, variable subset of teeth (1-8 in this dataset, mean 4.25), framed
however that region happened to be positioned for that patient/exposure -
there is no consistent "where quadrant 3 sits in the frame" the way there
is in a panoramic radiograph. If the coordinate-only shortcut is really
conditioned on precondition 2 as hypothesized, it should be substantially
weaker here than the ~69-70% seen on UFBA-425/DENTEX.

IMPORTANT - box-to-FDI-code correspondence is NOT given directly by the
dataset and was independently verified before this script was written, not
assumed:
  - Each image's annotation JSON (Key Points Annotations/<id>.json) gives a
    list of pixel-space bounding boxes in no documented order.
  - The dataset's own characteristics spreadsheet
    (RawData/Characteristics.xlsx) gives, per image, an ordered list of the
    FDI codes visible ("FDI notation of fully/partially visible teeth"),
    always written in ascending numeric order (e.g. "44,45,46,47,48").
  - Two images (id 1: lower-right premolars/molars, id 2: lower-left
    molars) were checked by hand: sorting that image's boxes by x-center
    ascending and zipping with the FDI list in the order given reproduces
    anatomically correct tooth shapes (premolar-shaped crown at the 44/36
    end, molar/wisdom-tooth shapes progressing toward 48/38) in both cases,
    for both a "Right" and a "Left" site image. This script applies that
    same rule (sort boxes by x-center ascending, zip with the FDI list as
    given) to every image, and additionally asserts the FDI list itself is
    strictly ascending per image as an automatic sanity check - a
    non-ascending list would mean this convention doesn't hold for that
    row, and the image is dropped rather than silently mislabeled.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "coord_baseline"))
from build_coord_baseline import (  # noqa: E402
    is_mirror_quadrant_error,
    is_neighbor_error,
    mean_ci95,
    quadrant_of,
    tooth_type_of,
)

RAW_DIR = Path(__file__).resolve().parent / "RawData"
OUTPUT_DIR = Path(__file__).resolve().parent
SPLITS = ["Training", "Validation", "Testing"]
SEEDS = [0, 1, 2, 3, 4]
TEST_FRACTION = 0.2
CONFUSION_MATRIX_SEED = SEEDS[0]

FDI_CODES = [
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
]
CODE_TO_IDX = {c: i for i, c in enumerate(FDI_CODES)}
FEATURE_COLS = ["x_center", "y_center", "width", "height", "area", "aspect_ratio"]


def load_fdi_lists() -> dict:
    """image_id (str, no extension) -> ordered list of FDI code strings, as
    given in the spreadsheet. Rows with unparseable or non-permanent-tooth
    codes are excluded (11 of 1000 - primary/deciduous-tooth codes in the
    51-85 range, or malformed entries), not silently coerced."""
    wb = openpyxl.load_workbook(RAW_DIR / "Characteristics.xlsx")
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))[1:]
    out = {}
    n_dropped = 0
    for r in rows:
        if r[0] is None:
            continue
        image_id = str(int(r[0]))
        try:
            codes = [str(int(float(c.strip()))) for c in str(r[3]).split(",") if c.strip()]
        except Exception:
            n_dropped += 1
            continue
        if not codes or not all(c in CODE_TO_IDX for c in codes):
            n_dropped += 1
            continue
        if any(int(codes[i]) >= int(codes[i + 1]) for i in range(len(codes) - 1)):
            # Not strictly ascending - the verified ordering convention
            # doesn't hold for this row. Drop rather than guess.
            n_dropped += 1
            continue
        out[image_id] = codes
    print(f"Loaded FDI lists for {len(out)} images ({n_dropped} dropped - "
          f"primary-tooth codes, malformed entries, or non-ascending order).")
    return out


def find_image_file(image_id: str):
    for split in SPLITS:
        p = RAW_DIR / "Dataset" / split / "Images" / f"{image_id}.jpg"
        if p.exists():
            return split, p
    return None, None


def load_instances() -> pd.DataFrame:
    fdi_lists = load_fdi_lists()
    rows = []
    n_count_mismatch = 0
    n_missing_files = 0

    for image_id, fdi_codes in fdi_lists.items():
        split, img_path = find_image_file(image_id)
        if img_path is None:
            n_missing_files += 1
            continue
        json_path = RAW_DIR / "Dataset" / split / "Key Points Annotations" / f"{image_id}.json"
        if not json_path.exists():
            n_missing_files += 1
            continue

        data = json.loads(json_path.read_text())
        boxes = data.get("bboxes", [])
        if len(boxes) != len(fdi_codes):
            n_count_mismatch += 1
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        # Verified convention: sort this image's boxes by x-center ascending,
        # zip with the FDI list (already ascending) in that order.
        boxes_sorted = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2)

        for box, code in zip(boxes_sorted, fdi_codes):
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2 / img_w
            y_center = (y1 + y2) / 2 / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            rows.append({
                "image_id": image_id,
                "split_folder": split,
                "class_id": CODE_TO_IDX[code],
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            })

    print(f"{n_count_mismatch} images dropped (bbox count != FDI-list count - "
          f"ambiguous correspondence, not guessed at).")
    print(f"{n_missing_files} images dropped (image or annotation file not found).")

    df = pd.DataFrame(rows)
    df["area"] = df["width"] * df["height"]
    df["aspect_ratio"] = df["width"] / df["height"].replace(0, np.nan)
    df["aspect_ratio"] = df["aspect_ratio"].fillna(df["aspect_ratio"].median())
    return df


def grouped_split(df: pd.DataFrame, seed: int, test_fraction: float):
    # Each DenPAR image is already one independent unit (no augmentation
    # duplication the way Roboflow-exported UFBA-425 has) - grouping by
    # image_id is still done for consistency with build_coord_baseline.py's
    # methodology, even though here it is equivalent to a plain instance
    # split at the image level.
    image_ids = np.array(df["image_id"].unique().tolist(), dtype=object)
    rng = np.random.RandomState(seed)
    shuffled = image_ids.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_fraction))
    test_ids = set(shuffled[:n_test])
    train_ids = set(shuffled[n_test:])
    return df[df["image_id"].isin(train_ids)], df[df["image_id"].isin(test_ids)]


def evaluate(y_true, y_pred, train_y):
    top1 = float(np.mean(y_true == y_pred))
    quad_acc = float(np.mean(quadrant_of(y_true) == quadrant_of(y_pred)))
    type_acc = float(np.mean(tooth_type_of(y_true) == tooth_type_of(y_pred)))
    majority_class = np.bincount(train_y, minlength=32).argmax()
    majority_acc = float(np.mean(y_true == majority_class))
    wrong = y_true != y_pred
    n_wrong = int(wrong.sum())
    if n_wrong > 0:
        mirror_frac = float(np.mean([is_mirror_quadrant_error(t, p)
                                      for t, p in zip(y_true[wrong], y_pred[wrong])]))
        neighbor_frac = float(np.mean([is_neighbor_error(t, p)
                                        for t, p in zip(y_true[wrong], y_pred[wrong])]))
    else:
        mirror_frac, neighbor_frac = 0.0, 0.0
    return {
        "top1_acc": top1, "quadrant_acc": quad_acc, "tooth_type_acc": type_acc,
        "majority_baseline_acc": majority_acc,
        "mirror_quadrant_error_frac": mirror_frac, "neighbor_error_frac": neighbor_frac,
        "n_test": len(y_true),
    }


def main():
    df = load_instances()
    n_images = df["image_id"].nunique()
    print(f"\nFinal instance table: {len(df)} tooth instances from {n_images} images.")
    print(f"Teeth per image: min={df.groupby('image_id').size().min()}, "
          f"max={df.groupby('image_id').size().max()}, "
          f"mean={df.groupby('image_id').size().mean():.2f} "
          f"(UFBA-425/DENTEX show every visible tooth of a full jaw per image; "
          f"DenPAR images show a small, variable partial-arch subset by design).")
    print()

    classifiers = {
        "logistic_regression": lambda: LogisticRegression(max_iter=2000),
        "gradient_boosted_tree": lambda: HistGradientBoostingClassifier(random_state=0),
    }
    per_seed_results = {name: [] for name in classifiers}
    confusion_data = {}

    for seed in SEEDS:
        train_df, test_df = grouped_split(df, seed, TEST_FRACTION)
        x_train = train_df[FEATURE_COLS].to_numpy()
        y_train = train_df["class_id"].to_numpy()
        x_test = test_df[FEATURE_COLS].to_numpy()
        y_test = test_df["class_id"].to_numpy()

        scaler = StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        for name, make_clf in classifiers.items():
            clf = make_clf()
            if name == "logistic_regression":
                clf.fit(x_train_scaled, y_train)
                y_pred = clf.predict(x_test_scaled)
            else:
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)

            metrics = evaluate(y_test, y_pred, y_train)
            metrics["seed"] = seed
            per_seed_results[name].append(metrics)
            if seed == CONFUSION_MATRIX_SEED:
                confusion_data[name] = (y_test, y_pred)

        print(f"seed {seed}: train_images={train_df['image_id'].nunique()} "
              f"test_images={test_df['image_id'].nunique()} "
              f"train_instances={len(train_df)} test_instances={len(test_df)}")

    print("\n" + "=" * 88)
    print("SUMMARY (mean +/- 95% CI over 5 seeds, image-level grouped split) - DenPAR periapical")
    print("=" * 88)
    summary_rows = []
    metric_keys = ["top1_acc", "quadrant_acc", "tooth_type_acc",
                    "majority_baseline_acc", "mirror_quadrant_error_frac", "neighbor_error_frac"]
    for name, results in per_seed_results.items():
        print(f"\n{name}")
        row = {"classifier": name}
        for key in metric_keys:
            values = [r[key] for r in results]
            mean, ci = mean_ci95(values)
            print(f"  {key:28s} {mean:.4f} +/- {ci:.4f}")
            row[f"{key}_mean"] = mean
            row[f"{key}_ci95"] = ci
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "summary.csv", index=False)
    per_seed_rows = [{"classifier": name, **r} for name, results in per_seed_results.items() for r in results]
    pd.DataFrame(per_seed_rows).to_csv(OUTPUT_DIR / "per_seed_results.csv", index=False)

    for name, (y_test, y_pred) in confusion_data.items():
        cm = confusion_matrix(y_test, y_pred, labels=list(range(32)))
        pd.DataFrame(cm, index=FDI_CODES, columns=FDI_CODES).to_csv(
            OUTPUT_DIR / f"confusion_matrix_{name}_seed{CONFUSION_MATRIX_SEED}.csv")

    print(f"\nSaved summary.csv, per_seed_results.csv, confusion matrices to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
