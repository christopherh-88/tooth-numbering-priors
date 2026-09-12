"""Task 2 PIVOT-branch case study (GO_NO_GO.md Section 3): characterize
what visual signal the real trained detector (YOLOv8, seed-0 split) is
using, by studying cases where it is correct and the coordinate-only
baseline (GBT, same split) is wrong.

Exploratory/diagnostic - no pre-registered pass/fail rule applies here,
unlike Section 21's GO/NO-GO call. Runs entirely on CPU: the coordinate
model is refit locally in seconds (deterministic, seed 0, same as
build_coord_baseline.py's own run), and YOLO inference only needs to run
once over the ~198 val images using the checkpoint already downloaded
from the Kaggle run (runs/yolov8_seed0split/weights/best.pt) - no GPU
needed for this.

Join key: coordinate-only instances (from load_instances(), one row per
label-file line) and YOLO's ground-truth boxes (from load_gt_boxes(),
also one row per label-file line, same files) are both parsed by
splitting each label .txt file's lines in file order and are never
reordered or filtered before that point - so (label_file, line index) is
a stable, exact join key identifying the same physical tooth annotation
in both pipelines. This is verified below (see the assertion in
`build_coord_predictions`) rather than assumed.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_yolo import (  # noqa: E402
    base_image_id,
    iou_xyxy,
    load_gt_boxes,
    xywhn_to_xyxyn,
    MATCH_IOU_THRESHOLD,
    PREPARED_DIR,
    EVAL_CONF,
    EVAL_NMS_IOU,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coord_baseline"))
from build_coord_baseline import (  # noqa: E402
    FDI_CODES,
    FEATURE_COLS,
    TEST_FRACTION,
    grouped_split,
    is_mirror_quadrant_error,
    is_neighbor_error,
    load_instances,
    quadrant_of,
    tooth_type_of,
)

from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 0
REPO_ROOT = Path(__file__).resolve().parents[2]
YOLO_SOURCE_ROOT = REPO_ROOT / "Dataset" / "yolo_train_dataset"
BEST_WEIGHTS = Path(__file__).resolve().parent / "runs" / "yolov8_seed0split" / "weights" / "best.pt"

OUT_DIR = Path(__file__).resolve().parent / "eval_results"
CROPS_DIR = OUT_DIR / "case_study_crops"
N_SAMPLE = 18
CROP_PAD_FRAC = 1.5  # extra context around each box, as a fraction of box size -
# a tight crop (~0.15) showed only the single tooth with no neighbors, useless
# for judging crowding/adjacent-tooth context; widened after visually checking
# an initial tight-crop sample


def build_coord_predictions() -> pd.DataFrame:
    """Refit the coordinate-only GBT model exactly as build_coord_baseline.py
    does for seed 0, but keep per-instance identity (label_file, line index)
    instead of discarding it into an aggregate confusion matrix."""
    df = load_instances()
    df = df.reset_index(drop=True)
    df["line_idx"] = df.groupby("label_file").cumcount()

    train_df, test_df = grouped_split(df, SEED, TEST_FRACTION)
    x_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df["class_id"].to_numpy()
    x_test = test_df[FEATURE_COLS].to_numpy()

    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    result = test_df[["image_id", "label_file", "line_idx", "class_id"]].copy()
    result["coord_pred"] = y_pred
    result = result.rename(columns={"class_id": "true_class"})

    # Sanity check the join key really identifies unique rows before it's
    # relied on below - a silent duplicate would corrupt the join.
    assert result.duplicated(subset=["label_file", "line_idx"]).sum() == 0, (
        "label_file+line_idx is not unique in the coordinate baseline's own "
        "test set - join key assumption is wrong, do not proceed."
    )
    return result


def run_yolo_predictions() -> pd.DataFrame:
    """Run best.pt on every val-split image, IoU-match predictions to ground
    truth, and return one row per ground-truth box (unmatched boxes get
    yolo_pred=None) carrying the same (label_file, line_idx) identity as
    build_coord_predictions()."""
    from ultralytics import YOLO

    val_paths = (PREPARED_DIR / "val.txt").read_text().splitlines()
    if not val_paths:
        raise FileNotFoundError(
            f"{PREPARED_DIR / 'val.txt'} is empty or missing - run "
            f"prepare_yolo_dataset() from train_yolo.py first."
        )

    model = YOLO(str(BEST_WEIGHTS))
    rows = []

    for image_path in val_paths:
        p = Path(image_path)
        label_path = p.parents[1] / "labels" / f"{p.stem}.txt"
        gt_classes, gt_boxes = load_gt_boxes(label_path)
        if not gt_classes:
            continue

        result = model.predict(source=image_path, conf=EVAL_CONF, iou=EVAL_NMS_IOU,
                                device="cpu", verbose=False)[0]
        pred_classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
        pred_boxes = result.boxes.xyxyn.cpu().numpy().tolist()

        gt_arr = np.asarray(gt_boxes, dtype=float)
        pred_arr = np.asarray(pred_boxes, dtype=float)
        matched_gt_idx, matched_pred_idx = set(), set()
        if len(gt_arr) and len(pred_arr):
            iou = iou_xyxy(gt_arr, pred_arr)
            pairs = sorted(
                ((iou[i, j], i, j) for i in range(len(gt_arr)) for j in range(len(pred_arr))
                 if iou[i, j] >= MATCH_IOU_THRESHOLD),
                key=lambda t: t[0], reverse=True,
            )
            gt_to_pred = {}
            for _, i, j in pairs:
                if i in matched_gt_idx or j in matched_pred_idx:
                    continue
                matched_gt_idx.add(i)
                matched_pred_idx.add(j)
                gt_to_pred[i] = j
        else:
            gt_to_pred = {}

        for line_idx, true_cls in enumerate(gt_classes):
            pred_cls = pred_classes[gt_to_pred[line_idx]] if line_idx in gt_to_pred else None
            rows.append({
                "label_file": label_path.name,
                "line_idx": line_idx,
                "yolo_pred": pred_cls,
                "image_path": str(p),
                "gt_box_xyxyn": gt_boxes[line_idx],
            })

    return pd.DataFrame(rows)


def crop_case(image_path: str, box_xyxyn, out_path: Path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = box_xyxyn
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * CROP_PAD_FRAC
    x2 += bw * CROP_PAD_FRAC
    y1 -= bh * CROP_PAD_FRAC
    y2 += bh * CROP_PAD_FRAC
    px1, py1 = max(0, int(x1 * w)), max(0, int(y1 * h))
    px2, py2 = min(w, int(x2 * w)), min(h, int(y2 * h))
    img.crop((px1, py1, px2, py2)).save(out_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    print("Refitting coordinate-only baseline (seed 0, GBT) with instance identity...")
    coord_df = build_coord_predictions()
    print(f"  {len(coord_df)} test instances.")

    print("Running YOLOv8 (best.pt) on the val split for per-instance predictions...")
    yolo_df = run_yolo_predictions()
    print(f"  {len(yolo_df)} ground-truth instances scanned.")

    merged = coord_df.merge(yolo_df, on=["label_file", "line_idx"], how="inner")
    print(f"Joined on (label_file, line_idx): {len(merged)} instances present in both.")

    case_set = merged[
        (merged["yolo_pred"].notna())
        & (merged["yolo_pred"].astype("Int64") == merged["true_class"])
        & (merged["coord_pred"] != merged["true_class"])
    ].copy()
    print(f"Case set (YOLO correct, coordinate-only wrong): {len(case_set)} instances "
          f"out of {len(merged)} joined ({100 * len(case_set) / len(merged):.1f}%).")

    case_set["true_fdi"] = [FDI_CODES[c] for c in case_set["true_class"]]
    case_set["coord_pred_fdi"] = [FDI_CODES[c] for c in case_set["coord_pred"]]
    case_set["true_quadrant"] = quadrant_of(case_set["true_class"].to_numpy())
    case_set["true_tooth_type"] = tooth_type_of(case_set["true_class"].to_numpy())
    case_set["coord_error_type"] = [
        "mirror_quadrant" if is_mirror_quadrant_error(t, p)
        else "neighbor" if is_neighbor_error(t, p)
        else "other"
        for t, p in zip(case_set["true_class"], case_set["coord_pred"])
    ]

    case_set.to_csv(OUT_DIR / "case_study_yolo_correct_coord_wrong.csv", index=False)

    print("\nGround-truth FDI class distribution in the case set:")
    print(case_set["true_fdi"].value_counts().to_string())
    print("\nGround-truth tooth-type distribution:")
    print(case_set["true_tooth_type"].value_counts().to_string())
    print("\nCoordinate model's error-type breakdown on this case set:")
    print(case_set["coord_error_type"].value_counts().to_string())

    # Stratified sample across the dominant error-type categories found above,
    # not uniform random - so the manual review set actually covers whatever
    # categories turn out to matter rather than whatever happens to dominate.
    rng = random.Random(0)
    manifest_rows = []
    groups = list(case_set.groupby("coord_error_type"))
    groups.sort(key=lambda g: -len(g[1]))
    n_groups = len(groups)
    per_group = max(1, N_SAMPLE // n_groups)
    sampled_parts = []
    for name, grp in groups:
        take = min(len(grp), per_group)
        idx = rng.sample(list(grp.index), take)
        sampled_parts.append(grp.loc[idx])
    sampled = pd.concat(sampled_parts)
    if len(sampled) < N_SAMPLE:
        remaining = case_set.drop(sampled.index)
        extra = rng.sample(list(remaining.index), min(N_SAMPLE - len(sampled), len(remaining)))
        sampled = pd.concat([sampled, remaining.loc[extra]])

    for i, (_, row) in enumerate(sampled.iterrows()):
        crop_name = f"case_{i:02d}_true{row['true_fdi']}_coordpred{row['coord_pred_fdi']}_{row['coord_error_type']}.jpg"
        crop_case(row["image_path"], row["gt_box_xyxyn"], CROPS_DIR / crop_name)
        manifest_rows.append({
            "crop_file": crop_name,
            "source_image": row["image_path"],
            "label_file": row["label_file"],
            "line_idx": row["line_idx"],
            "true_fdi": row["true_fdi"],
            "yolo_pred_fdi": FDI_CODES[int(row["yolo_pred"])],
            "coord_pred_fdi": row["coord_pred_fdi"],
            "coord_error_type": row["coord_error_type"],
            "true_tooth_type": row["true_tooth_type"],
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(CROPS_DIR / "manifest.csv", index=False)
    print(f"\nSaved {len(manifest)} sampled crops + manifest.csv to {CROPS_DIR}")
    print(f"Full case set: {OUT_DIR / 'case_study_yolo_correct_coord_wrong.csv'}")


if __name__ == "__main__":
    main()
