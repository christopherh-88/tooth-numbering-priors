"""Build joined_seed{N}.csv tables (seeds 5-9) for the three MPS-trained
detectors, in the same schema as eval_results/{multiseed,fasterrcnn_multiseed,
rtdetr_multiseed}/joined_seed{0-4}.csv, so cross_architecture_agreement.py
and other seed-0-4 analyses can be extended to seeds 5-9.

Seeds 0-4's tables pair each detector's own prediction with the
coordinate-only baseline's prediction on the SAME split (build_coord_baseline
only ever ran seeds 0-4 - see build_coord_baseline.SEEDS). This script trains
a fresh coordinate baseline for seeds 5-9 the same way
multiseed_analysis.build_coord_predictions() does (identical
HistGradientBoostingClassifier, identical grouped_split, identical
FEATURE_COLS), then joins it against each detector's MPS per_tooth.csv
(from per_tooth_predictions_mps.py / per_tooth_predictions_ultralytics_mps.py)
on (label_file, line_idx). per_tooth.csv's (image_id, gt_idx) is (label_file
minus ".txt", line_idx) by construction - both come from the same
train_yolo.load_gt_boxes() enumeration order - verified by an inner join
that must recover every row (assertion below).

Verification-before-use: for seed 0-4-equivalent sanity, this script's coord
predictions are NOT compared to build_coord_baseline.py's own per_seed_results
numbers (different classifier - HistGradientBoostingClassifier, matching
multiseed_analysis.py's convention, not build_coord_baseline.py's
logistic_regression/gradient_boosted_tree) - so coord_top1 here will differ
from BACKEND_COMPARISON.md's baseline numbers and from per_seed_results.csv;
that is expected, not a bug (see multiseed_analysis.py's own coord_pred,
which uses the same classifier and is not claimed equivalent to
build_coord_baseline.py's numbers either).

Writes eval_results/{multiseed,fasterrcnn_multiseed,rtdetr_multiseed}/joined_seed{5..9}.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "coord_baseline"))
from build_coord_baseline import FEATURE_COLS, TEST_FRACTION, grouped_split, load_instances  # noqa: E402

SEEDS = range(5, 10)
ARCHS = {
    "yolo": ("multiseed", "yolo_mps", "yolo_pred"),
    "fasterrcnn": ("fasterrcnn_multiseed", "fasterrcnn", "fasterrcnn_pred"),
    "rtdetr": ("rtdetr_multiseed", "rtdetr_mps", "rtdetr_pred"),
}


def build_coord_predictions(seed: int) -> pd.DataFrame:
    """Same recipe as multiseed_analysis.py's function of the same name."""
    df = load_instances()
    df = df.reset_index(drop=True)
    df["line_idx"] = df.groupby("label_file").cumcount()

    train_df, test_df = grouped_split(df, seed, TEST_FRACTION)
    x_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df["class_id"].to_numpy()
    x_test = test_df[FEATURE_COLS].to_numpy()

    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    result = test_df[["image_id", "label_file", "line_idx", "class_id"]].copy()
    result["coord_pred"] = y_pred
    result = result.rename(columns={"class_id": "true_class"})

    assert result.duplicated(subset=["label_file", "line_idx"]).sum() == 0, (
        f"seed {seed}: label_file+line_idx not unique - join key assumption broken."
    )
    return result


def coco_id_to_stem(seed: int) -> dict:
    """Faster R-CNN's per_tooth.csv (per_tooth_predictions_mps.py) keys rows by
    the COCO integer image id, not the filename stem the other two detectors
    use - map back via prepared_coco/seed{N}/instances_val.json's file_name."""
    val_json = HERE / "prepared_coco" / f"seed{seed}" / "instances_val.json"
    images = json.load(open(val_json))["images"]
    return {im["id"]: Path(im["file_name"]).stem for im in images}


def load_detector(arch: str, seed: int) -> pd.DataFrame:
    _, eval_subdir, pred_col = ARCHS[arch]
    path = HERE / "eval_results" / eval_subdir / f"seed{seed}" / "per_tooth.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run per_tooth_predictions_mps.py (fasterrcnn) or "
            f"per_tooth_predictions_ultralytics_mps.py (yolo/rtdetr) for seed {seed} first."
        )
    df = pd.read_csv(path)
    if arch == "fasterrcnn":
        id_map = coco_id_to_stem(seed)
        stems = df["image_id"].map(id_map)
        assert stems.notna().all(), f"seed {seed} fasterrcnn: some COCO image ids not in instances_val.json"
        df["label_file"] = stems + ".txt"
    else:
        df["label_file"] = df["image_id"] + ".txt"
    df = df.rename(columns={"gt_idx": "line_idx", "pred_class": pred_col})
    return df[["label_file", "line_idx", pred_col]]


def main():
    for seed in SEEDS:
        coord_df = build_coord_predictions(seed)
        for arch, (out_subdir, _, pred_col) in ARCHS.items():
            det_df = load_detector(arch, seed)
            merged = coord_df.merge(det_df, on=["label_file", "line_idx"], how="inner")
            assert len(merged) == len(coord_df), (
                f"seed {seed} {arch}: inner join dropped rows ({len(merged)} of {len(coord_df)}) - "
                f"per_tooth.csv and the coord baseline test set disagree on which teeth exist."
            )
            merged[f"{arch}_correct"] = (
                merged[pred_col].notna() & (merged[pred_col].astype("Int64") == merged["true_class"])
            )
            merged["coord_correct"] = merged["coord_pred"] == merged["true_class"]
            merged = merged[["image_id", "label_file", "line_idx", "true_class", "coord_pred",
                              pred_col, f"{arch}_correct", "coord_correct"]]

            out_dir = HERE / "eval_results" / out_subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"joined_seed{seed}.csv"
            merged.to_csv(dest, index=False)
            print(f"seed {seed} {arch}: n={len(merged)}  coord_top1={merged['coord_correct'].mean():.4f}  "
                  f"{arch}_top1={merged[f'{arch}_correct'].mean():.4f}  wrote {dest}")


if __name__ == "__main__":
    main()
