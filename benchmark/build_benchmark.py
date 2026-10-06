"""Build benchmark/teeth.csv from the confidence-first per-tooth file
(RESULTS.md Section 65) and the two CV splits.

    python benchmark/build_benchmark.py \
        --per-tooth cpu_repro/cv/results/kaggle_confmatch/confmatch/score/per_tooth_predictions.csv

One row per labeled tooth in UFBA-425 (uncropped): its box, the test fold
under split seeds 0 and 1, whether an arch neighbor has no label (a gap),
the out-of-fold position-only answer under each split, the three reference detectors'
answers (seed 0 split), and whether the tooth is a joint failure.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cpu_repro/cv"))
from gap_check import NEIGHBORS  # noqa: E402

DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tooth", type=Path, required=True)
    a = ap.parse_args()
    t = pd.read_csv(a.per_tooth, dtype={"fdi": str})
    s1 = pd.read_csv(ROOT / "cpu_repro/cv/folds_seed1.csv")[["image_id", "fold"]]
    t = t.rename(columns={"fold": "fold_seed0"}).merge(s1.rename(columns={"fold": "fold_seed1"}), on="image_id")
    p1 = pd.read_csv(ROOT / "cpu_repro/cv/results/kaggle_seed1/seed1_2det/per_tooth_predictions.csv", dtype={"fdi": str})
    t = t.merge(p1[["image_id", "fdi", "coord_pred"]].rename(columns={"coord_pred": "position_only_pred_seed1"}),
                on=["image_id", "fdi"], how="left", validate="one_to_one")
    assert t["position_only_pred_seed1"].notna().all()
    present = t.groupby("image_id")["fdi"].agg(set).to_dict()
    t["next_to_gap"] = [any(n not in present[i] for n in NEIGHBORS[f]) for i, f in zip(t["image_id"], t["fdi"])]
    joint = t["coord_pred"] != t["class_id"]
    for d in DETS:
        joint &= (t[f"{d}_pred"] != t["class_id"]) & t[f"{d}_pred"].notna()
    joint &= (t["yolov8x_pred"] == t["rtdetr_l_pred"]) & (t["yolov8x_pred"] == t["fasterrcnn_pred"])
    t["joint_failure"] = joint
    t["shared_wrong_class"] = t["yolov8x_pred"].where(joint).astype("Int64")
    out = t[["image_id", "category", "fold_seed0", "fold_seed1", "fdi", "class_id", "x_center", "y_center",
             "width", "height", "next_to_gap", "coord_pred", "position_only_pred_seed1"] + [f"{d}_pred" for d in DETS]
            + ["joint_failure", "shared_wrong_class"]].rename(columns={"coord_pred": "position_only_pred_seed0"})
    for c in [f"{d}_pred" for d in DETS]:
        out[c] = out[c].astype("Int64")
    out.to_csv(ROOT / "benchmark/teeth.csv", index=False)
    print(len(out), "teeth,", out["image_id"].nunique(), "X-rays,", int(joint.sum()), "joint failures,",
          int(out["next_to_gap"].sum()), "next to a gap")


if __name__ == "__main__":
    main()
