"""DENTEX boxes and folds for training within DENTEX (cpu_repro/cv/DENTEX_INDOMAIN_RULES.md).

    python dentex_prepare.py

Writes dentex_boxes.csv (one row per labeled tooth, as boxes.csv) and
dentex_folds.csv (test fold as Section 85, plus inner_val_fold{f}: 15% of
each fold's training X-rays, seed 0, for epoch selection), and checks the
test folds against the Section 85 per-tooth table when it is present.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_uncropped_cv as prep  # noqa: E402
from dentex_external import load_teeth  # noqa: E402

LABELS = HERE.parent / "anomaly_scan/dentex_raw/train_quadrant_enumeration.json"


def main():
    t = load_teeth(LABELS)
    ids = sorted(t["image_id"].unique())
    perm = np.random.default_rng(0).permutation(len(ids))  # same assignment as dentex_external.py
    fold_of = {ids[k]: int(r % 5) for r, k in enumerate(perm)}
    folds = pd.DataFrame({"image_id": ids, "fold": [fold_of[i] for i in ids]})
    folds["n_teeth"] = folds["image_id"].map(t.groupby("image_id").size())
    rng = np.random.default_rng(0)
    for f in range(prep.N_FOLDS):
        train = sorted(folds.loc[folds["fold"] != f, "image_id"])
        rng.shuffle(train)
        folds[f"inner_val_fold{f}"] = folds["image_id"].isin(train[:int(round(len(train) * prep.VAL_FRACTION))])
    boxes = t[["image_id", "fdi", "class_id", "x_center", "y_center", "width", "height"]]
    boxes = boxes.sort_values(["image_id", "class_id"]).reset_index(drop=True)

    s85 = HERE / "results/kaggle_dentex/dentex_teeth.csv"
    if s85.exists():
        old = pd.read_csv(s85).drop_duplicates("image_id").set_index("image_id")["fold"]
        assert (folds.set_index("image_id")["fold"] == old.loc[folds["image_id"]].to_numpy()).all()
        print("test folds match Section 85")
    boxes.to_csv(HERE / "dentex_boxes.csv", index=False, float_format="%.6f")
    folds.to_csv(HERE / "dentex_folds.csv", index=False)
    print(f"{len(ids)} X-rays, {len(boxes)} teeth")
    for f in range(prep.N_FOLDS):
        tr = folds[folds["fold"] != f]
        print(f"fold {f}: train {int((~tr[f'inner_val_fold{f}']).sum())}, "
              f"val {int(tr[f'inner_val_fold{f}'].sum())}, test {int((folds['fold'] == f).sum())}")


if __name__ == "__main__":
    main()
