"""Build the uncropped UFBA-425 cross-validation dataset (Phase 2 CV).

The Roboflow export in Dataset/yolo_train_dataset holds 3 randomly cropped
copies of each X-ray in its train split (RESULTS.md Section 59). The
uncropped originals of all X-rays are in Dataset/bb_u_net_dataset
(panoramic_x_rays/*.jpg, 512x512, byte-identical to the FigShare
"segmentation_xrays" release), with one mask per tooth named by FDI number
(labels/<category folder>/<id>_<FDI>.ome.tiff). This script turns each
tooth mask into a box (the mask's full pixel extent) and assigns folds.

Outputs (small, committed):
  cpu_repro/cv/boxes.csv   one row per tooth: image_id, category, fdi,
                           class_id (FDI_CODES index), normalized
                           x_center, y_center, width, height
  cpu_repro/cv/folds.csv   (folds_seed<s>.csv for --split-seed s) one row per X-ray: image_id, category, n_teeth,
                           fold (0-4, the X-ray's test fold), and
                           inner_val_fold{0..4} (True if the X-ray is in the
                           validation set used for epoch selection when that
                           fold is the test fold)

With --write-yolo DIR it also writes YOLO-format label files and
per-fold train/val/test lists under DIR (not committed; regenerated on
each machine, as with yolo_training/prepared/).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "Dataset" / "bb_u_net_dataset"
OUT = Path(__file__).resolve().parent

FDI_CODES = [f"{q}{t}" for q in "1234" for t in "12345678"]  # same order as build_coord_baseline
N_FOLDS = 5
VAL_FRACTION = 0.15
SPLIT_SEED = 0


def mask_boxes():
    rows = []
    for p in sorted((SRC / "labels").glob("*/*.ome.tiff")):
        stem = p.name.replace(".ome.tiff", "")
        image_id, tooth = stem.rsplit("_", 1)
        if not tooth.isdigit():  # "<id>_background" masks
            continue
        assert tooth in FDI_CODES, p
        m = np.squeeze(np.array(Image.open(p)))
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            rows.append(dict(image_id=image_id, fdi=tooth, empty=True))
            continue
        h, w = m.shape
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        rows.append(dict(image_id=image_id, fdi=tooth, empty=False, mask_w=w, mask_h=h,
                         x_center=(x0 + x1) / 2 / w, y_center=(y0 + y1) / 2 / h,
                         width=(x1 - x0) / w, height=(y1 - y0) / h))
    return pd.DataFrame(rows)


def stratified_folds(ids, cats, n, rng):
    """Deal each category's shuffled X-rays round-robin into n folds,
    starting each category at a random fold, so fold sizes stay within one
    of each other and every fold gets its share of each category."""
    fold = {}
    for c in sorted(set(cats)):
        members = sorted(i for i, k in zip(ids, cats) if k == c)
        rng.shuffle(members)
        start = rng.integers(n)
        for j, i in enumerate(members):
            fold[i] = (start + j) % n
    return fold


def assign_folds(boxes, seed):
    """Test fold and inner validation sets per X-ray, from boxes.csv alone."""
    ids = sorted(boxes["image_id"].unique())
    cats = [i.split("-")[0] for i in ids]
    rng = np.random.default_rng(seed)
    test_fold = stratified_folds(ids, cats, N_FOLDS, rng)
    folds = pd.DataFrame({"image_id": ids, "category": cats})
    folds["n_teeth"] = folds["image_id"].map(boxes.groupby("image_id").size())
    folds["fold"] = folds["image_id"].map(test_fold)
    for f in range(N_FOLDS):
        train = folds[folds["fold"] != f]
        val = set()
        for c, grp in train.groupby("category"):
            members = sorted(grp["image_id"])
            rng.shuffle(members)
            val.update(members[:int(round(len(members) * VAL_FRACTION))])
        folds[f"inner_val_fold{f}"] = folds["image_id"].isin(val)
    return folds


def folds_path(seed):
    return OUT / ("folds.csv" if seed == SPLIT_SEED else f"folds_seed{seed}.csv")


def build():
    boxes = mask_boxes()
    n_empty = int(boxes["empty"].sum())
    boxes = boxes[~boxes["empty"]].drop(columns="empty")
    images = sorted(p.stem for p in (SRC / "panoramic_x_rays").glob("*.jpg"))
    sizes = {i: Image.open(SRC / "panoramic_x_rays" / f"{i}.jpg").size for i in images}

    # Every box comes from a mask the same size as its X-ray.
    for r in boxes.drop_duplicates("image_id").itertuples():
        assert sizes[r.image_id] == (r.mask_w, r.mask_h), r.image_id
    boxes = boxes.drop(columns=["mask_w", "mask_h"])
    assert not boxes.duplicated(["image_id", "fdi"]).any()

    no_masks = sorted(set(images) - set(boxes["image_id"]))
    no_image = sorted(set(boxes["image_id"]) - set(images))
    assert not no_image, no_image
    boxes["class_id"] = boxes["fdi"].map({c: i for i, c in enumerate(FDI_CODES)})
    boxes["category"] = boxes["image_id"].str.split("-").str[0]
    boxes = boxes[["image_id", "category", "fdi", "class_id",
                   "x_center", "y_center", "width", "height"]]
    boxes = boxes.sort_values(["image_id", "class_id"]).reset_index(drop=True)

    folds = assign_folds(boxes, SPLIT_SEED)

    boxes.to_csv(OUT / "boxes.csv", index=False, float_format="%.6f")
    folds.to_csv(OUT / "folds.csv", index=False)
    print(f"{len(images)} X-rays on disk; {len(folds)} with at least one tooth mask; "
          f"no masks: {no_masks}; empty masks skipped: {n_empty}")
    print(f"{len(boxes)} teeth")
    print(folds.groupby("fold").agg(xrays=("image_id", "size"), teeth=("n_teeth", "sum")))
    for f in range(N_FOLDS):
        tr = folds[folds["fold"] != f]
        print(f"fold {f}: train {int((~tr[f'inner_val_fold{f}']).sum())}, "
              f"val {int(tr[f'inner_val_fold{f}'].sum())}, test {int((folds['fold'] == f).sum())}")
    return boxes, folds


def write_yolo(out_dir: Path, boxes, folds):
    """images/ symlinks to the X-rays, labels/ in YOLO format, and
    fold{f}_{train,val,test}.txt lists of absolute image paths."""
    img_dir, lab_dir = out_dir / "images", out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for image_id, grp in boxes.groupby("image_id"):
        link = img_dir / f"{image_id}.jpg"
        if not link.exists():
            link.symlink_to(SRC / "panoramic_x_rays" / f"{image_id}.jpg")
        lines = [f"{r.class_id} {r.x_center:.6f} {r.y_center:.6f} {r.width:.6f} {r.height:.6f}"
                 for r in grp.itertuples()]
        (lab_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n")
    for f in range(N_FOLDS):
        val_col = f"inner_val_fold{f}"
        parts = {"test": folds["fold"] == f,
                 "val": (folds["fold"] != f) & folds[val_col],
                 "train": (folds["fold"] != f) & ~folds[val_col]}
        for name, mask in parts.items():
            paths = [str(img_dir / f"{i}.jpg") for i in folds.loc[mask, "image_id"]]
            (out_dir / f"fold{f}_{name}.txt").write_text("\n".join(paths) + "\n")
    print(f"wrote YOLO labels and fold lists to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-yolo", type=Path, default=None)
    ap.add_argument("--split-seed", type=int, default=None,
                    help="only write folds_seed<s>.csv from the committed boxes.csv (no masks read)")
    args = ap.parse_args()
    if args.split_seed is not None:
        f = assign_folds(pd.read_csv(OUT / "boxes.csv"), args.split_seed)
        f.to_csv(folds_path(args.split_seed), index=False)
        print(f.groupby("fold").agg(xrays=("image_id", "size"), teeth=("n_teeth", "sum")))
        raise SystemExit
    b, f = build()
    if args.write_yolo:
        write_yolo(args.write_yolo, b, f)
