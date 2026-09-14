"""YOLO-txt -> COCO JSON conversion for the Faster R-CNN scoping run
(RESULTS.md Section 42). Produces instances_train.json / instances_val.json
under cpu_repro/yolo_training/prepared_coco/, reusing the exact same
image_split_seed0.csv split (via base_image_id(), copied identically from
train_yolo.py - see its own comment on why it's duplicated rather than
imported) that every other detector run in this project uses, so Faster
R-CNN is trained/evaluated on the same held-out images as YOLOv8x and
RT-DETR-l, not a re-derived split that could silently leak.

Two lossy-conversion risks handled explicitly here, per Section 42:
  1. Background-label shift: torchvision's FasterRCNN reserves class 0
     for an implicit background - this project's FDI classes are 0-31
     (background-free), so every category_id here is written as
     class_id + 1 (COCO categories 1-32), with a positive assertion that
     no annotation is ever written with category_id 0.
  2. Per-image (width, height) is read from the actual file via
     PIL.Image.size, not assumed to be 640x640 from the Roboflow
     README's stated preprocessing - that note describes what Roboflow
     did before export, not a verified guarantee of every file on disk.

Usage:
    python build_coco_dataset.py           # convert, then run built-in checks
    python build_coco_dataset.py --verify  # only re-run the round-trip checks
    on an existing instances_*.json (no reconversion)
"""

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coord_baseline"))
from build_coord_baseline import FDI_CODES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLO_SOURCE_ROOT = REPO_ROOT / "Dataset" / "yolo_train_dataset"

# Seed-parameterized the same way train_rtdetr.py's SEED/SPLIT_FILE/PREPARED_DIR
# are - SEED=0 here, train_fasterrcnn_seed{1,2,3,4}.py override these three
# module attributes before calling convert()/verify(), same thin-wrapper
# pattern as train_rtdetr_seed{1,2,3,4}.py. Seed 0's OUT_DIR is deliberately
# left as the pre-existing unsuffixed prepared_coco/ (matching PREPARED_DIR's
# own f"seed{SEED}" if SEED != 0 else "" convention) so the already-trained
# seed-0 run's instances_*.json is not invalidated/moved by this change.
SEED = 0
SPLIT_FILE = REPO_ROOT / "cpu_repro" / "coord_baseline" / f"image_split_seed{SEED}.csv"
OUT_DIR = Path(__file__).resolve().parent / "prepared_coco" / (f"seed{SEED}" if SEED != 0 else "")

AUG_SUFFIX_RE = re.compile(r"_jpg\.rf\.[0-9a-f]+$")


def base_image_id(label_stem: str) -> str:
    """Identical to train_yolo.py's base_image_id() / build_coord_baseline's
    - duplicated for the same standalone-runnability reason train_yolo.py
    gives, not re-derived differently."""
    return AUG_SUFFIX_RE.sub("", label_stem)


def load_split() -> dict:
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"{SPLIT_FILE} not found. This script reads the coordinate baseline's "
            f"persisted split - it does not regenerate it. Run "
            f"cpu_repro/coord_baseline/export_split.py once first."
        )
    import pandas as pd
    split_df = pd.read_csv(SPLIT_FILE)
    return dict(zip(split_df["image_id"], split_df["split"]))


def parse_yolo_label(label_path: Path):
    """Returns a list of (class_id, cx, cy, w, h), all normalized [0,1],
    directly from the txt file. Empty file -> empty list (valid - some
    images may have zero annotated teeth)."""
    boxes = []
    text = label_path.read_text().strip()
    if not text:
        return boxes
    for line in text.splitlines():
        parts = line.split()
        class_id = int(parts[0])
        cx, cy, w, h = (float(x) for x in parts[1:5])
        boxes.append((class_id, cx, cy, w, h))
    return boxes


def yolo_box_to_coco_xywh(cx, cy, w, h, img_w, img_h):
    """Denormalize + convert center-xywh to COCO's top-left-xywh, in
    absolute pixels."""
    abs_w = w * img_w
    abs_h = h * img_h
    x_min = (cx * img_w) - abs_w / 2
    y_min = (cy * img_h) - abs_h / 2
    return x_min, y_min, abs_w, abs_h


def build_split(split_name: str, label_files, split_map: dict) -> dict:
    images, annotations = [], []
    ann_id = 1
    image_id = 1
    n_empty = 0

    for lf in label_files:
        bid = base_image_id(lf.stem)
        raw_split = split_map.get(bid)
        if raw_split is None:
            continue
        # image_split_seed0.csv uses "train"/"test" labels, not "train"/"val" -
        # train_yolo.py's prepare_yolo_dataset() treats anything not "train" as
        # its val set (line: `(train_paths if split == "train" else val_paths)`).
        # Mirrored exactly here so this script's "val" split is the identical
        # set of images YOLOv8x/RT-DETR-l were evaluated on, not a different
        # 0-image set from a naive string-equality bug.
        resolved_split = "train" if raw_split == "train" else "val"
        if resolved_split != split_name:
            continue

        image_path = lf.parents[1] / "images" / f"{lf.stem}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Expected image not found: {image_path}")

        with Image.open(image_path) as im:
            img_w, img_h = im.size  # actual per-file dims, not assumed 640x640

        images.append({
            "id": image_id,
            "file_name": str(image_path),
            "width": img_w,
            "height": img_h,
        })

        boxes = parse_yolo_label(lf)
        if not boxes:
            n_empty += 1
        for class_id, cx, cy, w, h in boxes:
            x_min, y_min, abs_w, abs_h = yolo_box_to_coco_xywh(cx, cy, w, h, img_w, img_h)
            category_id = class_id + 1  # background-label shift: FDI 0-31 -> COCO 1-32
            assert category_id != 0, "background-shift invariant violated"
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x_min, y_min, abs_w, abs_h],
                "area": abs_w * abs_h,
                "iscrowd": 0,
            })
            ann_id += 1
        image_id += 1

    categories = [{"id": i + 1, "name": code} for i, code in enumerate(FDI_CODES)]
    assert all(c["id"] != 0 for c in categories), "background-shift invariant violated in categories"

    print(f"  {split_name}: {len(images)} images, {len(annotations)} boxes, "
          f"{n_empty} images with zero boxes")
    return {"images": images, "annotations": annotations, "categories": categories}


def convert():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_map = load_split()
    label_files = sorted(YOLO_SOURCE_ROOT.glob("*/labels/*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files found under {YOLO_SOURCE_ROOT}")

    unmapped = sum(1 for lf in label_files if base_image_id(lf.stem) not in split_map)
    if unmapped:
        print(f"WARNING: {unmapped} label files had an image_id not present in the "
              f"split file and were skipped (same convention as train_yolo.py).")

    for split_name, out_name in [("train", "instances_train.json"), ("val", "instances_val.json")]:
        print(f"Converting {split_name}...")
        coco = build_split(split_name, label_files, split_map)
        out_path = OUT_DIR / out_name
        out_path.write_text(json.dumps(coco))
        print(f"  wrote {out_path}")


def verify(n_checks: int = 5):
    """Round-trip check: re-read instances_train.json, recompute the
    original normalized YOLO cx/cy/w/h from each COCO bbox + the image's
    recorded width/height, and compare against the source label file's
    actual line for the same image - not a self-consistency check against
    the conversion script's own math, but against the untouched source
    file on disk."""
    train_path = OUT_DIR / "instances_train.json"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found - run convert() first.")

    coco = json.loads(train_path.read_text())
    images_by_id = {im["id"]: im for im in coco["images"]}

    checked = 0
    failures = []
    for ann in coco["annotations"]:
        if checked >= n_checks:
            break
        img = images_by_id[ann["image_id"]]
        img_w, img_h = img["width"], img["height"]
        x_min, y_min, w, h = ann["bbox"]

        # recompute normalized center-xywh from the COCO bbox
        recon_cx = (x_min + w / 2) / img_w
        recon_cy = (y_min + h / 2) / img_h
        recon_w = w / img_w
        recon_h = h / img_h
        recon_class = ann["category_id"] - 1  # undo the background shift

        # find the matching source line by class + closest center (label
        # files carry no per-line id, so match on rounded geometry)
        label_path = Path(img["file_name"]).parents[1] / "labels" / (Path(img["file_name"]).stem + ".txt")
        source_boxes = parse_yolo_label(label_path)
        match = None
        for class_id, cx, cy, w0, h0 in source_boxes:
            if (class_id == recon_class
                    and abs(cx - recon_cx) < 1e-6 and abs(cy - recon_cy) < 1e-6
                    and abs(w0 - recon_w) < 1e-6 and abs(h0 - recon_h) < 1e-6):
                match = (class_id, cx, cy, w0, h0)
                break

        status = "PASS" if match else "FAIL"
        if not match:
            failures.append((img["file_name"], ann))
        print(f"  [{status}] image={Path(img['file_name']).name} "
              f"category_id={ann['category_id']} (FDI {FDI_CODES[recon_class]}) "
              f"bbox={[round(v, 2) for v in ann['bbox']]} "
              f"-> recon norm=({recon_cx:.6f},{recon_cy:.6f},{recon_w:.6f},{recon_h:.6f})")
        checked += 1

    if failures:
        raise AssertionError(f"{len(failures)}/{checked} spot-checked boxes did NOT "
                              f"round-trip back to a source YOLO-txt line - DO NOT "
                              f"trust this conversion on the full dataset.")
    print(f"\nAll {checked} spot-checked boxes round-tripped exactly to their source "
          f"YOLO-txt line (within 1e-6). category_id never 0 in either split's "
          f"categories or annotations (checked at write time via assert).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="skip conversion, only run round-trip checks")
    parser.add_argument("--n-checks", type=int, default=5)
    args = parser.parse_args()

    if not args.verify:
        convert()
    print()
    verify(args.n_checks)
