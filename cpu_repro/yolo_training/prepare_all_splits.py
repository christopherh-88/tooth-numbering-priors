"""Regenerate the gitignored per-seed data folders without training anything.

Builds, for every committed split (coord_baseline/image_split_seed{N}.csv,
seeds 0-14):
- prepared/[seed{N}/]train.txt, val.txt, data.yaml (train_yolo.prepare_yolo_dataset)
- prepared_coco/[seed{N}/]instances_{train,val}.json (build_coco_dataset.convert)

Seed 0 uses the unsuffixed folders, like the training scripts. The analysis
scripts that map predictions back to label files (missed_tooth_analysis.py,
build_joined_mps_seeds.py, build_joined_cuda_10_14.py) need these folders, so
run this once after a fresh clone. Both builders read only the committed
splits (coord_baseline/image_split_seed{N}.csv) and Dataset/yolo_train_dataset.
prepared/*.txt hold absolute image paths, so they are machine-specific.

Usage: python prepare_all_splits.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_coco_dataset as bcd  # noqa: E402
import train_yolo as ty  # noqa: E402


def seed_dir(name, seed):
    return HERE / name / (f"seed{seed}" if seed != 0 else "")


def main():
    split_dir = ty.REPO_ROOT / "cpu_repro" / "coord_baseline"
    seeds = sorted(int(f.stem.removeprefix("image_split_seed")) for f in split_dir.glob("image_split_seed*.csv"))
    for seed in seeds:
        split_file = split_dir / f"image_split_seed{seed}.csv"

        ty.SEED = seed
        ty.SPLIT_FILE = split_file
        ty.PREPARED_DIR = seed_dir("prepared", seed)
        ty.prepare_yolo_dataset()

        bcd.SEED = seed
        bcd.SPLIT_FILE = split_file
        bcd.OUT_DIR = seed_dir("prepared_coco", seed)
        bcd.convert()
        print(f"seed {seed}: {ty.PREPARED_DIR.relative_to(HERE)}, {bcd.OUT_DIR.relative_to(HERE)}")


if __name__ == "__main__":
    main()
