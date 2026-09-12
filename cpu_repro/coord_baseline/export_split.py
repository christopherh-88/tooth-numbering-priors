"""Persist the coordinate baseline's image-level train/test split to a file.

The main baseline (build_coord_baseline.py) computes its grouped split
on the fly with grouped_split(df, seed, TEST_FRACTION) - reproducible given
the same code/environment, but not written down anywhere. Any downstream
consumer that needs the "same split" (e.g. a YOLOv8 training run being
compared against this baseline) should read a persisted file instead of
re-deriving the split independently, so there is exactly one source of
truth and zero risk of silent drift if this code ever changes.

This script calls the coordinate baseline's own load_instances() and
grouped_split() - unmodified - for a given seed, and writes the resulting
image_id -> split assignment to image_split_seed<N>.csv. Seed 0 is also
the seed build_coord_baseline.py uses for its saved confusion matrices
(CONFUSION_MATRIX_SEED), so it was the first (and, until the multi-seed
YOLO replication scoped in RESULTS.md Section 32, only) split persisted
this way.

Usage: `python export_split.py [seed]` - defaults to seed 0 if omitted,
for backward compatibility with every existing caller
(train_yolo.py/train_yolo_zerojitter.py's own SPLIT_FILE default).
"""

import json
import sys
from pathlib import Path

from build_coord_baseline import SEEDS, TEST_FRACTION, grouped_split, load_instances

OUTPUT_DIR = Path(__file__).resolve().parent


def export_split(seed: int):
    df = load_instances()
    train_df, test_df = grouped_split(df, seed, TEST_FRACTION)

    train_ids = sorted(train_df["image_id"].unique().tolist())
    test_ids = sorted(test_df["image_id"].unique().tolist())

    out_path = OUTPUT_DIR / f"image_split_seed{seed}.csv"
    with out_path.open("w") as f:
        f.write("image_id,split\n")
        for image_id in train_ids:
            f.write(f"{image_id},train\n")
        for image_id in test_ids:
            f.write(f"{image_id},test\n")

    meta = {
        "seed": seed,
        "test_fraction": TEST_FRACTION,
        "n_train_images": len(train_ids),
        "n_test_images": len(test_ids),
        "n_train_instances": int(len(train_df)),
        "n_test_instances": int(len(test_df)),
        "source": f"build_coord_baseline.grouped_split(load_instances(), seed={seed}, TEST_FRACTION)",
    }
    meta_path = OUTPUT_DIR / f"image_split_seed{seed}.meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {out_path} ({len(train_ids)} train images, {len(test_ids)} test images)")
    print(f"Wrote {meta_path}")


def main():
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        if seed not in SEEDS:
            raise ValueError(f"seed {seed} not in this project's SEEDS={SEEDS} - "
                              f"every other 5-seed result in this repo uses these same seeds, "
                              f"so an out-of-set seed would not be comparable to them.")
        export_split(seed)
    else:
        export_split(SEEDS[0])


if __name__ == "__main__":
    main()
