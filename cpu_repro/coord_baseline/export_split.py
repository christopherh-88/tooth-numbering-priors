"""Persist the coordinate baseline's image-level train/test split to a file.

The main baseline (build_coord_baseline.py) computes its grouped split
on the fly with grouped_split(df, seed, TEST_FRACTION) - reproducible given
the same code/environment, but not written down anywhere. Any downstream
consumer that needs the "same split" (e.g. a YOLOv8 training run being
compared against this baseline) should read a persisted file instead of
re-deriving the split independently, so there is exactly one source of
truth and zero risk of silent drift if this code ever changes.

This script calls the coordinate baseline's own load_instances() and
grouped_split() - unmodified - for SEEDS[0] (seed 0), and writes the
resulting image_id -> split assignment to image_split_seed0.csv. Seed 0 is
also the seed build_coord_baseline.py uses for its saved confusion matrices
(CONFUSION_MATRIX_SEED), so it's the natural canonical split to compare a
single real-model training run against.
"""

import json
from pathlib import Path

from build_coord_baseline import SEEDS, TEST_FRACTION, grouped_split, load_instances

OUTPUT_DIR = Path(__file__).resolve().parent
SPLIT_SEED = SEEDS[0]


def main():
    df = load_instances()
    train_df, test_df = grouped_split(df, SPLIT_SEED, TEST_FRACTION)

    train_ids = sorted(train_df["image_id"].unique().tolist())
    test_ids = sorted(test_df["image_id"].unique().tolist())

    out_path = OUTPUT_DIR / "image_split_seed0.csv"
    with out_path.open("w") as f:
        f.write("image_id,split\n")
        for image_id in train_ids:
            f.write(f"{image_id},train\n")
        for image_id in test_ids:
            f.write(f"{image_id},test\n")

    meta = {
        "seed": SPLIT_SEED,
        "test_fraction": TEST_FRACTION,
        "n_train_images": len(train_ids),
        "n_test_images": len(test_ids),
        "n_train_instances": int(len(train_df)),
        "n_test_instances": int(len(test_df)),
        "source": "build_coord_baseline.grouped_split(load_instances(), seed=SEEDS[0], TEST_FRACTION)",
    }
    with (OUTPUT_DIR / "image_split_seed0.meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {out_path} ({len(train_ids)} train images, {len(test_ids)} test images)")
    print(f"Wrote {OUTPUT_DIR / 'image_split_seed0.meta.json'}")


if __name__ == "__main__":
    main()
