"""CPU-only, tiny-subset reproduction of notebooks/Unet/unet_training.ipynb.

Purpose: prove the environment/data pipeline works end-to-end, NOT to reach
the README's reported Dice numbers (73.29 for U-Net incisors etc.) - that
requires the full 425-image dataset, many more epochs, and (per the
original notebooks) a GPU.

Differences from the original notebook, and why:

1. Bounding-box prior channels (the 32 extra input channels beyond the RGB
   image) are zero-filled here instead of coming from a trained YOLOv8
   checkpoint. In `get_model()` below (copied verbatim from the notebook),
   the plain U-Net slices `inputs1` (the bbox-prior channels) off the input
   tensor but NEVER uses it in the forward pass - only `inputs0` (the image)
   feeds the encoder. So for this vanilla U-Net, the bbox channels are inert
   padding. (Contrast with yolov8+unet_training.ipynb, where bbox priors ARE
   multiplied into the skip connections - that notebook would need real
   YOLO inference first.) Because they're inert here, zero-filling them
   changes nothing about what the model computes, and lets this script skip
   training/running YOLO entirely.
2. Ground-truth masks are assembled directly from the per-tooth-channel
   export tiffs already in Dataset/bb_u_net_dataset/labels/*/ (each file is
   named "<image_stem>_<FDI code>.ome.tiff"), replicating the channel-index
   <-> FDI-code mapping from notebooks/Data_gen/2ddatagen.ipynb, instead of
   requiring you to run that notebook first to produce combined per-image
   tiffs (the repo does not ship those combined tiffs).
3. IMG_SIZE defaults to 256, not the notebook's 512, purely for CPU epoch
   time. The bottleneck of this U-Net is 1024 filters at 1/16 resolution;
   at 512x512 that is expensive on CPU. Bump IMG_SIZE back to 512 once you
   have a GPU to match the paper's setting exactly.

Everything under CONFIG below is meant to be the only thing you touch to
scale this up (more images, more epochs, bigger image size) once you have
GPU quota again.
"""

import os
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import Precision, Recall

# --------------------------------------------------------------------------
# CONFIG - the only section you should need to edit to scale this up later.
# --------------------------------------------------------------------------
SUBSET_SIZE = 24          # number of images to use in total (train+val)
EPOCHS = 3
BATCH_SIZE = 2
IMG_SIZE = 256             # notebook default is 512; see module docstring
VAL_FRACTION = 0.2
SEED = 42

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO_ROOT / "Dataset" / "bb_u_net_dataset" / "panoramic_x_rays"
LABELS_ROOT = REPO_ROOT / "Dataset" / "bb_u_net_dataset" / "labels"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
# --------------------------------------------------------------------------

# Fixed FDI code -> channel index mapping. This is the *entire* numbering
# scheme: channel/class index j always means FDI_CHANNELS[j], both here and
# in the YOLOv8 dataset (Dataset/yolo_train_dataset/data.yaml names list).
# There is no spatial/geometric assignment anywhere in this repo - see the
# Task 2 writeup for detail.
FDI_CHANNELS = [
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
]

# Filename-prefix ("cateN") -> label export folder. Hardcoded because the
# folder names are inconsistently spaced/dashed ("cate1 - export" vs
# "cate4-export" vs "cat 10 - export") and there are only 10 of them.
LABEL_DIRS = {
    1: "cate1 - export",
    2: "cate2 - export",
    3: "cate3 - export",
    4: "cate4-export",
    5: "cate5 - export",
    6: "cate6 - export",
    7: "cate7-export",
    8: "cate8 - export",
    9: "cate9 - export",
    10: "cat 10 - export",
}

CATE_RE = re.compile(r"^cate(\d+)-")


def list_image_paths():
    paths = sorted(IMAGES_DIR.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(
            f"No images found under {IMAGES_DIR}. See README.md for the "
            f"expected dataset layout."
        )
    return paths


def load_mask(image_path: Path, size: int) -> np.ndarray:
    """Build a (size, size, 32) FDI-indexed binary mask from the per-tooth
    export tiffs, mirroring 2ddatagen.ipynb's channel assignment."""
    import tifffile as tiff

    m = CATE_RE.match(image_path.stem)
    if not m:
        raise ValueError(f"Unexpected filename, can't determine category: {image_path.name}")
    category = int(m.group(1))
    label_dir = LABELS_ROOT / LABEL_DIRS[category]

    mask = np.zeros((32, size, size), dtype=np.uint8)
    for j, code in enumerate(FDI_CHANNELS):
        tooth_path = label_dir / f"{image_path.stem}_{code}.ome.tiff"
        if not tooth_path.exists():
            continue  # tooth absent/not annotated in this image -> stays zero
        channel = tiff.imread(tooth_path)
        channel = (channel > 0).astype(np.uint8)
        channel = cv2.resize(channel, (size, size), interpolation=cv2.INTER_NEAREST)
        mask[j] = channel
    return np.transpose(mask, axes=[1, 2, 0])  # (H, W, 32)


def load_image(image_path: Path, size: int) -> np.ndarray:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-7)
    return arr


def build_dataset(image_paths, size):
    images, masks = [], []
    for p in image_paths:
        images.append(load_image(p, size))
        masks.append(load_mask(p, size))
    images = np.stack(images).astype(np.float32)   # (N, H, W, 3)
    masks = np.stack(masks).astype(np.float32)      # (N, H, W, 32)
    bbox_priors = np.zeros_like(masks)               # inert - see docstring
    model_input = np.concatenate([bbox_priors, images], axis=-1)  # (N, H, W, 35)
    return model_input, masks


# --------------------------------------------------------------------------
# Model, loss, metrics - copied from notebooks/Unet/unet_training.ipynb
# (get_model / dice_loss_with_l2_regularization / dice_coef) unchanged.
# --------------------------------------------------------------------------
DROP_RATE = 0.12


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (35,))

    inputs0 = inputs[:, :, :, 32:]   # image channels (used)
    # inputs1 (bbox-prior channels, inputs[:, :, :, :32]) is intentionally
    # unused below - this matches the original notebook's plain U-Net,
    # which never references it after slicing.

    skip_connections = []
    x = keras.layers.Conv2D(64, 3, strides=1, padding="same")(inputs0)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SpatialDropout2D(DROP_RATE)(x)

    x = keras.layers.Conv2D(64, 3, strides=1, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SpatialDropout2D(DROP_RATE)(x)
    skip_connections.append(x)

    for filters in [128, 256, 512, 1024]:
        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SpatialDropout2D(DROP_RATE)(x)

        x = keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SpatialDropout2D(DROP_RATE)(x)
        skip_connections.append(x)

    skip_connections.pop()
    for filters in [512, 256, 128]:
        x = keras.layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SpatialDropout2D(DROP_RATE)(x)
        skip_connection = skip_connections.pop()
        x = keras.layers.concatenate([x, skip_connection])

        x = keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SpatialDropout2D(DROP_RATE)(x)

        x = keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SpatialDropout2D(DROP_RATE)(x)

    filters = 64
    x = keras.layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SpatialDropout2D(DROP_RATE)(x)
    skip_connection = skip_connections.pop()
    x = keras.layers.concatenate([x, skip_connection])

    x = keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SpatialDropout2D(DROP_RATE)(x)

    x = keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)

    outputs = keras.layers.Conv2D(num_classes, 1, activation="softmax", padding="same")(x)
    return keras.Model(inputs, outputs)


def dice_loss_with_l2_regularization(target, predicted, epsilon=1e-7, l2_weight=0.1):
    intersection = tf.reduce_sum(predicted * target, axis=[1, 2])
    union = tf.reduce_sum(tf.square(predicted), axis=[1, 2]) + tf.reduce_sum(tf.square(target), axis=[1, 2])
    dice = (2 * intersection + epsilon) / (union + epsilon)
    mean_dice_loss = tf.reduce_mean(dice)
    l2_norm = tf.reduce_sum(tf.square(predicted - target), axis=[1, 2])
    l2_regularization = l2_weight * tf.reduce_mean(l2_norm)
    return mean_dice_loss + l2_regularization


def dice_coef(target, predicted, epsilon=1e-7):
    predicted = tf.where(predicted < 0.51, 0.0, 1.0)
    intersection = tf.reduce_sum(predicted * target, axis=[1, 2])
    union = tf.reduce_sum(tf.square(predicted), axis=[1, 2]) + tf.reduce_sum(tf.square(target), axis=[1, 2])
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return tf.reduce_mean(dice)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_paths = list_image_paths()
    random.shuffle(all_paths)
    subset = all_paths[:SUBSET_SIZE]
    if len(subset) < SUBSET_SIZE:
        print(f"WARNING: only found {len(subset)} images, wanted {SUBSET_SIZE}")

    print(f"Building tensors for {len(subset)} images at {IMG_SIZE}x{IMG_SIZE} ...")
    t0 = time.time()
    x, y = build_dataset(subset, IMG_SIZE)
    print(f"  done in {time.time() - t0:.1f}s -> x{x.shape} y{y.shape}")

    n_val = max(1, int(len(subset) * VAL_FRACTION))
    x_val, y_val = x[:n_val], y[:n_val]
    x_train, y_train = x[n_val:], y[n_val:]
    print(f"train={len(x_train)} val={len(x_val)}")

    with tf.device("/CPU:0"):
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE)
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)

        model = get_model(img_size=(IMG_SIZE, IMG_SIZE), num_classes=32)
        model.compile(
            optimizer=keras.optimizers.Adam(0.0003),
            loss=dice_loss_with_l2_regularization,
            metrics=[Precision(), Recall(), dice_coef],
        )

        print(model.summary())
        history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

    ckpt_path = OUTPUT_DIR / "unet_cpu_smoke_test.keras"
    model.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    print("Final metrics:", {k: v[-1] for k, v in history.history.items()})


if __name__ == "__main__":
    main()
