"""Check the horizontal-flip augmentation in the four U-Net notebooks on a
real UFBA-425 image.

The notebooks feed the U-Net two per-tooth arrays: the target mask
(input_mask) and the box map built from YOLO detections (input_bb). Both
hold one channel per FDI tooth (channel k is FDI_CODES[k]). The flip in
augment() has to swap those channels along with the pixels, or every
flipped sample carries left/right-swapped tooth labels.

The first fix (2026-09-08, notebooks/yolov8+unet/yolov8+unet_training.ipynb
only) swapped the input_bb channels but not the input_mask channels, and
the other three notebooks still had the original flip. RESULTS.md Section
58 has the details.

What this script does, for each notebook:
1. Pulls the augment() cell out of the notebook and runs it as-is.
2. Builds input_mask the way 2ddatagen.ipynb does (32 FDI channels plus one
   empty channel, from the per-tooth .ome.tiff files) and input_bb the way
   the notebooks' cell 13 does (a filled box per tooth), then puts both in
   (H, W, C) layout like resize_img().
3. Forces the flip branch and checks, for every tooth present:
   - quadrant 1 teeth sit left of quadrant 2 teeth, and quadrant 4 left of
     quadrant 3, as on an unflipped radiograph (patient's right on the
     image's left);
   - each flipped channel equals the mirror image of its mirror tooth's
     original channel (for example new channel 21 == fliplr(old channel 11));
   - the mask and the box map agree: every mask pixel of channel k lies
     inside the box of channel k.
It also runs the original, unfixed augment() to confirm the checks catch
the bug.

Usage: python verify_flip_augment.py   (exits 1 if any check fails)
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[3]
NOTEBOOKS = [
    "notebooks/Unet/unet_training.ipynb",
    "notebooks/Unet/unet+cv.ipynb",
    "notebooks/yolov8+unet/yolov8+unet_training.ipynb",
    "notebooks/yolov8+unet/yolov8+unet+cv.ipynb",
]
MASK_DIR = REPO / "Dataset" / "bb_u_net_dataset" / "labels" / "cate8 - export"
IMAGE_ID = "cate8-00390"  # the Fig. 1 / Fig. 3 image in the CJSJ paper

FDI_CODES = [
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
]

ORIGINAL_AUGMENT = '''
def augment(input_image,input_mask,input_bb):
    if np.random.uniform() > 0.5:
        input_image = np.fliplr(input_image)
        input_mask = np.fliplr(input_mask)
        input_bb = np.fliplr(input_bb)

    return input_image,input_mask,input_bb
'''


class AlwaysFlip:
    """Stands in for np.random so the flip branch always runs."""

    @staticmethod
    def uniform():
        return 1.0


def load_augment(source):
    np_flip = type("np_flip", (), {})()
    for name in dir(np):
        if not name.startswith("__"):
            setattr(np_flip, name, getattr(np, name))
    np_flip.random = AlwaysFlip
    namespace = {"np": np_flip}
    exec(source, namespace)
    return namespace["augment"]


def notebook_augment_source(path):
    cells = json.loads((REPO / path).read_text(encoding="utf-8"))["cells"]
    sources = ["".join(c["source"]) for c in cells if "def augment(" in "".join(c["source"])]
    assert len(sources) == 1, f"{path}: expected one augment() cell, found {len(sources)}"
    return sources[0]


def build_inputs():
    mask = np.zeros((len(FDI_CODES) + 1, 512, 512), dtype=np.uint8)
    for k, code in enumerate(FDI_CODES):
        f = MASK_DIR / f"{IMAGE_ID}_{code}.ome.tiff"
        if f.exists():
            mask[k] = tifffile.imread(f) > 0
    bb = np.zeros((len(FDI_CODES), 512, 512), dtype=np.uint8)
    for k in range(len(FDI_CODES)):
        ys, xs = np.nonzero(mask[k])
        if len(xs):
            bb[k, ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
    image = np.random.default_rng(0).random((512, 512, 3))
    # resize_img() transposes both to (H, W, C) before augment() runs.
    return image, np.transpose(mask, (1, 2, 0)), np.transpose(bb, (1, 2, 0))


def mean_x(arr, codes):
    xs = [np.nonzero(arr[:, :, FDI_CODES.index(c)])[1] for c in codes]
    xs = [x for x in xs if len(x)]
    return np.concatenate(xs).mean()


def check(augment, image, mask, bb):
    present = [k for k in range(len(FDI_CODES)) if mask[:, :, k].any()]
    _, m2, b2 = augment(image.copy(), mask.copy(), bb.copy())
    results = {}
    for name, arr in (("mask", m2), ("box map", b2)):
        q = {d: [c for c in FDI_CODES if c[0] == d] for d in "1234"}
        results[f"{name}: quadrant 1 left of quadrant 2"] = mean_x(arr, q["1"]) < mean_x(arr, q["2"])
        results[f"{name}: quadrant 4 left of quadrant 3"] = mean_x(arr, q["4"]) < mean_x(arr, q["3"])
        src = mask if name == "mask" else bb
        ok = True
        for k in range(len(FDI_CODES)):
            code = FDI_CODES[k]
            mirror = {"1": "2", "2": "1", "3": "4", "4": "3"}[code[0]] + code[1]
            ok &= np.array_equal(arr[:, :, k], np.fliplr(src[:, :, FDI_CODES.index(mirror)]))
        results[f"{name}: new channel k == flipped old mirror channel"] = ok
    inside = all(not (m2[:, :, k].astype(bool) & ~b2[:, :, k].astype(bool)).any() for k in present)
    results["mask pixels lie inside the box of the same channel"] = inside
    return results


def main():
    image, mask, bb = build_inputs()
    n_teeth = int(sum(mask[:, :, k].any() for k in range(len(FDI_CODES))))
    print(f"{IMAGE_ID}: {n_teeth} teeth with masks")

    print("\nOriginal (unfixed) augment(), expected to FAIL:")
    orig = check(load_augment(ORIGINAL_AUGMENT), image, mask, bb)
    for name, ok in orig.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    caught = not all(orig.values())

    all_ok = True
    for nb in NOTEBOOKS:
        print(f"\n{nb}:")
        res = check(load_augment(notebook_augment_source(nb)), image, mask, bb)
        for name, ok in res.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok &= all(res.values())

    print(f"\nChecks catch the original bug: {caught}")
    print("ALL NOTEBOOKS PASS" if all_ok and caught else "AT LEAST ONE CHECK FAILED")
    sys.exit(0 if all_ok and caught else 1)


if __name__ == "__main__":
    main()
