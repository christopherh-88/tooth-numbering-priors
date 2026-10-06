"""Fig. 1 candidate: a closed gap with a shared error, and space closure at all gaps.

    python paper/miccai/make_fig_closed_gap.py [--image cate10-00054 --missing 16]

(a) Crop of one X-ray where tooth `missing` has no label and its neighbors'
masks touch; the neighbor that every model (three detectors and the
position-only model) numbers as the missing tooth is outlined.
(b) Share of gaps with space S at or below x, for gaps with and without a
joint failure (RESULTS.md Section 72).
Writes fig_closed_gap.png and .pdf next to this script.
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "cpu_repro/cv"))

FDI = [f"{q}{t}" for q in "1234" for t in "12345678"]
FILLED, OPEN = "#eb6834", "#2a78d6"  # dataviz reference palette slots 2 and 1, validated light
INK, MUTED = "#222222", "#6b6b6b"


def mask(image_id, fdi):
    p = next((ROOT / "Dataset/bb_u_net_dataset/labels").glob(f"*/{image_id}_{fdi}.ome.tiff"))
    return (np.squeeze(np.array(Image.open(p))) > 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="cate10-00054")
    ap.add_argument("--missing", default="16")
    a = ap.parse_args()
    slots = pd.read_csv(ROOT / "cpu_repro/cv/results/kaggle_drift/drift/drift_slots.csv",
                        dtype={"missing": str, "left": str, "right": str})
    teeth = pd.read_csv(ROOT / "benchmark/teeth.csv", dtype={"fdi": str})
    slot = slots[(slots["image_id"] == a.image) & (slots["missing"] == a.missing)].iloc[0]
    g = teeth[(teeth["image_id"] == a.image) & teeth["fdi"].isin([slot.left, slot.right])]
    jf = g[g["joint_failure"]].iloc[0]
    other = slot.left if jf.fdi == slot.right else slot.right

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1, 1.15]))

    # (a) example crop
    img = cv2.imread(str(ROOT / "Dataset/bb_u_net_dataset/panoramic_x_rays" / f"{a.image}.jpg"), cv2.IMREAD_GRAYSCALE)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for fdi, color in ((jf.fdi, FILLED), (other, OPEN)):
        cs, _ = cv2.findContours(mask(a.image, fdi), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cs, -1, tuple(int(color[i:i + 2], 16) for i in (1, 3, 5)), 2)
    cx, cy = int(jf.x_center * 512), int(jf.y_center * 512)
    y0, y1 = max(cy - 95, 0), min(cy + 95, 512)
    x0, x1 = max(cx - 110, 0), min(cx + 110, 512)
    ax0.imshow(rgb[y0:y1, x0:x1])
    ax0.set_xticks([])
    ax0.set_yticks([])
    for s in ax0.spines.values():
        s.set_visible(False)
    shared = FDI[int(jf.shared_wrong_class)]
    ax0.set_title(f"(a) {a.missing} missing, space closed", fontsize=10, color=INK, loc="left")
    ax0.set_xlabel(f"Orange: labeled {jf.fdi}, every model says {shared}\nBlue: {other}, numbered right",
                   fontsize=8.5, color=INK, loc="left")

    # (b) cumulative share of gaps by space
    xs = np.linspace(0, 1.2, 241)
    for filled, color, label in ((True, FILLED, "gaps with a joint failure"), (False, OPEN, "gaps without")):
        v = np.sort(slots.loc[slots["filled_joint"] == filled, "S"].to_numpy())
        y = 100 * np.searchsorted(v, xs, side="right") / len(v)
        ax1.plot(xs, y, color=color, lw=2, label=f"{label} (n = {len(v)})")
    ax1.legend(loc="lower right", fontsize=8.5, frameon=False, labelcolor=INK)
    ax1.set_xlim(0, 1.2)
    ax1.set_ylim(0, 102)
    ax1.set_xlabel("Space S left between the neighbors\n(0 = touching, 1 = one tooth width)", fontsize=9, color=INK)
    ax1.set_ylabel("Gaps with space at most S (%)", fontsize=9, color=INK)
    ax1.set_title("(b) Space at 372 gaps (AUROC 0.78)", fontsize=10, color=INK, loc="left")
    ax1.grid(color="#e6e6e3", lw=0.8)
    ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax1.spines[s].set_color(MUTED)
    ax1.tick_params(colors=MUTED, labelsize=8.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"fig_closed_gap.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
