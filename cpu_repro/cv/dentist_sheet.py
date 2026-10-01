"""Blinded dentist sheet (cpu_repro/cv/LABEL_CHECK_RULES.md, check C).

    python dentist_sheet.py --per-tooth <confidence-first per_tooth_predictions.csv> \
        --images <dir of uncropped X-rays> --out <dir>

All joint same-wrong teeth (RESULTS.md Section 65) plus as many control
teeth, drawn with seed 0 from teeth all three detectors number correctly,
shuffled together (seed 0). Each page shows the whole uncropped X-ray,
enlarged 2x, with one tooth outlined and no label, so the dentist can count
along the arch. Writes:
  dentist_sheet.pdf     one item per page, "Item N" only
  response_form.csv     item, fdi_answer, cant_tell (to fill in)
  answer_key.csv        item -> image_id, fdi, group (keep away from the reader)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]
SCALE = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tooth", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    t = pd.read_csv(a.per_tooth)
    joint = t["coord_pred"] != t["class_id"]
    for d in DETS:
        joint &= (t[f"{d}_pred"] != t["class_id"]) & t[f"{d}_pred"].notna()
    p = np.stack([t[f"{d}_pred"].to_numpy() for d in DETS])
    joint &= (p == p[0]).all(0)
    control = np.logical_and.reduce([t[f"{d}_pred"] == t["class_id"] for d in DETS])
    J = t[joint].assign(group="joint")
    C = t[control].sample(len(J), random_state=int(rng.integers(2**31))).assign(group="control")
    s = pd.concat([J, C]).sample(frac=1, random_state=int(rng.integers(2**31))).reset_index(drop=True)
    s["item"] = np.arange(1, len(s) + 1)
    print(s["group"].value_counts().to_dict(), "X-rays:", s["image_id"].nunique(), flush=True)

    font = ImageFont.load_default(size=36)
    pages = []
    for r in s.itertuples():
        img = Image.open(a.images / f"{r.image_id}.jpg").convert("RGB")
        w, h = img.size
        img = img.resize((w * SCALE, h * SCALE), Image.LANCZOS)
        W, H = img.size
        page = Image.new("RGB", (W, H + 70), "white")
        page.paste(img, (0, 70))
        dr = ImageDraw.Draw(page)
        x0, x1 = (r.x_center - r.width / 2) * W, (r.x_center + r.width / 2) * W
        y0, y1 = (r.y_center - r.height / 2) * H + 70, (r.y_center + r.height / 2) * H + 70
        dr.rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], outline=(255, 0, 0), width=4)
        dr.text((20, 15), f"Item {r.item}", fill="black", font=font)
        pages.append(page)
    pages[0].save(a.out / "dentist_sheet.pdf", save_all=True, append_images=pages[1:], resolution=150)
    s[["item"]].assign(fdi_answer="", cant_tell="").to_csv(a.out / "response_form.csv", index=False)
    s[["item", "image_id", "fdi", "class_id", "group"]].to_csv(a.out / "answer_key.csv", index=False)
    print("wrote", len(pages), "pages", flush=True)


if __name__ == "__main__":
    main()
