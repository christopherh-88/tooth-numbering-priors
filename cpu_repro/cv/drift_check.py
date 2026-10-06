"""Space closure at gaps (cpu_repro/cv/DRIFT_RULES.md, frozen 2026-10-05).

    python drift_check.py --repo <repo checkout> \
        --per-tooth <confidence-first per_tooth_predictions.csv> --out <dir> [--images N]

For every missing non-third-molar tooth M with both arch neighbors X and Z
labeled: the space between X and Z, from the masks (primary) and from the
boxes (secondary), over M's median box width. Compares filled slots (X or
Z is a joint failure given M's number) with unfilled ones. Writes
drift_slots.csv and drift_summary.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from gap_check import ORDER

DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]
MODELS = {"joint": None, "yolov8x": "yolov8x_pred", "rtdetr_l": "rtdetr_l_pred",
          "fasterrcnn": "fasterrcnn_pred", "position_only": "coord_pred"}
FDI = [f"{q}{t}" for q in "1234" for t in "12345678"]
N_BOOT = 10_000
CLOSED = 0.5


def load_mask(path):
    from PIL import Image
    return np.squeeze(np.array(Image.open(path))) > 0


def mask_distance(a, b):
    """Shortest pixel distance between two boolean masks (0 if they touch or overlap)."""
    import cv2
    if (a & b).any():
        return 0.0
    dt = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return float(max(dt[a].min() - 1, 0.0))  # dt counts the step onto b's pixel; touching pixels give 0


def auroc(score, y):
    """P(score of a positive > score of a negative), ties count half."""
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot(slots, col, ycol, rng):
    """Point and 95% CI, resampling X-rays: AUROC of -col for ycol, and
    median col (unfilled) minus median col (filled)."""
    groups = [g.index.to_numpy() for _, g in slots.groupby("image_id")]
    s, y = slots[col].to_numpy(), slots[ycol].to_numpy(bool)
    def stats(ix):
        yy, ss = y[ix], s[ix]
        if yy.all() or not yy.any():
            return np.nan, np.nan
        return auroc(-ss, yy), np.median(ss[~yy]) - np.median(ss[yy])
    point = stats(np.arange(len(s)))
    draws = np.array([stats(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
                      for _ in range(N_BOOT)])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return dict(auroc=point[0], auroc_lo=lo[0], auroc_hi=hi[0],
                median_diff=point[1], median_diff_lo=lo[1], median_diff_hi=hi[1])


def rule(r):
    if r["auroc_lo"] <= 0.5 <= r["auroc_hi"]:
        return 2
    if r["median_diff_lo"] > 0 and r["auroc"] >= 0.75:
        return 1
    return 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--per-tooth", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", type=int, default=0, help="smoke test on N X-rays spread over the sorted list")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(a.per_tooth, dtype={"fdi": str})
    mask_path = {p.name.replace(".ome.tiff", ""): p
                 for p in (a.repo / "Dataset/bb_u_net_dataset/labels").glob("*/*.ome.tiff")}
    joint = t["coord_pred"] != t["class_id"]
    for d in DETS:
        joint &= (t[f"{d}_pred"] != t["class_id"]) & t[f"{d}_pred"].notna()
    joint &= (t["yolov8x_pred"] == t["rtdetr_l_pred"]) & (t["yolov8x_pred"] == t["fasterrcnn_pred"])
    t["joint_pred"] = t["yolov8x_pred"].where(joint)
    MODELS["joint"] = "joint_pred"
    ref_w = (t.groupby("fdi")["width"].median() * 512).to_dict()  # all X-rays are 512 x 512

    ids = sorted(t["image_id"].unique())
    if a.images:
        ids = ids[::max(1, len(ids) // a.images)][:a.images]
    rows, skipped = [], dict(third_molar=0, longer_gap=0)
    for image_id in ids:
        g = t[t["image_id"] == image_id].set_index("fdi")
        for o in ORDER.values():
            for i, m in enumerate(o):
                if m in g.index:
                    continue
                if m[1] == "8":
                    skipped["third_molar"] += 1
                    continue
                x, z = o[i - 1], o[i + 1]  # image-left and image-right neighbors
                if x not in g.index or z not in g.index:
                    skipped["longer_gap"] += 1
                    continue
                X, Z = g.loc[x], g.loc[z]
                w = ref_w[m]
                d_mask = mask_distance(load_mask(mask_path[f"{image_id}_{x}"]), load_mask(mask_path[f"{image_id}_{z}"]))
                d_box = ((Z.x_center - Z.width / 2) - (X.x_center + X.width / 2)) * 512
                row = dict(image_id=image_id, missing=m, left=x, right=z, ref_width_px=w,
                           dist_mask_px=d_mask, S=d_mask / w, S_box=d_box / w)
                mc = FDI.index(m)
                for name, col in MODELS.items():
                    row[f"filled_{name}"] = bool((X[col] == mc) or (Z[col] == mc))
                rows.append(row)
    slots = pd.DataFrame(rows)
    if slots.empty:
        print("no slots; skipped", skipped, flush=True)
        return
    slots.to_csv(a.out / "drift_slots.csv", index=False, float_format="%.4f")
    print(len(slots), "slots in", slots["image_id"].nunique(), "X-rays; skipped", skipped,
          "; filled (joint):", int(slots["filled_joint"].sum()), flush=True)

    rng = np.random.default_rng(0)
    out = []
    for measure in ("S", "S_box"):
        for name in MODELS:
            y = slots[f"filled_{name}"]
            r = dict(measure=measure, outcome=name, n_slots=len(slots), n_filled=int(y.sum()),
                     median_filled=slots.loc[y, measure].median(), median_unfilled=slots.loc[~y, measure].median(),
                     closed_pct_filled=100 * (slots.loc[y, measure] < CLOSED).mean(),
                     closed_pct_unfilled=100 * (slots.loc[~y, measure] < CLOSED).mean(),
                     filled_pct_among_closed=100 * y[slots[measure] < CLOSED].mean())
            r |= boot(slots, measure, f"filled_{name}", rng)
            r["rule"] = rule(r) if (measure == "S" and name == "joint") else np.nan
            out.append(r)
    out = pd.DataFrame(out)
    out.to_csv(a.out / "drift_summary.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
