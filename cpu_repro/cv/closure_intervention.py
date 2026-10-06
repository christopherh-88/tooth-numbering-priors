"""Closure intervention (cpu_repro/cv/CLOSURE_INTERVENTION_RULES.md, frozen 2026-10-05).

For each Section 71 target tooth T: erase T (open), then move T's distal
neighbor N toward the mesial neighbor M by half (half) or all (closed) of
the distance at which N's mask touches M's, and ask whether all three
detectors give N the number of T. Runs in a Kaggle GPU kernel (see
kaggle/make_phase3_kernels.py):

    python closure_intervention.py --repo <repo checkout> --models <dir with *_cvseed0/> \
        --per-tooth <confidence-first per_tooth_predictions.csv> --out <dir> [--images N]

Writes closure_rows.csv (one row per detector, target and condition),
closure_summary.csv (rates, paired CIs, rule) and example images
(intact, open, half, closed) for a visual check.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import shift_test as st
from gap_check import ORDER
from gap_intervention import DILATE_PX, INPAINT_RADIUS, dilate, draw_targets, load_mask

DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]
CONDITIONS = {"open": 0.0, "half": 0.5, "closed": 1.0}  # slide mode (Section 76)
TIP_CONDITIONS = {"open": None, "repaste": 0.0, "half": 0.5, "closed": 1.0}  # tip mode (TIPPING_RULES.md)
MAX_ANGLE, ANGLE_STEP = 45.0, 0.5
FEATURES = ["x_center", "y_center", "width", "height", "area", "aspect_ratio"]
MAX_SHIFT = 200
N_BOOT = 10_000
N_EXAMPLES = 8
POS = {f: (arch, i) for arch, o in ORDER.items() for i, f in enumerate(o)}


def neighbors_of(T):
    """(distal, mesial) arch neighbors of T, and +1 if the distal one is image-left of T else -1."""
    arch, i = POS[T]
    o = ORDER[arch]
    left, right = o[i - 1], o[i + 1]
    # positions 0..7 are image-left of the midline, 8..15 image-right; distal = farther from it
    if abs(i - 1 - 7.5) > abs(i + 1 - 7.5):
        return left, right, 1  # distal is on the left, moves right
    return right, left, -1


def shift_x(a, s):
    """Shift a 2D or 3D array s pixels along x, filling with zeros."""
    out = np.zeros_like(a)
    if s > 0:
        out[:, s:] = a[:, :-s]
    elif s < 0:
        out[:, :s] = a[:, -s:]
    else:
        out[:] = a
    return out


def touch_shift(n_mask, m_mask, direction):
    """Smallest shift (pixels, in direction) at which N's mask touches or overlaps M's."""
    m1 = dilate_px(m_mask, 1)
    for s in range(0, MAX_SHIFT + 1):
        if (shift_x(n_mask, direction * s) & m1).any():
            return s
    return None


def dilate_px(m, px):
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(m, k)


def move_tooth(img, t_region, n_region, s):
    """Erase T and N's old place, then paste N's pixels shifted by s along x."""
    import cv2
    base = cv2.inpaint(img, (t_region | n_region).astype(np.uint8), INPAINT_RADIUS, cv2.INPAINT_TELEA)
    if s == 0:
        return cv2.inpaint(img, t_region.astype(np.uint8), INPAINT_RADIUS, cv2.INPAINT_TELEA)
    moved = shift_x(n_region, s).astype(bool)
    base[moved] = shift_x(img, s)[moved]
    return base


def apex(n_mask, upper):
    """Root apex: mean x of the mask's top rows (upper teeth) or bottom rows (lower teeth)."""
    ys, xs = np.nonzero(n_mask)
    edge = ys.min() if upper else ys.max()
    near = np.abs(ys - edge) <= 2
    return float(xs[near].mean()), float(edge)


def rotate(a, center, angle, nearest=False):
    import cv2
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(a, m, (a.shape[1], a.shape[0]), flags=flags, borderValue=0)


def tip_sign(n_mask, center, direction):
    """Rotation sign (degrees, OpenCV counter-clockwise positive) that moves N's crown toward M."""
    x0 = np.nonzero(n_mask)[1].mean()
    x1 = np.nonzero(rotate(n_mask, center, 2.0, nearest=True))[1].mean()
    return 1.0 if np.sign(x1 - x0) == direction else -1.0


def touch_angle(n_mask, m_mask, center, sign):
    """Smallest tip angle (degrees) at which N's rotated mask touches or overlaps M's."""
    m1 = dilate_px(m_mask, 1)
    for ang in np.arange(ANGLE_STEP, MAX_ANGLE + 1e-9, ANGLE_STEP):
        if (rotate(n_mask, center, sign * ang, nearest=True) & m1).any():
            return float(ang)
    return None


def tip_tooth(img, t_region, n_region, center, angle):
    """Erase T and N's old place, then paste N rotated by angle about its apex (angle 0 = paste back unmoved)."""
    import cv2
    base = cv2.inpaint(img, (t_region | n_region).astype(np.uint8), INPAINT_RADIUS, cv2.INPAINT_TELEA)
    reg = rotate(n_region.astype(np.uint8), center, angle, nearest=True).astype(bool)
    base[reg] = rotate(img, center, angle)[reg]
    return base


def mask_box(m, shape):
    ys, xs = np.nonzero(m)
    h, w = shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return np.array([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h])


def position_models(boxes, folds):
    b = boxes.merge(folds[["image_id", "fold"]], on="image_id", validate="m:1")
    b = b.assign(area=b["width"] * b["height"], aspect_ratio=b["width"] / b["height"])
    return {f: HistGradientBoostingClassifier(random_state=0).fit(b.loc[b["fold"] != f, FEATURES],
                                                                 b.loc[b["fold"] != f, "class_id"])
            for f in sorted(b["fold"].unique())}


def boot_pair(df, a, b, rng):
    """Paired rates of a and b (columns of 0/1 per mover) and of a minus b, resampling X-rays."""
    s = df.assign(n=1).groupby("image_id")[["n", a, b]].sum().to_numpy(float)
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    bt = s[idx].sum(1)
    ra, rb = 100 * bt[:, 1] / bt[:, 0], 100 * bt[:, 2] / bt[:, 0]
    tot = s.sum(0)
    out = {}
    for name, v, pt in ((a, ra, 100 * tot[1] / tot[0]), (b, rb, 100 * tot[2] / tot[0]),
                        (f"{a}_minus_{b}", ra - rb, 100 * (tot[1] - tot[2]) / tot[0])):
        out[name], (out[f"{name}_lo"], out[f"{name}_hi"]) = pt, np.percentile(v, [2.5, 97.5])
    return out


def rule(r):
    if r["closed_minus_open_lo"] <= 0 <= r["closed_minus_open_hi"]:
        return 2
    if r["closed_minus_open_lo"] > 0 and r["closed"] >= 5:
        return 1
    return 3


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--per-tooth", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", type=int, default=0, help="smoke test on N X-rays spread over the list")
    ap.add_argument("--cv-seed", type=int, default=0, help="which split's models (0, or 1 for the replication)")
    ap.add_argument("--folds", type=Path, default=None, help="folds CSV for that split (default folds.csv)")
    ap.add_argument("--mode", choices=["slide", "tip"], default="slide", help="slide (Section 76) or tip")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    st.MATCH_ORDER = "conf"

    t = pd.read_csv(a.per_tooth, dtype={"fdi": str})
    img_dir = a.repo / "Dataset/bb_u_net_dataset/panoramic_x_rays"
    mask_path = {p.name.replace(".ome.tiff", ""): p
                 for p in (a.repo / "Dataset/bb_u_net_dataset/labels").glob("*/*.ome.tiff")}
    all_ids = sorted(t["image_id"].unique())
    plan = draw_targets(t, all_ids, mask_path, np.random.default_rng(0))  # identical to Section 71
    ids = all_ids[::max(1, len(all_ids) // a.images)][:a.images] if a.images else all_ids
    pos_models = position_models(pd.read_csv(a.repo / "cpu_repro/cv/boxes.csv", dtype={"fdi": str}),
                                 pd.read_csv(a.folds or a.repo / "cpu_repro/cv/folds.csv"))

    # Build every altered image once; reused by all detectors.
    jobs, skipped, examples = [], 0, 0
    for image_id in ids:
        if not plan[image_id]:
            continue
        g = t[t["image_id"] == image_id].reset_index(drop=True)
        img = cv2.imread(str(img_dir / f"{image_id}.jpg"))
        row_of = {fdi: i for i, fdi in enumerate(g["fdi"])}
        for T in plan[image_id]:
            N, M, direction = neighbors_of(T)
            if f"{image_id}_{N}" not in mask_path or f"{image_id}_{M}" not in mask_path:
                skipped += 1
                continue
            t_region = dilate(load_mask(mask_path[f"{image_id}_{T}"], img.shape[:2])).astype(bool)
            n_mask = load_mask(mask_path[f"{image_id}_{N}"], img.shape[:2])
            m_mask = load_mask(mask_path[f"{image_id}_{M}"], img.shape[:2])
            n_region = dilate(n_mask).astype(bool)
            i_n = row_of[N]
            base = dict(image_id=image_id, fold=int(g.loc[0, "fold"]), target=T, mover=N, mesial=M,
                        target_class=int(g.loc[row_of[T], "class_id"]), mover_class=int(g.loc[i_n, "class_id"]),
                        g=g, row_of=row_of)
            if a.mode == "slide":
                full = touch_shift(n_mask, m_mask, direction)
                if full is None:
                    skipped += 1
                    continue
                for cond, frac in CONDITIONS.items():
                    s = int(round(frac * full)) * direction
                    box = g.loc[i_n, ["x_center", "y_center", "width", "height"]].to_numpy(float).copy()
                    box[0] += s / img.shape[1]
                    jobs.append(base | dict(condition=cond, shift_px=abs(s), full_shift_px=full,
                                            img=move_tooth(img, t_region, n_region, s), mover_box=box))
            else:
                center = apex(n_mask, N[0] in "12")
                sign = tip_sign(n_mask, center, direction)
                full = touch_angle(n_mask, m_mask, center, sign)
                if full is None:
                    skipped += 1
                    continue
                for cond, frac in TIP_CONDITIONS.items():
                    if frac is None:  # open: only T erased
                        altered = move_tooth(img, t_region, n_region, 0)
                        box, ang = g.loc[i_n, ["x_center", "y_center", "width", "height"]].to_numpy(float), 0.0
                    else:
                        ang = frac * full
                        altered = tip_tooth(img, t_region, n_region, center, sign * ang)
                        box = mask_box(rotate(n_mask, center, sign * ang, nearest=True), img.shape[:2])
                    jobs.append(base | dict(condition=cond, shift_px=ang, full_shift_px=full, img=altered,
                                            mover_box=box))
            n_conds = len(CONDITIONS) if a.mode == "slide" else len(TIP_CONDITIONS)
            if examples < N_EXAMPLES:
                panels = [img] + [j["img"] for j in jobs[-n_conds:]]
                cv2.imwrite(str(a.out / f"example_{image_id}_{T}.png"), np.concatenate(panels, 1))
                examples += 1
    n_conds = len(CONDITIONS) if a.mode == "slide" else len(TIP_CONDITIONS)
    print(len(jobs) // n_conds, "targets with a mover;", skipped, "skipped", flush=True)

    # Position-only answer for the moved box.
    for j in jobs:
        x, y, w, h = j["mover_box"]
        f = pd.DataFrame([dict(x_center=x, y_center=y, width=w, height=h, area=w * h, aspect_ratio=w / h)])
        j["position_only_pred"] = int(pos_models[j["fold"]].predict(f[FEATURES])[0])
    # Check: on unmoved boxes the position-only answer must equal the stored one.
    opens = [j for j in jobs if j["condition"] == "open"]
    agree = np.mean([j["position_only_pred"] == j["g"].loc[j["row_of"][j["mover"]], "coord_pred"] for j in opens])
    print(f"position-only check on unmoved boxes: {agree:.4f} agree with the per-tooth file", flush=True)

    rows = []
    for det in DETS:
        runs = [p for p in a.models.rglob(f"{det}_cvseed{a.cv_seed}") if p.is_dir() and "/repo/" not in str(p)]
        assert len(runs) == 1, (det, runs)
        run = runs[0]
        for fold in range(5):
            fj = [j for j in jobs if j["fold"] == fold]
            if not fj:
                continue
            model = st.Detector(det, run / f"fold{fold}")
            intact = {}
            for j in fj:
                g, row_of = j["g"], j["row_of"]
                gt = np.stack([g.x_center - g.width / 2, g.y_center - g.height / 2,
                               g.x_center + g.width / 2, g.y_center + g.height / 2], 1)
                if j["image_id"] not in intact:
                    intact[j["image_id"]] = st.match(gt, *model(cv2.imread(str(img_dir / f"{j['image_id']}.jpg"))))
                keep = [i for i in range(len(g)) if g.fdi[i] != j["target"]]
                gt2 = gt.copy()
                x, y, w, h = j["mover_box"]
                gt2[row_of[j["mover"]]] = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
                pred = np.full(len(g), np.nan)
                pred[keep] = st.match(gt2[keep], *model(j["img"]))
                rows.append(dict(detector=det, image_id=j["image_id"], target=j["target"], mover=j["mover"],
                                 condition=j["condition"], target_class=j["target_class"],
                                 mover_class=j["mover_class"], shift_px=j["shift_px"],
                                 full_shift_px=j["full_shift_px"], position_only_pred=j["position_only_pred"],
                                 intact_pred=intact[j["image_id"]][row_of[j["mover"]]],
                                 pred=pred[row_of[j["mover"]]]))
            print(det, "fold", fold, "done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(a.out / "closure_rows.csv", index=False)

    summarize(d, a.out, list(CONDITIONS) if a.mode == "slide" else list(TIP_CONDITIONS))


def summarize(d, out_dir, conds=tuple(CONDITIONS)):
    """Movers all three detectors number right on the intact image; paired rates by condition."""
    key = ["image_id", "target", "condition"]
    base = d.drop_duplicates(key).set_index(key)[["target_class", "mover_class", "position_only_pred"]]
    by = {det: d[d["detector"] == det].set_index(key) for det in DETS}
    ok = np.logical_and.reduce([by[det]["intact_pred"].reindex(base.index) == base["mover_class"] for det in DETS])
    base = base[ok]
    m = pd.DataFrame(index=base.index)
    fills = []
    for det in DETS:
        pred = by[det]["pred"].reindex(base.index)
        m[f"fill_{det}"] = (pred == base["target_class"]).astype(int)
        m[f"missed_{det}"] = pred.isna().astype(int)
        fills.append(m[f"fill_{det}"].astype(bool))
    m["all3"] = np.logical_and.reduce(fills).astype(int)
    m["pos_fill"] = (base["position_only_pred"] == base["target_class"]).astype(int)
    m["all4"] = (m["all3"].astype(bool) & m["pos_fill"].astype(bool)).astype(int)
    wide = m.unstack("condition").dropna()
    n_movers = len(wide)

    rng = np.random.default_rng(0)
    out = []
    for measure in ["all3", "all4", "pos_fill"] + [f"fill_{x}" for x in DETS] + [f"missed_{x}" for x in DETS]:
        x = pd.DataFrame({c: wide[(measure, c)].to_numpy() for c in conds})
        x["image_id"] = wide.index.get_level_values("image_id")
        r = dict(measure=measure, n_movers=n_movers)
        r |= boot_pair(x, "closed", "open", rng)
        r |= {k: v for k, v in boot_pair(x, "half", "open", rng).items() if k.startswith("half")}
        if "repaste" in conds:  # TIPPING_RULES.md paste check
            r |= {k: v for k, v in boot_pair(x, "repaste", "open", rng).items() if k.startswith("repaste")}
            r |= {k: v for k, v in boot_pair(x, "closed", "repaste", rng).items() if k.startswith("closed_minus")}
        r["rule"] = rule(r) if measure == "all3" else np.nan
        if measure == "all3" and "repaste" in conds:
            r["paste_confounded"] = bool(r["repaste_minus_open_lo"] > 0
                                         and r["repaste_minus_open"] > r["closed_minus_open"] / 3)
        out.append(r)
    out = pd.DataFrame(out)
    out.to_csv(out_dir / "closure_summary.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(out.round(2).to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
