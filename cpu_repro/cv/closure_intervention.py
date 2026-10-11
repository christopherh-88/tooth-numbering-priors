"""Closure intervention (cpu_repro/cv/CLOSURE_INTERVENTION_RULES.md, frozen 2026-10-05).

For each Section 71 target tooth T: erase T (open), then move T's distal
neighbor N toward the mesial neighbor M by half (half) or all (closed) of
the distance at which N's mask touches M's, and ask whether all three
detectors give N the number of T. Runs in a Kaggle GPU kernel (see
kaggle/make_phase3_kernels.py):

    python closure_intervention.py --repo <repo checkout> --models <dir with *_cvseed0/> \
        --per-tooth <confidence-first per_tooth_predictions.csv> --out <dir> [--images N]

Writes closure_rows.csv (one row per detector, target and condition),
closure_summary.csv (rates, paired CIs, rule), intact_check.csv and example
images (intact, open, half, closed) for a visual check.

DENTEX mode (cpu_repro/cv/DENTEX_INTERVENTION_RULES.md): --dataset dentex with
--labels <train_quadrant_enumeration.json> --images-dir <xrays> --run-tag _dentex
(or _dentex_tseed1) --per-tooth <dentex_teeth.csv>. X-rays and tooth polygons
are resized to --work-width (640, the size the detectors read). The pixel
constants of the UFBA runs (3, 5, 200 px) are kept: the median tooth is 23.9 px
wide there against 24.0 px in UFBA's 512-px X-rays (--px-scale 1.0).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import gap_intervention as gi
import shift_test as st
from gap_check import ORDER
from gap_intervention import INPAINT_RADIUS, dilate, draw_targets, load_mask

DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]
CONDITIONS = {"open": 0.0, "half": 0.5, "closed": 1.0}  # slide mode (Section 76)
TIP_CONDITIONS = {"open": None, "repaste": 0.0, "half": 0.5, "closed": 1.0}  # tip mode (TIPPING_RULES.md)
MAX_ANGLE, ANGLE_STEP = 45.0, 0.5
FEATURES = ["x_center", "y_center", "width", "height", "area", "aspect_ratio"]
MAX_SHIFT = 200
TOUCH_PX = 1  # N touches M when their masks come within this many pixels
N_BOOT = 10_000
N_EXAMPLES = 8
POS = {f: (arch, i) for arch, o in ORDER.items() for i, f in enumerate(o)}


class UfbaData:
    """UFBA-425: 512 x 512 JPEG X-rays and one OME-TIFF mask per tooth."""

    def __init__(self, repo):
        self.img_dir = repo / "Dataset/bb_u_net_dataset/panoramic_x_rays"
        self.paths = {p.name.replace(".ome.tiff", ""): p
                      for p in (repo / "Dataset/bb_u_net_dataset/labels").glob("*/*.ome.tiff")}

    def __contains__(self, key):
        return key in self.paths

    def image(self, image_id):
        import cv2
        return cv2.imread(str(self.img_dir / f"{image_id}.jpg"))

    def mask(self, image_id, fdi, shape):
        return load_mask(self.paths[f"{image_id}_{fdi}"], shape)


class DentexData:
    """DENTEX quadrant enumeration: PNG X-rays resized to work_width, tooth polygons rasterized at that size.
    FDI = quadrant name + position name, as dentex_external.load_teeth."""

    def __init__(self, labels, images_dir, work_width):
        d = json.load(open(labels))
        self.images_dir, self.work_width = Path(images_dir), work_width
        self.size = {Path(im["file_name"]).stem: (im["width"], im["height"]) for im in d["images"]}
        stem = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
        q = {c["id"]: str(c["name"]) for c in d["categories_1"]}
        p = {c["id"]: str(c["name"]) for c in d["categories_2"]}
        self.polys = {}
        for a in d["annotations"]:
            key = f"{stem[a['image_id']]}_{q[a['category_id_1']]}{p[a['category_id_2']]}"
            seg = json.loads(a["segmentation"]) if isinstance(a["segmentation"], str) else a["segmentation"]
            self.polys.setdefault(key, []).extend(np.array(s, float).reshape(-1, 2) for s in seg)

    def __contains__(self, key):
        return key in self.polys

    def work_size(self, image_id):
        w, h = self.size[image_id]
        if not self.work_width:  # stored size
            return w, h
        return self.work_width, int(round(h * self.work_width / w))

    def image(self, image_id):
        import cv2
        img = cv2.imread(str(self.images_dir / f"{image_id}.png"))
        assert img is not None, image_id
        return cv2.resize(img, self.work_size(image_id), interpolation=cv2.INTER_AREA)

    def mask(self, image_id, fdi, shape):
        import cv2
        w, h = self.size[image_id]
        m = np.zeros(shape, np.uint8)
        for poly in self.polys[f"{image_id}_{fdi}"]:
            pts = (poly * [shape[1] / w, shape[0] / h]).round().astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
        return m


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
    m1 = dilate_px(m_mask, TOUCH_PX)
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
    m1 = dilate_px(m_mask, TOUCH_PX)
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


def set_constants(scale):
    """UFBA's pixel constants (set on 24-px teeth) times scale: DENTEX at 640 px wide has scale 1.0."""
    global INPAINT_RADIUS, MAX_SHIFT, TOUCH_PX
    gi.DILATE_PX = max(1, int(round(3 * scale)))
    INPAINT_RADIUS = max(1, int(round(5 * scale)))
    MAX_SHIFT = int(round(200 * scale))
    TOUCH_PX = max(1, int(round(scale)))


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
    ap.add_argument("--dataset", choices=["ufba", "dentex"], default="ufba")
    ap.add_argument("--labels", type=Path, help="dentex: train_quadrant_enumeration.json")
    ap.add_argument("--images-dir", type=Path, help="dentex: the quadrant_enumeration xrays folder")
    ap.add_argument("--run-tag", default="", help="model folder tag: '' (UFBA), _dentex or _dentex_tseed1")
    ap.add_argument("--work-width", type=int, default=640,
                    help="dentex: X-rays are resized to this width; 0 keeps each X-ray at its stored size and "
                         "scales the pixel constants by its own width / 640")
    ap.add_argument("--px-scale", type=float, default=1.0, help="dentex: factor on the UFBA pixel constants")
    ap.add_argument("--boxes", type=Path, default=None, help="position-only training boxes (default by dataset)")
    ap.add_argument("--size-bar", type=float, default=None, help="reported: j(closed) point estimate at least this")
    ap.add_argument("--chunk", type=int, default=0, help="altered images are built for this many X-rays of a fold "
                    "at a time (0: a whole fold); stored-size X-rays need about 40")
    ap.add_argument("--q1-warn-only", action="store_true", help="local tests only (scikit-learn differs from Kaggle's)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    st.MATCH_ORDER = "conf"
    dentex = a.dataset == "dentex"
    if dentex:  # pixel constants were set on UFBA, where a median tooth is 24 px wide; same here at 640 px wide
        set_constants(a.px_scale)
        print(f"dentex: width {a.work_width or 'stored size'}, dilate {gi.DILATE_PX} px, inpaint "
              f"{INPAINT_RADIUS} px, max shift {MAX_SHIFT} px, touch {TOUCH_PX} px", flush=True)

    t = pd.read_csv(a.per_tooth, dtype={"fdi": str}, **({"float_precision": "round_trip"} if dentex else {}))
    data = DentexData(a.labels, a.images_dir, a.work_width) if dentex else UfbaData(a.repo)
    all_ids = sorted(t["image_id"].unique())
    plan = draw_targets(t, all_ids, data, np.random.default_rng(0))  # UFBA: identical to Section 71
    ids = all_ids[::max(1, len(all_ids) // a.images)][:a.images] if a.images else all_ids
    if dentex:
        # The stored position-only answers were made from this table at full precision, in this row order
        # (dentex_external.py); dentex_boxes.csv is rounded to 6 decimals and ordered differently, which
        # changes about 6% of the answers (DENTEX_INTERVENTION_RULES.md amendment 2026-10-10).
        pos_boxes = t[["image_id", "fdi", "class_id", "x_center", "y_center", "width", "height"]]
        pos_folds = t.drop_duplicates("image_id")[["image_id", "fold"]]
    else:
        pos_boxes = pd.read_csv(a.boxes or a.repo / "cpu_repro/cv/boxes.csv", dtype={"fdi": str})
        pos_folds = pd.read_csv(a.folds or a.repo / "cpu_repro/cv/folds.csv")
    pos_models = position_models(pos_boxes, pos_folds)
    dup = set(map(tuple, t.loc[t.duplicated(["image_id", "fdi"], keep=False), ["image_id", "fdi"]].to_numpy()))

    # Altered images are built for a chunk of X-rays (one fold at a time, or --chunk X-rays of it), reused by
    # all three detectors, scored, and dropped, so memory does not grow with the number of targets. Rows are
    # put back in the order of an all-at-once run (detector, fold, target, condition) before they are written.
    fold_of = t.drop_duplicates("image_id").set_index("image_id")["fold"]
    chunks = []
    for f in sorted(fold_of.loc[ids].unique()):
        fids = [i for i in ids if fold_of[i] == f]
        step = a.chunk or len(fids)
        chunks += [fids[k:k + step] for k in range(0, len(fids), step)]
    rank = {i: k for k, i in enumerate(ids)}
    skipped, examples, n_jobs, n_open, n_agree = 0, 0, 0, 0, 0
    why = {"no polygon for N or M": 0, "T, N or M labeled twice in the X-ray": 0, "N never touches M": 0}
    n_conds = len(CONDITIONS) if a.mode == "slide" else len(TIP_CONDITIONS)
    rows, chk = [], {}
    for chunk in chunks:
        jobs = []
        for image_id in chunk:
            if not plan[image_id]:
                continue
            g = t[t["image_id"] == image_id].reset_index(drop=True)
            img = data.image(image_id)
            if dentex and not a.work_width:  # stored size: the 640-px geometry at this X-ray's own resolution
                set_constants(a.px_scale * img.shape[1] / 640)
            row_of = {fdi: i for i, fdi in enumerate(g["fdi"])}
            for k_t, T in enumerate(plan[image_id]):
                N, M, direction = neighbors_of(T)
                if f"{image_id}_{N}" not in data or f"{image_id}_{M}" not in data:
                    skipped += 1
                    why["no polygon for N or M"] += 1
                    continue
                if any((image_id, f) in dup for f in (T, N, M)):  # a repeated label leaves the box ambiguous
                    skipped += 1
                    why["T, N or M labeled twice in the X-ray"] += 1
                    continue
                t_region = dilate(data.mask(image_id, T, img.shape[:2])).astype(bool)
                n_mask = data.mask(image_id, N, img.shape[:2])
                m_mask = data.mask(image_id, M, img.shape[:2])
                n_region = dilate(n_mask).astype(bool)
                i_n = row_of[N]
                base = dict(image_id=image_id, fold=int(g.loc[0, "fold"]), target=T, mover=N, mesial=M,
                            target_class=int(g.loc[row_of[T], "class_id"]), mover_class=int(g.loc[i_n, "class_id"]),
                            g=g, row_of=row_of)
                first = len(jobs)
                if a.mode == "slide":
                    full = touch_shift(n_mask, m_mask, direction)
                    if full is None:
                        skipped += 1
                        why["N never touches M"] += 1
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
                        why["N never touches M"] += 1
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
                for k_c, j in enumerate(jobs[first:]):
                    j["order"] = (rank[image_id], k_t, k_c)
                if examples < N_EXAMPLES:
                    panels = [img] + [j["img"] for j in jobs[-n_conds:]]
                    if panels[0].shape[1] > 1600:  # stored-size X-rays: keep the example files small
                        f = 640 / panels[0].shape[1]
                        panels = [cv2.resize(p, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) for p in panels]
                    cv2.imwrite(str(a.out / f"example_{image_id}_{T}.png"), np.concatenate(panels, 1))
                    examples += 1
        n_jobs += len(jobs)

        # Position-only answer for the moved box.
        for j in jobs:
            x, y, w, h = j["mover_box"]
            f = pd.DataFrame([dict(x_center=x, y_center=y, width=w, height=h, area=w * h, aspect_ratio=w / h)])
            j["position_only_pred"] = int(pos_models[j["fold"]].predict(f[FEATURES])[0])
        # Check: on unmoved boxes the position-only answer must equal the stored one.
        same = [j["position_only_pred"] == j["g"].loc[j["row_of"][j["mover"]], "coord_pred"]
                for j in jobs if j["condition"] == "open"]
        n_open, n_agree = n_open + len(same), n_agree + sum(same)
        if dentex and not all(same) and not a.q1_warn_only:  # DENTEX_INTERVENTION_RULES.md Q1: hard stop
            raise SystemExit(f"Q1 failed: position-only agreement {n_agree / n_open:.4f} < 1.0000 "
                             f"after {n_open} unmoved boxes; not interpreted")

        for det in DETS:
            runs = [p for p in a.models.rglob(f"{det}_cvseed{a.cv_seed}{a.run_tag}")
                    if p.is_dir() and "/repo/" not in str(p)]
            assert len(runs) == 1, (det, runs)
            run = runs[0]
            for fold in sorted({j["fold"] for j in jobs}):
                fj = [j for j in jobs if j["fold"] == fold]
                model = st.Detector(det, run / f"fold{fold}")
                intact = {}
                for j in fj:
                    g, row_of = j["g"], j["row_of"]
                    gt = np.stack([g.x_center - g.width / 2, g.y_center - g.height / 2,
                                   g.x_center + g.width / 2, g.y_center + g.height / 2], 1)
                    if j["image_id"] not in intact:
                        intact[j["image_id"]] = st.match(gt, *model(data.image(j["image_id"])))
                        if f"{det}_pred" in g:  # intact answers against the stored ones (full-size X-rays for DENTEX)
                            now, was = intact[j["image_id"]], g[f"{det}_pred"].to_numpy(float)
                            c = chk.setdefault(det, np.zeros(4))
                            c += [len(g), ((now == was) | (np.isnan(now) & np.isnan(was))).sum(),
                                  (was == g["class_id"].to_numpy()).sum(), (now == g["class_id"].to_numpy()).sum()]
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
                                     pred=pred[row_of[j["mover"]]], _fold=fold, _order=j["order"]))
                print(det, "fold", fold, "done", len(fj), "altered images", flush=True)
        del jobs
    print(n_jobs // n_conds, "targets with a mover;", skipped, "skipped", flush=True)
    print("skipped by reason:", why, flush=True)
    pd.DataFrame([dict(drawn=sum(len(v) for k, v in plan.items() if k in set(ids)), movers=n_jobs // n_conds,
                       skipped=skipped, **why)]).to_csv(a.out / "flow.csv", index=False)
    print(f"position-only check on unmoved boxes: {n_agree / n_open:.4f} agree with the per-tooth file", flush=True)
    rows.sort(key=lambda r: (DETS.index(r["detector"]), r["_fold"], r["_order"]))
    for r in rows:
        del r["_fold"], r["_order"]
    d = pd.DataFrame(rows)
    d.to_csv(a.out / "closure_rows.csv", index=False)
    if chk:  # teeth of the X-rays scored here: working-size intact answer against the stored one
        c = pd.DataFrame({det: v for det, v in chk.items()}, index=["teeth", "same_answer", "stored_right", "now_right"]).T
        c["same_answer_pct"], c["stored_top1_pct"], c["now_top1_pct"] = (
            100 * c["same_answer"] / c["teeth"], 100 * c["stored_right"] / c["teeth"], 100 * c["now_right"] / c["teeth"])
        c.to_csv(a.out / "intact_check.csv", float_format="%.3f")
        print(c.round(2).to_string(), flush=True)

    summarize(d, a.out, list(CONDITIONS) if a.mode == "slide" else list(TIP_CONDITIONS), a.size_bar)


def summarize(d, out_dir, conds=tuple(CONDITIONS), size_bar=None):
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
        if measure == "all3" and size_bar is not None:  # DENTEX_INTERVENTION_RULES.md size bar (point estimate)
            r["size_bar"], r["size_met"] = size_bar, bool(r["closed"] >= size_bar)
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
