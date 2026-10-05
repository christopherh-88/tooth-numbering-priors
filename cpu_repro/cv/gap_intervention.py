"""Gap intervention (cpu_repro/cv/GAP_INTERVENTION_RULES.md, frozen 2026-10-01).

For each sampled tooth T: inpaint T's own mask (dilated 3 px, OpenCV Telea)
and check whether a neighbor N that the detector numbered correctly on the
intact image now gets T's number. Controls: the teeth two positions away
(N2) under the same removal, and N under a sham inpaint of T's mask shape
over bone just beyond T's root. Runs in a Kaggle GPU kernel (see
kaggle/make_phase3_kernels.py):

    python gap_intervention.py --repo <repo checkout> --models <dir with *_cvseed0/> \
        --per-tooth <confidence-first per_tooth_predictions.csv> --out <dir> [--images N]

Writes gap_intervention_rows.csv (one row per detector, condition and
scored tooth), gap_intervention_summary.csv (rates, bootstrap CIs, rule)
and a few example images (intact, removal, sham) for a visual check.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import shift_test as st
from gap_check import NEIGHBORS, ORDER

DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]
N_TARGETS = 3
DILATE_PX = 3
INPAINT_RADIUS = 5
SHAM_MARGIN_PX = 5
N_BOOT = 10_000
N_EXAMPLES = 6
POS = {f: (arch, i) for arch, o in ORDER.items() for i, f in enumerate(o)}


def two_away(f):
    arch, i = POS[f]
    o = ORDER[arch]
    return [o[i + s] for s in (-2, 2) if 0 <= i + s < len(o)]


def load_mask(path, shape):
    import cv2
    from PIL import Image
    m = (np.squeeze(np.array(Image.open(path))) > 0).astype(np.uint8)
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


def dilate(m):
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE_PX + 1, 2 * DILATE_PX + 1))
    return cv2.dilate(m, k)


def sham_mask(m, upper, teeth_union):
    """T's mask moved past its root (up for the upper arch, down for the lower),
    with any pixel on a labeled tooth removed."""
    ys = np.nonzero(m)[0]
    shift = (ys.max() - ys.min() + 1 + SHAM_MARGIN_PX) * (-1 if upper else 1)
    out = np.zeros_like(m)
    h = m.shape[0]
    if abs(shift) >= h:
        return out
    src = slice(max(0, -shift), min(h, h - shift))
    dst = slice(max(0, shift), min(h, h + shift))
    out[dst] = m[src]
    return out & (1 - teeth_union)


def boot(df, rng, pairs):
    """Per-X-ray sums, resampled. pairs: name -> (numerator col, denominator col).
    Returns point estimates and 95% CIs for each rate and for g minus each control."""
    cols = sorted({c for p in pairs.values() for c in p})
    s = df.groupby("image_id")[cols].sum()
    v = s.to_numpy(float)
    ix = {c: i for i, c in enumerate(cols)}
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    tot, bt = v.sum(0), v[idx].sum(1)
    rate = lambda u, k: 100 * u[..., ix[pairs[k][0]]] / u[..., ix[pairs[k][1]]]
    out = {}
    for k in pairs:
        out[k] = rate(tot, k)
        out[f"{k}_lo"], out[f"{k}_hi"] = np.nanpercentile(rate(bt, k), [2.5, 97.5])
    for k in pairs:
        if k == "g":
            continue
        out[f"g_minus_{k}"] = out["g"] - out[k]
        out[f"g_minus_{k}_lo"], out[f"g_minus_{k}_hi"] = np.nanpercentile(rate(bt, "g") - rate(bt, k), [2.5, 97.5])
    return out


def rule(r):
    if r["g"] < 1:
        return 2
    if r["g"] >= 5 and r["g_minus_n2_lo"] > 0 and r["g_minus_sham_lo"] > 0:
        return 1
    return 3


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--per-tooth", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", type=int, default=0, help="smoke test on the first N X-rays")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    st.MATCH_ORDER = "conf"
    rng = np.random.default_rng(0)

    t = pd.read_csv(a.per_tooth, dtype={"fdi": str})
    img_dir = a.repo / "Dataset/bb_u_net_dataset/panoramic_x_rays"
    mask_path = {p.name.replace(".ome.tiff", ""): p
                 for p in (a.repo / "Dataset/bb_u_net_dataset/labels").glob("*/*.ome.tiff")}
    ids = sorted(t["image_id"].unique())
    if a.images:
        ids = ids[:a.images]

    # Targets, drawn once with seed 0 and shared by all detectors.
    plan = {}
    for image_id in ids:
        g = t[t["image_id"] == image_id]
        present = set(g["fdi"])
        ok = sorted(f for f in present if f[1] != "8" and len(NEIGHBORS[f]) == 2
                    and all(n in present for n in NEIGHBORS[f]) and f"{image_id}_{f}" in mask_path)
        plan[image_id] = list(rng.choice(ok, size=min(N_TARGETS, len(ok)), replace=False)) if ok else []
    print("targets:", sum(map(len, plan.values())), "in", sum(map(bool, plan.values())), "X-rays", flush=True)

    rows, examples = [], 0
    for det in DETS:
        run = next(a.models.rglob(f"{det}_cvseed0"))
        for f in range(5):
            model = st.Detector(det, run / f"fold{f}")
            for image_id in [i for i in ids if plan[i] and (t.loc[t["image_id"] == i, "fold"] == f).all()]:
                g = t[t["image_id"] == image_id].reset_index(drop=True)
                img = cv2.imread(str(img_dir / f"{image_id}.jpg"))
                gt = np.stack([g.x_center - g.width / 2, g.y_center - g.height / 2,
                               g.x_center + g.width / 2, g.y_center + g.height / 2], 1)
                row_of = {fdi: i for i, fdi in enumerate(g["fdi"])}
                intact = st.match(gt, *model(img))
                masks = {fdi: load_mask(mask_path[f"{image_id}_{fdi}"], img.shape[:2])
                         for fdi in g["fdi"] if f"{image_id}_{fdi}" in mask_path}
                union = np.clip(sum(masks.values()), 0, 1).astype(np.uint8)
                for T in plan[image_id]:
                    m = dilate(masks[T])
                    sham = sham_mask(m, T[0] in "12", union)
                    removed = cv2.inpaint(img, m, INPAINT_RADIUS, cv2.INPAINT_TELEA)
                    shammed = cv2.inpaint(img, sham, INPAINT_RADIUS, cv2.INPAINT_TELEA)
                    keep = [i for i in range(len(g)) if g.fdi[i] != T]
                    after = np.full(len(g), np.nan)
                    after[keep] = st.match(gt[keep], *model(removed))
                    after_sham = st.match(gt, *model(shammed))
                    t_class = g.class_id[row_of[T]]
                    for role, scored in (("N", NEIGHBORS[T]), ("N2", two_away(T))):
                        for n in scored:
                            if n not in row_of:
                                continue
                            i = row_of[n]
                            base = dict(detector=det, image_id=image_id, fold=f, target=T, target_class=t_class,
                                        tooth=n, class_id=g.class_id[i], role=role, intact_pred=intact[i],
                                        sham_px=int(sham.sum()), removed_px=int(m.sum()))
                            rows.append(base | dict(condition="removal", pred=after[i]))
                            if role == "N":
                                rows.append(base | dict(condition="sham", pred=after_sham[i]))
                    if det == DETS[0] and examples < N_EXAMPLES:
                        cv2.imwrite(str(a.out / f"example_{image_id}_{T}.png"),
                                    np.concatenate([img, removed, shammed], 1))
                        examples += 1
            print(det, "fold", f, "done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(a.out / "gap_intervention_rows.csv", index=False)

    d = d[d["intact_pred"] == d["class_id"]].copy()  # scored only if right on the intact image
    d["fill"] = (d["pred"] == d["target_class"]).astype(int)
    d["n"] = 1
    summary = []
    for det, x in d.groupby("detector"):
        sel = {"g": (x.role == "N") & (x.condition == "removal"),
               "n2": (x.role == "N2") & (x.condition == "removal"),
               "sham": (x.role == "N") & (x.condition == "sham")}
        w = x[["image_id"]].copy()
        for k, s in sel.items():
            w[f"{k}_fill"] = (x["fill"] * s).astype(int)
            w[f"{k}_n"] = s.astype(int)
        r = dict(detector=det, n_targets=x.loc[sel["g"], ["image_id", "target"]].drop_duplicates().shape[0],
                 n_neighbors=int(sel["g"].sum()), n_n2=int(sel["n2"].sum()), n_sham=int(sel["sham"].sum()))
        r |= boot(w, rng, {k: (f"{k}_fill", f"{k}_n") for k in sel})
        for k, s in sel.items():
            y = x[s]
            r[f"{k}_correct_pct"] = 100 * (y.pred == y.class_id).mean()
            r[f"{k}_missed_pct"] = 100 * y.pred.isna().mean()
            r[f"{k}_other_wrong_pct"] = 100 * (y.pred.notna() & (y.pred != y.class_id)
                                               & (y.pred != y.target_class)).mean()
        r["rule"] = rule(r)
        summary.append(r)
    out = pd.DataFrame(summary)
    out.to_csv(a.out / "gap_intervention_summary.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(out.round(2).T.to_string(), flush=True)


if __name__ == "__main__":
    main()
