"""External test on DENTEX (cpu_repro/cv/DENTEX_EXTERNAL_RULES.md, frozen 2026-10-06).

    python dentex_external.py --labels <train_quadrant_enumeration.json> --images <xrays dir> \
        --models <dir holding {det}_cvseed0/> --ufba-boxes <boxes.csv> --ufba-folds <folds.csv> \
        --out <dir> [--images-limit N]

Runs the seed 0 UFBA fold models, unchanged, on the DENTEX quadrant_enumeration
X-rays (each X-ray to one fold, fixed random assignment), saves every detection
at confidence >= 0.05 in the train_cv.py format, then scores as score_cv.py
(confidence-first). Position-only: five-fold CV within DENTEX (primary) and the
UFBA-trained fold model (also reported). Writes dentex_teeth.csv,
dentex_detections.csv, dentex_summary.csv and dentex_gap_flag.csv.
--images-limit N keeps the first N X-rays (smoke test only).

    python dentex_external.py --labels <json> --saved <dir> --ufba-boxes ... --ufba-folds ... --out <dir>

--saved scores detections saved by train_cv.py --dataset (models trained
within DENTEX, DENTEX_INDOMAIN_RULES.md) instead of running the UFBA models:
<dir> holds, at any depth, {det}_cvseed0_dentex/fold{0..4}/test_detections.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch3 as b3  # noqa: E402
import gap_flag as gf  # noqa: E402
import score_cv as sc  # noqa: E402
import shift_test as st  # noqa: E402
from closure_intervention import position_models  # noqa: E402
from gap_check import NEIGHBORS  # noqa: E402

DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]
N_BOOT = 10_000
UFBA_POS_MATCH_CI = (85.71, 96.00)  # Section 65, seed 0


def load_teeth(path):
    """One row per labeled tooth; FDI looked up by category name, cast to str (CONVENTIONS.md)."""
    d = json.load(open(path))
    imgs = {im["id"]: im for im in d["images"]}
    q = {c["id"]: str(c["name"]) for c in d["categories_1"]}
    p = {c["id"]: str(c["name"]) for c in d["categories_2"]}
    rows = []
    for a in d["annotations"]:
        im = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        fdi = q[a["category_id_1"]] + p[a["category_id_2"]]
        rows.append(dict(image_id=Path(im["file_name"]).stem, fdi=fdi, class_id=sc.FDI.index(fdi),
                         x_center=(x + w / 2) / im["width"], y_center=(y + h / 2) / im["height"],
                         width=w / im["width"], height=h / im["height"]))
    return pd.DataFrame(rows)


def boot(t, cols, stat, rng):
    s = t.assign(_n=1).groupby("image_id")[["_n"] + cols].sum().to_numpy(float)
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    v = stat(s[idx].sum(1).T)
    return stat(s.sum(0)), *np.nanpercentile(v, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--images", type=Path)
    ap.add_argument("--models", type=Path)
    ap.add_argument("--saved", type=Path, help="score saved within-DENTEX detections, no inference")
    ap.add_argument("--saved-tag", default="_dentex", help="output-dir tag of the saved runs (DENTEX_TSEED_RULES.md: _dentex_tseed1)")
    ap.add_argument("--ufba-boxes", type=Path, required=True)
    ap.add_argument("--ufba-folds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images-limit", type=int, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sc.MATCH_ORDER = st.MATCH_ORDER = "conf"
    rng = np.random.default_rng(0)

    t = load_teeth(a.labels)
    ids = sorted(t["image_id"].unique())
    if a.images_limit:
        ids = ids[:a.images_limit]
        t = t[t["image_id"].isin(ids)]
    perm = np.random.default_rng(0).permutation(len(ids))
    fold_of = {ids[k]: int(r % 5) for r, k in enumerate(perm)}
    t = t.assign(fold=t["image_id"].map(fold_of)).sort_values(["image_id", "class_id"]).reset_index(drop=True)
    t["tid"] = np.arange(len(t))
    print(f"{len(ids)} X-rays, {len(t)} teeth", flush=True)

    if a.saved:
        parts = []
        for d in DETS:
            runs = [p for p in a.saved.rglob(f"{d}_cvseed0{a.saved_tag}") if p.is_dir() and "/repo/" not in str(p)]
            assert len(runs) == 1, (d, runs)
            files = sorted(runs[0].glob("fold*/test_detections.csv"))
            assert len(files) == 5, (d, files)
            parts.append(pd.concat([pd.read_csv(f) for f in files]).assign(detector=d))
        dets = pd.concat(parts)[["detector", "image_id", "class_id", "conf", "x1", "y1", "x2", "y2"]]
        dets = dets[dets["image_id"].isin(ids)].reset_index(drop=True)
        for d in DETS:  # every X-ray tested exactly once, by its own fold's models
            seen = dets.loc[dets["detector"] == d, "image_id"].unique()
            assert len(seen) >= 0.99 * len(ids), (d, len(seen))
    # ---- inference (GPU), saved in the train_cv.py detection format
    import cv2
    det_rows = []
    for d in ([] if a.saved else DETS):
        runs = [p for p in a.models.rglob(f"{d}_cvseed0") if p.is_dir() and "/repo/" not in str(p)]
        assert len(runs) == 1, (d, runs)
        for f in range(5):
            model = st.Detector(d, runs[0] / f"fold{f}")
            for i in [i for i in ids if fold_of[i] == f]:
                img = cv2.imread(str(a.images / f"{i}.png"))
                assert img is not None, i
                cls, conf, xyxy = model(img)
                det_rows += [dict(detector=d, image_id=i, class_id=int(c), conf=float(s), x1=b[0], y1=b[1],
                                  x2=b[2], y2=b[3]) for c, s, b in zip(cls, conf, xyxy)]
            del model
            print(d, "fold", f, "done", flush=True)
    if not a.saved:
        dets = pd.DataFrame(det_rows)
    dets.to_csv(a.out / "dentex_detections.csv", index=False)

    # ---- scoring
    for d in DETS:
        sc.add_detections(t, d, dets.loc[dets["detector"] == d].drop(columns="detector"))
    cv = sc.position_only(t.drop(columns="fold"), pd.DataFrame({"image_id": ids, "fold": [fold_of[i] for i in ids]}))
    t = t.merge(cv[["tid", "coord_pred"]], on="tid", how="left", validate="1:1")
    assert t["coord_pred"].notna().all()
    ub = position_models(pd.read_csv(a.ufba_boxes), pd.read_csv(a.ufba_folds))
    feats = t.assign(area=t["width"] * t["height"], aspect_ratio=t["width"] / t["height"])[sc.FEATURES]
    t["coord_ufba_pred"] = np.nan
    for f, clf in ub.items():
        m = t["fold"] == f
        t.loc[m, "coord_ufba_pred"] = clf.predict(feats[m])
    t["coord_correct"] = t["coord_pred"] == t["class_id"]
    t["coord_ufba_correct"] = t["coord_ufba_pred"] == t["class_id"]

    present = t.groupby("image_id")["fdi"].agg(set).to_dict()
    t["gap"] = [any(n not in present[i] for n in NEIGHBORS[f]) for i, f in zip(t["image_id"], t["fdi"])]
    joint = b3.joint_mask(t, DETS)
    shared = t["yolov8x_pred"]
    t["joint"] = joint
    t["pos_match"] = joint & (t["coord_pred"] == shared)
    t["all_right"] = np.logical_and.reduce([t[f"{d}_correct"] for d in DETS])
    t["joint_gap"] = joint & t["gap"]
    t["right_gap"] = t["all_right"] & t["gap"]
    t["joint_gap_missing_number"] = t["joint_gap"] & [sc.FDI[int(c)] not in present[i] if c == c else False
                                                      for i, c in zip(t["image_id"], shared)]
    t.to_csv(a.out / "dentex_teeth.csv", index=False)

    rows = []

    def add(metric, cols, stat, **kw):
        p, lo, hi = boot(t, cols, stat, rng)
        rows.append(dict(metric=metric, point=p, lo=lo, hi=hi, **kw))

    for c in ["coord_correct", "coord_ufba_correct"] + [f"{d}_correct" for d in DETS] + [f"{d}_missed" for d in DETS]:
        t[f"_{c}"] = t[c].astype(int)
        add(f"pct_{c}", [f"_{c}"], lambda u: 100 * u[1] / u[0])
    add("floor_rule1", ["_coord_correct"], lambda u: 100 * u[1] / u[0])
    rows[-1]["pass"] = rows[-1]["lo"] >= 68.5
    for d in DETS:
        add(f"margin_{d}", [f"_{d}_correct", "_coord_correct"], lambda u: 100 * (u[1] - u[2]) / u[0])
        rows[-1]["pass"] = rows[-1]["lo"] >= (10 if a.saved else 5)  # DENTEX_INDOMAIN_RULES.md raises it to 10
    for c in ["joint", "pos_match", "all_right", "joint_gap", "right_gap", "joint_gap_missing_number"]:
        t[f"_{c}"] = t[c].astype(int)
    n_joint = int(joint.sum())
    add("pos_match_rule3", ["_joint", "_pos_match"], lambda u: 100 * u[2] / u[1], n=n_joint)
    rows[-1]["pass"] = (rows[-1]["point"] >= UFBA_POS_MATCH_CI[0]) if n_joint >= 50 else "underpowered"
    add("gap_ratio_rule4", ["_joint", "_joint_gap", "_all_right", "_right_gap"],
        lambda u: (u[2] / u[1]) / (u[4] / u[3]), n=n_joint)
    rows[-1]["pass"] = (rows[-1]["point"] >= 4.2 and rows[-1]["lo"] > 3) if n_joint >= 50 else "underpowered"
    add("joint_with_gap_pct", ["_joint", "_joint_gap"], lambda u: 100 * u[2] / u[1])
    add("all_right_with_gap_pct", ["_all_right", "_right_gap"], lambda u: 100 * u[2] / u[1])
    add("gap_joint_missing_number_pct", ["_joint_gap", "_joint_gap_missing_number"], lambda u: 100 * u[2] / u[1])
    S = pd.DataFrame(rows)
    S.to_csv(a.out / "dentex_summary.csv", index=False)

    # ---- Section 84 gap flag on DENTEX (no rule)
    fr = []
    for d in DETS:
        raw = dets.loc[dets["detector"] == d].drop(columns="detector").reset_index(drop=True)
        flag = np.zeros(len(t), bool)
        for image_id, g in t.groupby("image_id"):
            r = raw[raw["image_id"] == image_id]
            if r.empty:
                continue
            anc, scores, member = b3.candidates(r)
            kept = np.where(anc["conf"].to_numpy() >= sc.CONF)[0]
            cls = scores.argmax(1)
            gap, dup = gf.candidate_flags(list(cls[kept]), gf.GAP_POSITIONS["no_third_molars"])
            pos = {j: k for k, j in enumerate(kept)}
            _, _, ix = sc.match(g, r, return_index=True)
            for gi, di in zip(g.index, ix):
                if di != -1 and member[int(di)] in pos:
                    k = pos[member[int(di)]]
                    flag[gi] = gap[k] or dup[k]
        matched = ~t[f"{d}_missed"].to_numpy()
        flag &= matched
        order = np.argsort(np.where(matched, t[f"{d}_conf"].fillna(-1).to_numpy(), np.inf), kind="stable")
        cflag = np.zeros(len(t), bool)
        cflag[order[:int(flag.sum())]] = True
        rf, rc, diff, lo, hi = gf.recall_ci(t, flag, cflag, joint.to_numpy(), rng)
        fr.append(dict(detector=d, cost_pct=100 * flag.sum() / matched.sum(), joint_recall_flag=rf,
                       joint_recall_conf=rc, diff_pp=diff, diff_lo=lo, diff_hi=hi))
    pd.DataFrame(fr).to_csv(a.out / "dentex_gap_flag.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(S.round(2).to_string(index=False))
    print(pd.DataFrame(fr).round(2).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
