"""Phase 3 batch 3, CPU only (cpu_repro/cv/PHASE3_BATCH3_RULES.md, frozen 2026-10-01).

    python batch3.py --repo <repo checkout> --det yolov8x=<dir> --det rtdetr_l=<dir> \
        --det fasterrcnn=<dir> --out <dir> [--images N]

A. position-only model on each detector's matched predicted box;
B. tooth-order post-processor (Viterbi relabeling of each arch);
C. risk-coverage (AURC) and the look-alike review flag;
D. minimum detectable differences between detectors and between seeds.
All scoring is confidence-first (score_cv.MATCH_ORDER = "conf"), confidence
>= 0.5, IoU >= 0.5. --images N keeps the first N X-rays (smoke test only).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_cv as sc  # noqa: E402

DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]
N_BOOT = 10_000
EPS = 1e-3
GROUP_IOU = 0.5
LOOKALIKE_CONF = 0.1
Z = 1.959964 + 0.841621  # two-sided alpha 0.05, power 0.8
# Arch order as it appears left to right in the image (patient's right on the left;
# checked against boxes.csv: FDI 18 has the smallest mean x, 28 the largest).
FDI = sc.FDI
ORDER = {"upper": [FDI.index(f) for f in ["18", "17", "16", "15", "14", "13", "12", "11",
                                           "21", "22", "23", "24", "25", "26", "27", "28"]],
         "lower": [FDI.index(f) for f in ["48", "47", "46", "45", "44", "43", "42", "41",
                                           "31", "32", "33", "34", "35", "36", "37", "38"]]}
ARCH = {c: a for a, cs in ORDER.items() for c in cs}


def xyxy_of(teeth):
    g = teeth[["x_center", "y_center", "width", "height"]].to_numpy()
    return np.stack([g[:, 0] - g[:, 2] / 2, g[:, 1] - g[:, 3] / 2, g[:, 0] + g[:, 2] / 2, g[:, 1] + g[:, 3] / 2], 1)


def candidates(d):
    """Group one X-ray's raw detections into tooth candidates: in order of
    confidence, a detection joins the first candidate whose anchor (its
    highest-confidence detection) it overlaps at IoU >= GROUP_IOU.
    Returns anchors (rows of d), a 32-class score matrix and det -> candidate."""
    d = d.sort_values("conf", ascending=False)
    boxes = d[["x1", "y1", "x2", "y2"]].to_numpy()
    anchors, scores, member = [], [], {}
    for k, (idx, r) in enumerate(d.iterrows()):
        if anchors:
            m = sc.iou(boxes[k:k + 1], boxes[[a for a, _ in anchors]])[0]
            j = int(m.argmax())
            if m[j] >= GROUP_IOU:
                scores[j][int(r.class_id)] = max(scores[j][int(r.class_id)], r.conf)
                member[idx] = j
                continue
        anchors.append((k, idx))
        s = np.zeros(32)
        s[int(r.class_id)] = r.conf
        scores.append(s)
        member[idx] = len(anchors) - 1
    return d.loc[[idx for _, idx in anchors]], np.array(scores).reshape(-1, 32), member


def viterbi(score_rows, order):
    """One class per candidate (sorted left to right) maximizing the summed
    log score, with arch positions strictly increasing."""
    n, P = len(score_rows), len(order)
    cost = np.log(score_rows[:, order] + EPS)
    dp = np.full((n, P), -np.inf)
    back = np.zeros((n, P), int)
    dp[0] = cost[0]
    for i in range(1, n):
        best_prev = np.maximum.accumulate(dp[i - 1])
        arg_prev = np.array([int(np.argmax(dp[i - 1][:p + 1])) for p in range(P)])
        dp[i, 1:] = cost[i, 1:] + best_prev[:-1]
        back[i, 1:] = arg_prev[:-1]
    p = int(np.argmax(dp[-1]))
    path = [p]
    for i in range(n - 1, 0, -1):
        p = back[i, p]
        path.append(p)
    return [order[q] for q in path[::-1]]


def postprocess(raw):
    """Collapse-only and order-constrained detections, from raw detections."""
    collapse, ordered = [], []
    for image_id, d in raw.groupby("image_id"):
        anc, scores, _ = candidates(d)
        keep = anc["conf"].to_numpy() >= sc.CONF
        anc, scores = anc[keep], scores[keep]
        top = scores.argmax(1)
        c = anc.assign(class_id=top)
        collapse.append(c)
        o = c.copy()
        xc = ((anc["x1"] + anc["x2"]) / 2).to_numpy()
        for arch, order in ORDER.items():
            ix = np.where([ARCH[t] == arch for t in top])[0]
            if len(ix) == 0:
                continue
            if len(ix) > len(order):  # more candidates than positions: keep the most confident
                ix = ix[np.argsort(-anc["conf"].to_numpy()[ix])[:len(order)]]
            ix = ix[np.argsort(xc[ix])]
            o.iloc[ix, o.columns.get_loc("class_id")] = viterbi(scores[ix], order)
        ordered.append(o)
    return pd.concat(collapse), pd.concat(ordered)


def joint_mask(t, names):
    j = t["coord_pred"] != t["class_id"]
    for n in names:
        j &= (t[f"{n}_pred"] != t["class_id"]) & t[f"{n}_pred"].notna()
    p = np.stack([t[f"{n}_pred"].to_numpy() for n in names])
    return j & (p == p[0]).all(0)


def boot_paired(t, a, b, rng):
    """Top-1 of b minus a (pp) with a 95% X-ray bootstrap CI."""
    s = t.assign(n=1, _a=t[a].astype(int), _b=t[b].astype(int)).groupby("image_id")[["n", "_a", "_b"]].sum()
    s = s.to_numpy(float)
    stat = lambda u: 100 * (u[2] - u[1]) / u[0]
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    v = stat(s[idx].sum(1).T)
    return stat(s.sum(0)), *np.percentile(v, [2.5, 97.5]), float(np.std(v))


def aurc(conf, err, w=None):
    """Area under the risk-coverage curve; teeth sorted by confidence, highest first."""
    o = np.argsort(-conf, kind="stable")
    e, w = err[o], (np.ones_like(err) if w is None else w[o])
    cw = np.cumsum(w)
    keep = cw > 0
    return float(np.sum((w * np.cumsum(w * e) / np.where(keep, cw, 1))[keep]) / cw[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--det", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", type=int, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sc.MATCH_ORDER = "conf"
    rng = np.random.default_rng(0)
    det_dirs = dict(s.split("=", 1) for s in a.det)
    boxes = pd.read_csv(a.repo / "cpu_repro/cv/boxes.csv")
    folds = pd.read_csv(a.repo / "cpu_repro/cv/folds.csv")
    if a.images:
        keep = sorted(boxes["image_id"].unique())[:a.images]
        boxes = boxes[boxes["image_id"].isin(keep)]
        folds = folds[folds["image_id"].isin(keep)]
    t = sc.position_only(boxes, folds)
    t["coord_correct"] = t["coord_pred"] == t["class_id"]
    # Same fold models as score_cv.position_only (deterministic, random_state=0).
    b = boxes.merge(folds[["image_id", "fold"]], on="image_id")
    b = b.assign(area=b["width"] * b["height"], aspect_ratio=b["width"] / b["height"])
    coord = {f: HistGradientBoostingClassifier(random_state=0).fit(b.loc[b["fold"] != f, sc.FEATURES],
                                                                  b.loc[b["fold"] != f, "class_id"])
             for f in sorted(b["fold"].unique())}

    raw, cand = {}, {}
    for d in [x for x in DETS if x in det_dirs]:
        r = pd.concat([pd.read_csv(p) for p in sorted(Path(det_dirs[d]).glob("fold*/test_detections.csv"))])
        raw[d] = r[r["image_id"].isin(t["image_id"])].reset_index(drop=True)
        sc.add_detections(t, d, raw[d])
        col, ordd = postprocess(raw[d])
        sc.add_detections(t, f"{d}_collapse", col)
        sc.add_detections(t, f"{d}_order", ordd)
        cand[d] = {i: candidates(g) for i, g in raw[d].groupby("image_id")}
        print(d, "post-processed", flush=True)
    dets = list(raw)
    t["fold"] = t["fold"].astype(int)

    # ---- A. position-only on predicted boxes
    rows_a = []
    for d in dets:
        pred_box = np.full((len(t), 4), np.nan)
        for image_id, g in t.groupby("image_id"):
            _, _, ix = sc.match(g, raw[d][raw[d]["image_id"] == image_id], return_index=True)
            ok = ix != -1
            pred_box[g.index[ok]] = raw[d].loc[ix[ok].astype(int), ["x1", "y1", "x2", "y2"]].to_numpy()
        m = ~np.isnan(pred_box[:, 0])
        f = pd.DataFrame({"x_center": (pred_box[:, 0] + pred_box[:, 2]) / 2,
                          "y_center": (pred_box[:, 1] + pred_box[:, 3]) / 2,
                          "width": pred_box[:, 2] - pred_box[:, 0], "height": pred_box[:, 3] - pred_box[:, 1]})
        f = f.assign(area=f["width"] * f["height"], aspect_ratio=f["width"] / f["height"])
        cp = np.full(len(t), np.nan)
        for fo, clf in coord.items():
            sel = m & (t["fold"].to_numpy() == fo)
            if sel.any():
                cp[sel] = clf.predict(f.loc[sel, sc.FEATURES])
        t[f"coord_on_{d}_box"] = cp
        t[f"_cpb_{d}"] = cp == t["class_id"]
        mt = t[m]
        ch, lo, hi, _ = boot_paired(mt, "coord_correct", f"_cpb_{d}", rng)
        diff = abs(ch)
        rows_a.append(dict(detector=d, n_matched=int(m.sum()),
                           coord_on_label_box_matched_top1=100 * mt["coord_correct"].mean(),
                           coord_on_pred_box_matched_top1=100 * mt[f"_cpb_{d}"].mean(),
                           change_pp=ch, change_lo=lo, change_hi=hi,
                           coord_on_pred_box_all_teeth_top1=100 * t[f"_cpb_{d}"].mean(),
                           coord_on_label_box_all_teeth_top1=100 * t["coord_correct"].mean(),
                           gap_on_pred_box_pp=100 * (t[f"{d}_correct"].mean() - t[f"_cpb_{d}"].mean()),
                           gap_on_label_box_pp=100 * (t[f"{d}_correct"].mean() - t["coord_correct"].mean()),
                           rule=1 if diff < 2 else 2))
    A = pd.DataFrame(rows_a)

    # ---- B. tooth-order post-processor
    rows_b = []
    j_base = joint_mask(t, dets)
    for d in dets:
        others = [x for x in dets if x != d]
        for variant in ("collapse", "order"):
            ch, lo, hi, _ = boot_paired(t, f"{d}_correct", f"{d}_{variant}_correct", rng)
            tt = t.assign(**{f"{d}_pred": t[f"{d}_{variant}_pred"]})
            j_new = joint_mask(tt, [d] + others)
            rows_b.append(dict(detector=d, variant=variant, top1_base=100 * t[f"{d}_correct"].mean(),
                               top1_post=100 * t[f"{d}_{variant}_correct"].mean(), change_pp=ch,
                               change_lo=lo, change_hi=hi,
                               missed_base=100 * t[f"{d}_missed"].mean(),
                               missed_post=100 * t[f"{d}_{variant}_missed"].mean(),
                               joint_base=int(j_base.sum()), joint_post=int(j_new.sum()),
                               joint_change_pct=100 * (j_new.sum() - j_base.sum()) / max(j_base.sum(), 1)))
            r = rows_b[-1]
            if variant == "order":
                r["rule"] = (1 if ch >= 0.5 and lo > 0 and r["joint_change_pct"] <= -30
                             else 3 if ch <= -0.5 else 2 if abs(ch) < 0.5 else 0)  # 0: none of the three
    tt = t.assign(**{f"{d}_pred": t[f"{d}_order_pred"] for d in dets})
    j_all = joint_mask(tt, dets)
    rows_b.append(dict(detector="all three", variant="order", joint_base=int(j_base.sum()),
                       joint_post=int(j_all.sum()),
                       joint_change_pct=100 * (j_all.sum() - j_base.sum()) / max(j_base.sum(), 1)))
    B = pd.DataFrame(rows_b)

    # ---- C. risk-coverage and the look-alike flag
    rows_c = []
    img_codes, img_index = np.unique(t["image_id"], return_inverse=True)
    for d in dets:
        conf = t[f"{d}_conf"].fillna(-1).to_numpy()
        err = (~t[f"{d}_correct"]).to_numpy(float)
        base = aurc(conf, err)
        bs = []
        for _ in range(N_BOOT):
            w = np.bincount(rng.integers(0, len(img_codes), len(img_codes)), minlength=len(img_codes))[img_index]
            bs.append(aurc(conf, err, w.astype(float)))
        flag = np.zeros(len(t), bool)
        for image_id, g in t.groupby("image_id"):
            r = raw[d][raw[d]["image_id"] == image_id]
            _, _, ix = sc.match(g, r, return_index=True)
            if image_id not in cand[d]:
                continue
            _, scores, member = cand[d][image_id]
            for pos, (gi, di) in enumerate(zip(g.index, ix)):
                if di == -1:
                    continue
                pc = int(r.loc[di, "class_id"])
                q, n = FDI[pc][0], int(FDI[pc][1])
                adj = [FDI.index(f"{q}{n + s}") for s in (-1, 1) if 1 <= n + s <= 8]
                flag[gi] = (scores[member[di]][adj] >= LOOKALIKE_CONF).any()
        matched = ~t[f"{d}_missed"].to_numpy()
        n_flag = int(flag.sum())
        order = np.argsort(np.where(matched, conf, np.inf), kind="stable")
        conf_flag = np.zeros(len(t), bool)
        conf_flag[order[:n_flag]] = True
        jb = j_base.to_numpy()
        mis = (matched & ~t[f"{d}_correct"]).to_numpy()
        caught_l, caught_c = int((flag & jb).sum()), int((conf_flag & jb).sum())
        rows_c.append(dict(detector=d, aurc=base, aurc_lo=np.percentile(bs, 2.5), aurc_hi=np.percentile(bs, 97.5),
                           n_flagged=n_flag, flagged_pct=100 * n_flag / len(t), n_joint=int(jb.sum()),
                           joint_caught_lookalike=caught_l, joint_caught_conf=caught_c,
                           ratio=caught_l / caught_c if caught_c else np.inf,
                           misnumbered_caught_lookalike=int((flag & mis).sum()),
                           misnumbered_caught_conf=int((conf_flag & mis).sum()), n_misnumbered=int(mis.sum()),
                           rule=1 if (caught_c == 0 and caught_l > 0) or (caught_c and caught_l / caught_c >= 1.5)
                           else 2))
    C = pd.DataFrame(rows_c)

    # ---- D. minimum detectable differences
    rows_d = []
    se = {}
    for d in dets:
        _, _, _, s = boot_paired(t.assign(_z=False), "_z", f"{d}_correct", rng)
        se[d] = s
        rows_d.append(dict(comparison=f"{d}: seed 0 vs seed 1 (independent-split approximation)",
                           observed_pp=np.nan, se_pp=np.sqrt(2) * s, mde_pp=Z * np.sqrt(2) * s))
    for i, x in enumerate(dets):
        for y in dets[i + 1:]:
            ch, lo, hi, s = boot_paired(t, f"{x}_correct", f"{y}_correct", rng)
            rows_d.append(dict(comparison=f"{y} minus {x} (paired by X-ray)", observed_pp=ch, ci_lo=lo,
                               ci_hi=hi, se_pp=s, mde_pp=Z * s))
    D = pd.DataFrame(rows_d)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for name, df in [("A_position_on_pred_boxes", A), ("B_order_postprocessor", B),
                     ("C_risk_coverage_lookalike", C), ("D_power", D)]:
        df.to_csv(a.out / f"{name}.csv", index=False, float_format="%.4f")
        print(f"\n== {name}\n" + df.round(3).to_string(index=False), flush=True)
    keep = ["image_id", "fold", "fdi", "class_id", "coord_pred"] + [
        c for d in dets for c in (f"{d}_pred", f"{d}_conf", f"{d}_collapse_pred", f"{d}_order_pred",
                                  f"coord_on_{d}_box")]
    t[keep].to_csv(a.out / "batch3_per_tooth.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
