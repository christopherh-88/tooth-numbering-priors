"""Closure-ranked review flag (cpu_repro/cv/GAP_RANK_RULES.md, approved 2026-10-25).

    python gap_rank.py --repo <repo checkout> --folds <folds csv> --seed N \
        --det yolov8x=<dir> --det rtdetr_l=<dir> --det fasterrcnn=<dir> --out <dir> \
        [--images N] [--expect-joint K]
    python gap_rank.py --grade <dir holding gap_rank_seed{0,1,2}.csv>

Takes the Section 84 flag (gap_flag.py: teeth numbered next to a position the
detector left empty, plus teeth sharing a number) and ranks the flagged teeth by
the estimated space at the empty position: the horizontal distance between the
boxes of its two arch neighbors over the mean of their widths, from the
detector's own output only. Flagged teeth with no computable space come last,
ties are broken by x position then image id, never by confidence. Compares, at a
review budget of B% of the matched teeth, the joint-failure recall of the ranked
flag with the least-confident teeth and with the unranked flag cut to the same
count (its expected recall). Writes gap_rank_seed{N}.csv. --images N keeps the
first N X-rays (smoke test only).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch3 as b3  # noqa: E402
import gap_flag as gf  # noqa: E402
import score_cv as sc  # noqa: E402

N_BOOT = 10_000
N_BOOT_AUC = 1_000
MAIN_BUDGET = 4.0
BUDGETS = (2.0, 4.0, 8.0)  # percent of the detector's matched teeth; 4 is the graded one
RATIO_BAR = 1.5
NEIGHBORS = gf.NEIGHBORS
GAPS = gf.GAP_POSITIONS["no_third_molars"]


def x_space(a, b):
    """Horizontal space between two boxes (x1, y1, x2, y2) over their mean width; 0 if they touch or overlap."""
    gap = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    return gap / ((a[2] - a[0] + b[2] - b[0]) / 2)


def candidate_ranking(cls, box):
    """Per candidate (in confidence order): gap-adjacent, duplicate, estimated space (inf if none),
    and the empty position that gave the smallest space (-1 if none)."""
    first = {}
    for k, c in enumerate(cls):
        first.setdefault(c, k)  # most confident candidate of each number
    adj = np.zeros(len(cls), bool)
    space = np.full(len(cls), np.inf)
    pos = np.full(len(cls), -1)
    for m in GAPS:
        if m in first:
            continue
        nb = NEIGHBORS[m]
        s = x_space(box[first[nb[0]]], box[first[nb[1]]]) if len(nb) == 2 and all(n in first for n in nb) else np.inf
        for k, c in enumerate(cls):
            if c in nb:
                adj[k] = True
                if s < space[k]:
                    space[k], pos[k] = s, m
    dup = np.bincount(cls, minlength=32)[cls] > 1
    return adj, dup, space, pos


def rank_order(idx, space, xkey, image):
    """Flagged teeth idx sorted by estimated space, then x position, then image id."""
    return idx[np.lexsort((image[idx], xkey[idx], space[idx]))]


def sums(t, joint, sel):
    """Per X-ray counts: joint failures, and joint failures inside each selection (a dict of bool arrays)."""
    cols = {"j": joint}
    cols.update({k: v & joint for k, v in sel.items()})
    return pd.DataFrame(cols).assign(image_id=t["image_id"].to_numpy()).groupby("image_id").sum().to_numpy(float)


def boot_idx(rng, n):
    return rng.integers(0, n, size=(N_BOOT, n))


def auroc(score, y):
    r = rankdata(score)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def auroc_ci(score, y, image, rng):
    codes = pd.factorize(image)[0]
    groups = np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])
    vals = []
    for _ in range(N_BOOT_AUC):
        pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        vals.append(auroc(score[pick], y[pick]))
    return np.nanpercentile(vals, [2.5, 97.5])


def run_detector(t, d, raw):
    """Per labeled tooth: flag (Section 84), estimated space, empty position, candidate x center."""
    n = len(t)
    flag, space = np.zeros(n, bool), np.full(n, np.inf)
    pos, xkey = np.full(n, -1), np.full(n, np.inf)
    for image_id, g in t.groupby("image_id"):
        r = raw[raw["image_id"] == image_id]
        if r.empty:
            continue
        anc, scores, member = b3.candidates(r)
        cls = scores.argmax(1)
        kept = np.where(anc["conf"].to_numpy() >= sc.CONF)[0]
        box = anc[["x1", "y1", "x2", "y2"]].to_numpy()[kept]
        adj, dup, sp, ps = candidate_ranking(cls[kept], box)
        a84, d84 = gf.candidate_flags(list(cls[kept]), GAPS)
        assert (adj == a84).all() and (dup == d84).all(), "flag differs from Section 84"
        where = {j: k for k, j in enumerate(kept)}
        _, _, ix = sc.match(g, r, return_index=True)
        for gi, di in zip(g.index, ix):
            if di == -1 or member[int(di)] not in where:
                continue
            k = where[member[int(di)]]
            flag[gi] = adj[k] or dup[k]
            space[gi], pos[gi] = sp[k], ps[k]
            xkey[gi] = (box[k][0] + box[k][2]) / 2
    return flag, space, pos, xkey


def true_space(t, idx, pos):
    """Space from the labeled boxes at each flagged tooth's empty position (inf if not computable).

    Not a valid ceiling (RESULTS.md Section 98, found after the run): for a joint failure whose wrong number is
    the missing tooth's, one neighbor of the empty position has no labeled box, so the space is undefined.
    """
    lab = {}
    for im, c, xc, w in zip(t["image_id"], t["class_id"], t["x_center"], t["width"]):
        lab[(im, int(c))] = (xc - w / 2, 0.0, xc + w / 2, 1.0)
    out = np.full(len(t), np.inf)
    for i in idx:
        if pos[i] < 0:
            continue
        nb = NEIGHBORS[pos[i]]
        keys = [(t["image_id"].iat[i], n) for n in nb]
        if len(nb) == 2 and all(k in lab for k in keys):
            out[i] = x_space(lab[keys[0]], lab[keys[1]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path)
    ap.add_argument("--folds", type=Path)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--det", action="append")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--images", type=int, default=None)
    ap.add_argument("--expect-joint", type=int, default=None)
    ap.add_argument("--grade", type=Path, default=None)
    a = ap.parse_args()
    if a.grade:
        return grade(a.grade)
    a.out.mkdir(parents=True, exist_ok=True)
    sc.MATCH_ORDER = "conf"
    rng = np.random.default_rng(0)
    det_dirs = dict(s.split("=", 1) for s in a.det)
    boxes = pd.read_csv(a.repo / "cpu_repro/cv/boxes.csv")
    folds = pd.read_csv(a.folds)
    if a.images:
        keep = sorted(boxes["image_id"].unique())[:a.images]
        boxes, folds = boxes[boxes["image_id"].isin(keep)], folds[folds["image_id"].isin(keep)]
    t = sc.position_only(boxes, folds)
    names = list(det_dirs)
    raw = {}
    for d in names:
        r = pd.concat([pd.read_csv(p) for p in sorted(Path(det_dirs[d]).glob("fold*/test_detections.csv"))])
        raw[d] = r[r["image_id"].isin(t["image_id"])].reset_index(drop=True)
        sc.add_detections(t, d, raw[d])
    joint = b3.joint_mask(t, names).to_numpy()
    print(f"seed {a.seed}: {int(joint.sum())} joint failures", flush=True)
    if a.expect_joint is not None:
        assert int(joint.sum()) == a.expect_joint, (int(joint.sum()), a.expect_joint)
    image = t["image_id"].to_numpy()

    rows = []
    for d in names:
        flag, space, pos, xkey = run_detector(t, d, raw[d])
        matched = ~t[f"{d}_missed"].to_numpy()
        flag &= matched
        idx = np.where(flag)[0]
        conf = t[f"{d}_conf"].fillna(-1).to_numpy()
        by_conf = np.argsort(np.where(matched, conf, np.inf), kind="stable")
        ranked_order = rank_order(idx, space, xkey, image)
        ceiling_order = rank_order(idx, true_space(t, idx, pos), xkey, image)
        y = joint[idx].astype(int)
        au = auroc(-space[idx], y)
        au_lo, au_hi = auroc_ci(-space[idx], y, image[idx], rng)
        for pct in BUDGETS:
            nb = min(int(round(pct / 100 * matched.sum())), len(idx))
            sel = {k: np.zeros(len(t), bool) for k in ("rank", "conf", "flag", "ceiling")}
            sel["rank"][ranked_order[:nb]] = True
            sel["conf"][by_conf[:nb]] = True
            sel["flag"][idx] = True
            sel["ceiling"][ceiling_order[:nb]] = True
            s = sums(t, joint, sel)  # columns: j, rank, conf, flag, ceiling
            frac = nb / len(idx)
            u = s[rng.integers(0, len(s), size=(N_BOOT, len(s)))].sum(1)
            tot = s.sum(0)
            rec = lambda x, j: 100 * x / j
            ci = lambda v: np.percentile(v, [2.5, 97.5])
            d_rc = rec(u[:, 1] - u[:, 2], u[:, 0])
            d_rt = rec(u[:, 1] - u[:, 3] * frac, u[:, 0])
            rank_rec, conf_rec, trunc_rec = rec(tot[1], tot[0]), rec(tot[2], tot[0]), rec(tot[3] * frac, tot[0])
            rc_lo, rc_hi = ci(d_rc)
            rt_lo, rt_hi = ci(d_rt)
            ratio_rc = tot[1] / tot[2] if tot[2] else np.inf
            ratio_rt = tot[1] / (tot[3] * frac) if tot[3] else np.inf
            rows.append(dict(
                seed=a.seed, detector=d, budget_pct=pct, n_review=nb, n_flag=len(idx), n_joint=int(tot[0]),
                recall_ranked=rank_rec, recall_conf=conf_rec, recall_trunc=trunc_rec,
                recall_full_flag=rec(tot[3], tot[0]), recall_ceiling=rec(tot[4], tot[0]),
                diff_vs_conf=rank_rec - conf_rec, diff_vs_conf_lo=rc_lo, diff_vs_conf_hi=rc_hi, ratio_vs_conf=ratio_rc,
                diff_vs_trunc=rank_rec - trunc_rec, diff_vs_trunc_lo=rt_lo, diff_vs_trunc_hi=rt_hi,
                ratio_vs_trunc=ratio_rt, precision_ranked=100 * tot[1] / max(nb, 1),
                auroc_flagged=au, auroc_lo=au_lo, auroc_hi=au_hi,
                pass_rule1=bool(ratio_rc >= RATIO_BAR and rc_lo > 0),
                pass_rule2=bool(ratio_rt >= RATIO_BAR and rt_lo > 0)))
        print(d, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(a.out / f"gap_rank_seed{a.seed}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(out.round(2).to_string(index=False), flush=True)


def grade(folder):
    df = pd.concat([pd.read_csv(p) for p in sorted(folder.glob("gap_rank_seed*.csv"))])
    df = df[df["budget_pct"] == MAIN_BUDGET]
    ok = {}
    for seed, g in df.groupby("seed"):
        r1, r2 = int(g["pass_rule1"].sum()), int(g["pass_rule2"].sum())
        both = int((g["pass_rule1"] & g["pass_rule2"]).sum())
        ok[seed] = both >= 2
        print(f"seed {seed}: rule 1 on {r1} of {len(g)} detectors, rule 2 on {r2}, both on {both}", flush=True)
    r1 = ok.get(0, False)
    print(f"GRADED rule 1 and 2 (seed 0, same >= 2 of 3 detectors): {'met' if r1 else 'not met'}")
    r3 = r1 and ok.get(1, False) and ok.get(2, False)
    print(f"GRADED rule 3 (also seeds 1 and 2): {'met' if r3 else 'not met'}")


if __name__ == "__main__":
    main()
