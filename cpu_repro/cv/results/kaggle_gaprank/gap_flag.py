"""Gap flag against the confidence flag (cpu_repro/cv/GAP_FLAG_RULES.md, frozen 2026-10-06).

    python gap_flag.py --repo <repo checkout> --folds <folds csv> --seed N \
        --det yolov8x=<dir> --det rtdetr_l=<dir> --det fasterrcnn=<dir> --out <dir> [--images N]

Flags, from each detector's own output, the teeth next to a position with
no detection and the teeth sharing a number with another detection, and
compares how many joint failures that catches with the same number of
least-confident teeth. Writes gap_flag_seed{N}.csv. --images N keeps the
first N X-rays (smoke test only).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch3 as b3  # noqa: E402
import score_cv as sc  # noqa: E402

N_BOOT = 10_000
FDI = sc.FDI
# Arch neighbors in image order (11 and 21 are neighbors, as are 41 and 31).
NEIGHBORS = {c: [o[i + s] for s in (-1, 1) if 0 <= i + s < len(o)] for o in b3.ORDER.values() for i, c in enumerate(o)}
GAP_POSITIONS = {"no_third_molars": [c for c in range(32) if FDI[c][1] != "8"], "with_third_molars": list(range(32))}


def candidate_flags(cls, gaps):
    """Per candidate: (gap-adjacent, duplicate), from one X-ray's candidate classes."""
    present = set(cls)
    missing = [c for c in gaps if c not in present]
    adj = {n for m in missing for n in NEIGHBORS[m]}
    counts = pd.Series(cls).value_counts()
    return np.array([c in adj for c in cls]), np.array([counts[c] > 1 for c in cls])


def recall_ci(t, a, b, joint, rng):
    """Joint-failure recall of flag a and flag b (%) and a minus b with a 95% X-ray bootstrap CI."""
    s = (t.assign(_n=joint, _a=a & joint, _b=b & joint).groupby("image_id")[["_n", "_a", "_b"]].sum()
         .to_numpy(float))
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    u = s[idx].sum(1)
    d = 100 * (u[:, 1] - u[:, 2]) / u[:, 0]
    tot = s.sum(0)
    return 100 * tot[1] / tot[0], 100 * tot[2] / tot[0], 100 * (tot[1] - tot[2]) / tot[0], *np.percentile(d, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--folds", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--det", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", type=int, default=None)
    a = ap.parse_args()
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

    raw = {}
    for d in b3.DETS:
        r = pd.concat([pd.read_csv(p) for p in sorted(Path(det_dirs[d]).glob("fold*/test_detections.csv"))])
        raw[d] = r[r["image_id"].isin(t["image_id"])].reset_index(drop=True)
        sc.add_detections(t, d, raw[d])
    joint = b3.joint_mask(t, b3.DETS).to_numpy()
    print(f"seed {a.seed}: {int(joint.sum())} joint failures", flush=True)

    rows = []
    for d in b3.DETS:
        flags = {k: np.zeros(len(t), bool) for k in ("gap", "dup", "gap_m3")}
        for image_id, g in t.groupby("image_id"):
            r = raw[d][raw[d]["image_id"] == image_id]
            if r.empty:
                continue
            anc, scores, member = b3.candidates(r)
            cls = scores.argmax(1)
            keep = anc["conf"].to_numpy() >= sc.CONF
            kept = np.where(keep)[0]
            gap, dup = candidate_flags(list(cls[kept]), GAP_POSITIONS["no_third_molars"])
            gap_m3, _ = candidate_flags(list(cls[kept]), GAP_POSITIONS["with_third_molars"])
            pos = {j: k for k, j in enumerate(kept)}
            _, _, ix = sc.match(g, r, return_index=True)
            for gi, di in zip(g.index, ix):
                if di == -1 or member[int(di)] not in pos:  # match returns row labels of r
                    continue
                k = pos[member[int(di)]]
                flags["gap"][gi], flags["dup"][gi], flags["gap_m3"][gi] = gap[k], dup[k], gap_m3[k]
        matched = ~t[f"{d}_missed"].to_numpy()
        mis = matched & ~t[f"{d}_correct"].to_numpy()
        conf = t[f"{d}_conf"].fillna(-1).to_numpy()
        order = np.argsort(np.where(matched, conf, np.inf), kind="stable")
        variants = {"flag": flags["gap"] | flags["dup"], "gap_only": flags["gap"], "dup_only": flags["dup"],
                    "flag_with_third_molars": flags["gap_m3"] | flags["dup"]}
        for name, f in variants.items():
            f = f & matched
            n = int(f.sum())
            cflag = np.zeros(len(t), bool)
            cflag[order[:n]] = True
            rf, rc, diff, lo, hi = recall_ci(t, f, cflag, joint, rng)
            rows.append(dict(seed=a.seed, detector=d, variant=name, n_joint=int(joint.sum()), n_flagged=n,
                             cost_pct=100 * n / int(matched.sum()), joint_recall_flag=rf, joint_recall_conf=rc,
                             diff_pp=diff, diff_lo=lo, diff_hi=hi, ratio=rf / rc if rc else np.inf,
                             misnumbered_recall_flag=100 * (f & mis).sum() / max(mis.sum(), 1),
                             misnumbered_recall_conf=100 * (cflag & mis).sum() / max(mis.sum(), 1)))
        print(d, "done", flush=True)
    out = pd.DataFrame(rows)
    main_rows = out["variant"] == "flag"
    out.loc[main_rows, "useful"] = (out.loc[main_rows, "ratio"] >= 1.5) & (out.loc[main_rows, "diff_lo"] > 0)
    out.loc[main_rows, "practical"] = out.loc[main_rows, "cost_pct"] <= 15
    out.to_csv(a.out / f"gap_flag_seed{a.seed}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(out.round(2).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
