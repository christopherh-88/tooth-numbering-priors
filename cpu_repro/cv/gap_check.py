"""Missing-neighbor check (cpu_repro/cv/PHASE3_BATCH3_RULES.md, note
"missing-neighbor check", frozen 2026-10-01).

    python gap_check.py --per-tooth <confidence-first per_tooth_predictions.csv> --out <dir>

Uses only the per-tooth file: every labeled tooth is a row, so a position
with no row in an X-ray has no labeled tooth.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]
FDI = [f"{q}{t}" for q in "1234" for t in "12345678"]
ORDER = {"upper": ["18", "17", "16", "15", "14", "13", "12", "11", "21", "22", "23", "24", "25", "26", "27", "28"],
         "lower": ["48", "47", "46", "45", "44", "43", "42", "41", "31", "32", "33", "34", "35", "36", "37", "38"]}
NEIGHBORS = {f: [o[i + s] for s in (-1, 1) if 0 <= i + s < len(o)] for o in ORDER.values() for i, f in enumerate(o)}
N_BOOT = 10_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tooth", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    t = pd.read_csv(a.per_tooth, dtype={"fdi": str})
    present = t.groupby("image_id")["fdi"].agg(set).to_dict()
    t["gap"] = [any(n not in present[i] for n in NEIGHBORS[f]) for i, f in zip(t["image_id"], t["fdi"])]
    t["gap_no_m3"] = [any(n not in present[i] for n in NEIGHBORS[f] if n[1] != "8")
                      for i, f in zip(t["image_id"], t["fdi"])]
    ok = np.logical_and.reduce([t[f"{d}_pred"] == t["class_id"] for d in DETS])
    p = np.stack([t[f"{d}_pred"].to_numpy() for d in DETS])
    joint = (t["coord_pred"] != t["class_id"]) & (p == p[0]).all(0)
    for d in DETS:
        joint &= (t[f"{d}_pred"] != t["class_id"]) & t[f"{d}_pred"].notna()
    t["group"] = np.where(joint, "J", np.where(ok & (t["coord_pred"] == t["class_id"]), "C_coord_right",
                                               np.where(ok, "C_coord_wrong", "other")))
    t["in_C"] = ok
    t["in_J"] = joint
    t["in_Cw"] = ok & (t["coord_pred"] != t["class_id"])

    # Per X-ray sums for the bootstrap: count and gaps in J, C, Cw.
    cols = []
    for g in ("J", "C", "Cw"):
        for k in ("", "_gap", "_gapm3"):
            name = f"{g}{k}"
            base = t[f"in_{g}"]
            t[name] = (base & (t["gap"] if k == "_gap" else t["gap_no_m3"] if k == "_gapm3" else True)).astype(int)
            cols.append(name)
    s = t.groupby("image_id")[cols].sum()
    S = s.to_numpy(float)
    ix = {c: i for i, c in enumerate(cols)}
    idx = rng.integers(0, len(S), size=(N_BOOT, len(S)))
    B = S[idx].sum(1)
    tot = S.sum(0)

    def rate(v, g, k):
        return v[..., ix[f"{g}{k}"]] / v[..., ix[g]]

    rows = []
    for k, label in (("_gap", "all gaps"), ("_gapm3", "third-molar gaps ignored")):
        rJ, rC, rCw = rate(tot, "J", k), rate(tot, "C", k), rate(tot, "Cw", k)
        bJ, bC = rate(B, "J", k), rate(B, "C", k)
        diff = bJ - bC
        ratio = rJ / rC
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rule = 1 if ratio >= 2 and (lo > 0 or hi < 0) else (2 if ratio < 1.25 else 3)
        rows.append(dict(gaps=label, n_J=int(tot[ix["J"]]), r_J_pct=100 * rJ,
                         r_J_lo=100 * np.percentile(bJ, 2.5), r_J_hi=100 * np.percentile(bJ, 97.5),
                         n_C=int(tot[ix["C"]]), r_C_pct=100 * rC,
                         r_C_lo=100 * np.percentile(bC, 2.5), r_C_hi=100 * np.percentile(bC, 97.5),
                         diff_pp=100 * (rJ - rC), diff_lo=100 * lo, diff_hi=100 * hi, ratio=ratio,
                         ratio_lo=np.percentile(bJ / bC, 2.5), ratio_hi=np.percentile(bJ / bC, 97.5),
                         n_C_coord_wrong=int(tot[ix["Cw"]]), r_C_coord_wrong_pct=100 * rCw,
                         rule=rule if k == "_gap" else np.nan))
    J = t[t["in_J"]]
    shared = J["yolov8x_pred"].astype(int).map(lambda c: FDI[c])
    absent = [f not in present[i] for i, f in zip(J["image_id"], shared)]
    rows[0]["J_wrong_answer_is_absent_tooth_pct"] = 100 * np.mean(absent)
    rows[0]["J_wrong_answer_is_absent_tooth_n"] = int(np.sum(absent))
    out = pd.DataFrame(rows)
    out.to_csv(a.out / "gap_check_summary.csv", index=False, float_format="%.4f")
    J.assign(shared_fdi=shared.to_numpy(), wrong_answer_absent=absent)[
        ["image_id", "fdi", "shared_fdi", "gap", "gap_no_m3", "wrong_answer_absent"]].to_csv(
        a.out / "gap_check_joint.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).T.to_string())
    print(pd.crosstab(J["gap"], np.array(absent), rownames=["J has gap"], colnames=["wrong answer absent"]))


if __name__ == "__main__":
    main()
