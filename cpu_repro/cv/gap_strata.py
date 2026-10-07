"""Gap ratio by number of missing teeth (cpu_repro/cv/GAP_STRATA_RULES.md). Reported only.

    python gap_strata.py --ufba <confmatch per_tooth_predictions.csv> \
        --dentex <Section 86 dentex_teeth.csv> --dentex-ext <Section 85 dentex_teeth.csv> --out <dir>

Writes gap_strata.csv (per dataset and stratum) and gap_strata_standardized.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gap_check import NEIGHBORS  # noqa: E402

DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]
STRATA = [(0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 99, "6+")]
POS17 = [f"{q}{t}" for q in "1234" for t in "1234567"]
N_BOOT = 10_000


def prepare(t):
    t = t.copy()
    t["fdi"] = t["fdi"].astype(str)
    present = t.groupby("image_id")["fdi"].agg(set).to_dict()
    t["gap"] = [any(n not in present[i] for n in NEIGHBORS[f]) for i, f in zip(t["image_id"], t["fdi"])]
    p = np.stack([t[f"{d}_pred"].to_numpy(float) for d in DETS])
    wrong = np.logical_and.reduce([~np.isnan(p[k]) & (p[k] != t["class_id"]) for k in range(3)])
    t["J"] = wrong & (p == p[0]).all(0) & (t["coord_pred"] != t["class_id"]).to_numpy()
    t["C"] = np.logical_and.reduce([p[k] == t["class_id"] for k in range(3)])
    n_missing = {i: sum(f not in s for f in POS17) for i, s in present.items()}
    m = t["image_id"].map(n_missing)
    t["stratum"] = ""
    for lo, hi, name in STRATA:
        t.loc[(m >= lo) & (m <= hi), "stratum"] = name
    return t


def per_xray(t):
    """Per X-ray and stratum: J, J at gap, C, C at gap (each X-ray has one stratum)."""
    t = t.assign(Jg=t["J"] & t["gap"], Cg=t["C"] & t["gap"])
    return t.groupby(["image_id", "stratum"])[["J", "Jg", "C", "Cg"]].sum().reset_index()


def standardized(x, wJ, wC):
    g = x.groupby("stratum")[["J", "Jg", "C", "Cg"]].sum().reindex([s[2] for s in STRATA]).fillna(0)
    pJ = (g["Jg"] / g["J"]).where(g["J"] > 0)
    pC = (g["Cg"] / g["C"]).where(g["C"] > 0)
    keep = pJ.notna() & pC.notna()
    sJ = (pJ[keep] * wJ[keep]).sum() / wJ[keep].sum()
    sC = (pC[keep] * wC[keep]).sum() / wC[keep].sum()
    return sJ, sC, sJ / sC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufba", type=Path, required=True)
    ap.add_argument("--dentex", type=Path, required=True)
    ap.add_argument("--dentex-ext", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    data = {"UFBA": prepare(pd.read_csv(a.ufba)), "DENTEX within": prepare(pd.read_csv(a.dentex)),
            "DENTEX external": prepare(pd.read_csv(a.dentex_ext))}
    assert int(data["UFBA"]["J"].sum()) == 161, data["UFBA"]["J"].sum()
    assert int(data["DENTEX within"]["J"].sum()) == 303
    assert int(data["DENTEX external"]["J"].sum()) == 160

    rows = []
    for name, t in data.items():
        x = per_xray(t)
        for _, _, s in STRATA + [(0, 0, "all")]:
            g = x if s == "all" else x[x["stratum"] == s]
            J, Jg, C, Cg = g[["J", "Jg", "C", "Cg"]].sum()
            rows.append(dict(dataset=name, stratum=s, xrays=len(g), shared_errors=int(J),
                             shared_at_gap_pct=100 * Jg / J if J else np.nan, all_right=int(C),
                             all_right_at_gap_pct=100 * Cg / C if C else np.nan,
                             ratio=(Jg / J) / (Cg / C) if J and C and Cg else np.nan))
    S = pd.DataFrame(rows)
    S.to_csv(a.out / "gap_strata.csv", index=False)

    u = per_xray(data["UFBA"]).groupby("stratum")[["J", "C"]].sum().reindex([s[2] for s in STRATA]).fillna(0)
    wJ, wC = u["J"], u["C"]
    out = []
    for name in ("DENTEX within", "DENTEX external"):
        x = per_xray(data[name])
        sJ, sC, r = standardized(x, wJ, wC)
        k = x["stratum"].map({s[2]: i for i, s in enumerate(STRATA)}).to_numpy()
        v = x[["J", "Jg", "C", "Cg"]].to_numpy(float)
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = rng.integers(0, len(x), len(x))
            g = np.stack([np.bincount(k[pick], v[pick, c], minlength=len(STRATA)) for c in range(4)], 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                pJ, pC = g[:, 1] / g[:, 0], g[:, 3] / g[:, 2]
            ok = ~np.isnan(pJ) & ~np.isnan(pC)
            boots[b] = ((pJ[ok] * wJ.to_numpy()[ok]).sum() / wJ.to_numpy()[ok].sum()) / (
                (pC[ok] * wC.to_numpy()[ok]).sum() / wC.to_numpy()[ok].sum())
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        out.append(dict(dataset=name, std_shared_at_gap_pct=100 * sJ, std_all_right_at_gap_pct=100 * sC,
                        std_ratio=r, lo=lo, hi=hi))
    T = pd.DataFrame(out)
    T.to_csv(a.out / "gap_strata_standardized.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.round(2).to_string(index=False))
    print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
