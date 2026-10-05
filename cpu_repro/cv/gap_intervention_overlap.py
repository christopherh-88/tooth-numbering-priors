"""Follow-ups on the gap intervention, not in the frozen rule (RESULTS.md Section 71).

    python gap_intervention_overlap.py --rows <gap_intervention_rows.csv> --out <dir>

1. Overlap: among neighbors all three detectors number correctly on the
   intact image, how often each pair and all three give the neighbor the
   erased tooth's number, against the count expected if they were independent.
2. Fill rate by the erased tooth's type.
3. N2 errors: share that take the number of the tooth between N2 and the
   erased tooth (the whole run shifting one place).
Writes overlap.csv, by_type.csv and n2_shift.csv.
"""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from gap_check import ORDER

DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]
FDI = [f"{q}{t}" for q in "1234" for t in "12345678"]
POS = {f: (arch, i) for arch, o in ORDER.items() for i, f in enumerate(o)}
TYPE = {"1": "incisor", "2": "incisor", "3": "canine", "4": "premolar", "5": "premolar", "6": "molar", "7": "molar"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.rows, dtype={"target": str, "tooth": str})
    d = d[(d["intact_pred"] == d["class_id"]) & (d["condition"] == "removal")].copy()
    d["fill"] = d["pred"] == d["target_class"]
    n = d[d["role"] == "N"]

    f = n.pivot_table(index=["image_id", "target", "tooth"], columns="detector", values="fill",
                      aggfunc="first").dropna().astype(bool)
    rows = [dict(detectors=det, n=len(f), observed=int(f[det].sum()), expected_if_independent=float("nan"))
            for det in DETS]
    for k in (2, 3):
        for c in combinations(DETS, k):
            exp = len(f)
            for det in c:
                exp *= f[det].mean()
            rows.append(dict(detectors=" ".join(c), n=len(f), observed=int(f[list(c)].all(axis=1).sum()),
                             expected_if_independent=exp))
    ov = pd.DataFrame(rows)
    ov.to_csv(a.out / "overlap.csv", index=False, float_format="%.2f")

    bt = (n.assign(type=n["target"].str[1].map(TYPE)).groupby(["detector", "type"])["fill"]
          .agg(fill_pct=lambda x: 100 * x.mean(), n="size").reset_index())
    bt.to_csv(a.out / "by_type.csv", index=False, float_format="%.2f")

    n2 = d[d["role"] == "N2"].copy()
    n2["mid_class"] = [FDI.index(ORDER[POS[t][0]][(POS[t][1] + POS[s][1]) // 2])
                       for t, s in zip(n2["target"], n2["tooth"])]
    w = n2[n2["pred"].notna() & (n2["pred"] != n2["class_id"])]
    sh = (w.assign(takes_mid=w["pred"] == w["mid_class"]).groupby("detector")
          .agg(n2_scored=("takes_mid", "size"), takes_mid_pct=("takes_mid", lambda x: 100 * x.mean())).reset_index())
    sh["n2_scored"] = sh["detector"].map(n2.groupby("detector").size())
    sh["n2_wrong"] = sh["detector"].map(w.groupby("detector").size())
    sh.to_csv(a.out / "n2_shift.csv", index=False, float_format="%.2f")

    pd.set_option("display.width", 200)
    for x in (ov, bt, sh):
        print(x.round(2).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
