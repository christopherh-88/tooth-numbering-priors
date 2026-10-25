"""Re-derive the post hoc numbers of RESULTS.md Section 92 (stored size against 640 px) from the
rows files, with the definition of closure_intervention.py's summary (RESULTS.md Section 95).

    python post_hoc_two_detector.py --out results/post_hoc_two_detector

A mover is a tooth every listed detector numbers right on the intact image (the 'open' rows'
intact_pred). j is the share of movers to which every listed detector gives the erased tooth's
number in the full-closure condition ('closed'). Missed detections count as not filling.
Writes post_hoc_two_detector.csv.
"""
import argparse
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent / "results"
DETS3 = ["yolov8x", "rtdetr_l", "fasterrcnn"]
DETS2 = ["yolov8x", "rtdetr_l"]
RUNS = {("slide", "stored"): "kaggle_dentex_closure_s0n/dentex-closure-s0n", ("slide", "640"): "kaggle_dentex_closure_s0/dentex-closure-s0",
        ("tip", "stored"): "kaggle_dentex_tip_s0n/dentex-tip-s0n", ("tip", "640"): "kaggle_dentex_tip_s0/dentex-tip-s0",
        ("slide", "ufba"): "kaggle_closure/closure_intervention", ("tip", "ufba"): "kaggle_tip/closure_tip"}
K = ["image_id", "target"]


def movers_and_fills(path, dets):
    d = pd.read_csv(R / path / "closure_rows.csv")
    d = d[d.detector.isin(dets)]
    o = d[d.condition == "open"].assign(r=lambda x: x.intact_pred == x.mover_class).groupby(K).r.agg(
        lambda s: bool(s.all()) and len(s) == len(dets))
    movers = set(o[o].index)
    c = d[d.condition == "closed"].assign(f=lambda x: x.pred == x.target_class).groupby(K).f.agg(
        lambda s: bool(s.all()) and len(s) == len(dets))
    return movers, {i for i in c[c].index if i in movers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, res = [], {}
    for (mode, size), p in RUNS.items():
        for label, dets in (("all3", DETS3), ("two", DETS2)):
            mv, fl = movers_and_fills(p, dets)
            res[(mode, size, label)] = (mv, fl)
            rows.append(dict(mode=mode, size=size, detectors=label, movers=len(mv), fills=len(fl),
                             j_pct=100 * len(fl) / len(mv), movers_at_both_sizes=float("nan"), j_at_both_pct=float("nan")))
    out = pd.DataFrame(rows)
    for mode in ("slide", "tip"):  # all three detectors, movers scored at both sizes
        both = res[(mode, "stored", "all3")][0] & res[(mode, "640", "all3")][0]
        for size in ("stored", "640"):
            j = 100 * len(res[(mode, size, "all3")][1] & both) / len(both)
            i = out[(out["mode"] == mode) & (out["size"] == size) & (out.detectors == "all3")].index[0]
            out.loc[i, ["movers_at_both_sizes", "j_at_both_pct"]] = [len(both), j]
    out.to_csv(a.out / "post_hoc_two_detector.csv", index=False, float_format="%.4f")
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
