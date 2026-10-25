"""Check the numbers in paper/miccai/latex/main.tex against the result files.

    python paper/miccai/check_numbers.py

Each check recomputes a number from a CSV under cpu_repro/cv/results (or from the
per-tooth table), rounds it as the paper does, and requires that the same text
appears in main.tex. Exits with status 1 if any check fails. Numbers that exist
only in RESULTS.md prose are not covered here.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "cpu_repro" / "cv" / "results"
TEX = (ROOT / "paper/miccai/latex/main.tex").read_text()
TEXT = re.sub(r"\s+", " ", TEX.replace("\\%", "%").replace("~", " ").replace("\\,", " ").replace("$", ""))
DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]
fails, n_ok = [], 0


def check(label, got, expected, text=None, nd=1):
    """got rounded to nd decimals must equal expected; text must be in the paper."""
    global n_ok
    ok = True
    if expected is not None:
        ok = abs(round(float(got), nd) - float(expected)) < 1e-9
    if text is not None and re.sub(r"\s+", " ", text) not in TEXT:
        ok = False
        label += f" [text not in paper: {text!r}]"
    if ok:
        n_ok += 1
    else:
        fails.append(f"{label}: computed {got!r}, paper {expected!r}")


def in_paper(label, text):
    global n_ok
    if re.sub(r"\s+", " ", text) in TEXT:
        n_ok += 1
    else:
        fails.append(f"{label}: not in paper: {text!r}")


def csv(p):
    return pd.read_csv(R / p)


# ---- Table 1 and the abstract: from the per-tooth table and the score files
pt = pd.read_csv(R / "kaggle_confmatch/confmatch/score/per_tooth_predictions.csv")
check("teeth", len(pt), 11602, "11,602 teeth", 0)
check("xrays", pt.image_id.nunique(), 425, "425 uncropped X-rays", 0)
for d in DETS:
    s = csv(f"kaggle_f1/f1/{d}_score.csv").set_index("metric")
    check(f"{d} top1 recomputed", 100 * (pt[f"{d}_pred"] == pt.class_id).mean(), round(s.loc["top1", "value"], 1))
pos = csv("kaggle_f1/f1/yolov8x_score.csv").set_index("metric")
check("position-only top1 recomputed", 100 * (pt.coord_pred == pt.class_id).mean(), 78.5, "78.5 (77.0 to 80.0)")
# the position-only CI comes from the benchmark scorer's bootstrap (kaggle_f1), which is what benchmark/score.py reproduces
check("position-only CI lo", pos.loc["position_only_top1", "ci_lo"], 77.0)
check("position-only CI hi", pos.loc["position_only_top1", "ci_hi"], 80.0)
names = {"yolov8x": "YOLOv8x", "rtdetr_l": "RT-DETR-l", "fasterrcnn": "Faster R-CNN"}
splits = {}
for tag, f_ in ((1, "kaggle_seed1_3det/seed1_grade.csv"), (2, "kaggle_seed2_3det/seed2_grade.csv")):
    g_ = pd.read_csv(R / f_)
    for d in DETS:
        splits[(tag, d)] = g_[(g_.seed == tag) & (g_.metric == f"gap_pp_{d}")].point.iloc[0]
for d in DETS:
    s_ = csv(f"kaggle_f1/f1/{d}_score.csv").set_index("metric")
    t, g, f1_ = s_.loc["top1"], s_.loc["gap_pp"], s_.loc["f1_numbered"]
    r1 = lambda x: f"{x:.1f}"
    row = (f"{names[d]} & {r1(t.value)} ({r1(t.ci_lo)} to {r1(t.ci_hi)}) & {r1(g.value)} ({r1(g.ci_lo)} to {r1(g.ci_hi)}) "
           f"& {r1(f1_.value)} & {r1(splits[(1, d)])}, {r1(splits[(2, d)])}")
    in_paper(f"Table 1 row {d}", row)
jf = csv("kaggle_confmatch/cv_joint_failures.csv").iloc[0]
check("same-wrong teeth", jf.n_same_wrong, 161, "161 teeth (1.4%) are joint failures", 0)
check("joint share", 100 * 161 / 11602, 1.4)
check("position match", jf.position_match_pct, 91.3, "In 91.3% of them", 1)
check("null p95", jf.null_p95_pct, 6.8, "at most 6.8%")
check("neighbor pct", jf.neighbor_pct, 94.4, "In 94.4% the wrong number")
cal = csv("kaggle_confmatch/calibration_joint_failures.csv")
check("AUROC min", cal.auroc.min(), 0.53, "AUROC 0.53 to 0.66", 2)
check("AUROC max", cal.auroc.max(), 0.66, None, 2)
lc = csv("kaggle_confmatch/label_check_summary.csv").iloc[0]
check("label error share", lc.r_J_pct, 10.0, "About 10% of joint failures may be label", 0)
check("position match without label errors", lc.position_match_pct_after_excluding, 91.7, "position match is 91.7%")

# independent recomputation of the joint set from the per-tooth table
P = np.stack([pt[f"{d}_pred"].to_numpy(float) for d in DETS])
cls = pt.class_id.to_numpy()
wrong = np.logical_and.reduce([~np.isnan(P[k]) & (P[k] != cls) for k in range(3)]) & (pt.coord_pred.to_numpy() != cls)
J = wrong & (P == P[0]).all(0)
check("joint set recomputed", J.sum(), 161, None, 0)
check("position match recomputed", 100 * (pt.coord_pred.to_numpy()[J] == P[0][J]).mean(), 91.3)
cls2fdi = dict(zip(pt.class_id, pt.fdi.astype(str)))
nb = [(cls2fdi[int(a)][0] == cls2fdi[int(b)][0]) and abs(int(cls2fdi[int(a)][1]) - int(cls2fdi[int(b)][1])) == 1
      for a, b in zip(cls[J], P[0][J])]
check("neighbor recomputed", 100 * np.mean(nb), 94.4)

# ---- predicted boxes, power, order
A = csv("kaggle_batch3/A_position_on_pred_boxes.csv")
check("pred-box baseline min", A.coord_on_pred_box_matched_top1.min(), 79.0, "(79.0 to 80.2%)")
check("pred-box baseline max", A.coord_on_pred_box_matched_top1.max(), 80.2)
check("pred-box gap min", A.gap_on_pred_box_pp.min(), 14.5, "14.5 to 16.2 points")
check("pred-box gap max", A.gap_on_pred_box_pp.max(), 16.2)
D = csv("kaggle_batch3/D_power.csv")
r = D[D.comparison.str.startswith("rtdetr_l minus yolov8x")].iloc[0]
check("tie diff", r.observed_pp, 0.17, "difference 0.17 points", 2)
check("MDE", r.mde_pp, 0.50, "is 0.50", 2)
f1 = {d: csv(f"kaggle_f1/f1/{d}_score.csv").set_index("metric").loc["f1_numbered", "value"] for d in DETS}
check("F1 gap yolo vs rtdetr", f1["yolov8x"] - f1["rtdetr_l"], 2.7, "lowers its F1 by 2.7 points")

# unmatched (extra) boxes per X-ray = false positives minus misnumbered teeth, per X-ray
ex = {}
for d in DETS:
    s_ = csv(f"kaggle_f1/f1/{d}_score.csv").set_index("metric")
    ex[d] = s_.loc["false_pos_per_xray", "value"] - s_.loc["misnumbered", "value"] / 100 * 11602 / 425
check("extra boxes RT-DETR-l", ex["rtdetr_l"], 1.9, "about 1.9 unmatched boxes per X-ray against 0.4")
check("extra boxes YOLOv8x", ex["yolov8x"], 0.4)

# ---- per-number gaps
pf = pd.read_csv(ROOT / "benchmark/per_fdi.csv")
for d in DETS:
    pf[f"gap_{d}"] = pf[f"{d}_top1"] - pf.position_only_top1
check("detectors beat position on all numbers", int((pf[[f"gap_{d}" for d in DETS]] > 0).all().all()), 1, "on all 32 tooth numbers", 0)
check("32 numbers", len(pf), 32, None, 0)
tm = pf[pf.type == "third_molar"] if "third_molar" in set(pf.type) else pf[pf.fdi.astype(int) % 10 == 8]
check("third molar gap min", tm[[f"gap_{d}" for d in DETS]].min().min(), 4.3, "4.3 to 5.8")
check("third molar smallest per detector max", max(tm[f"gap_{d}"].min() for d in DETS), 5.8)
check("largest gap", pf[[f"gap_{d}" for d in DETS]].max().max(), 25.5, "up to 25.5")

# ---- shift and masking
sh = csv("kaggle_shift_v2/shift_summary.csv")
big = sh[(sh.dx.abs() == 0.10) & (sh.dy == 0)]
check("shift change below 0.3", float(big.change_pp.abs().max() < 0.3), 1.0, "by under 0.3 points", 0)
check("position-only after shift max", big.coord_top1_shifted.max(), 8.6, "under 9% of teeth")
cx = csv("kaggle_context/context_summary.csv")
k1 = cx[cx.k == 1.0].set_index("detector")
check("mask k=1 FRCNN", k1.loc["fasterrcnn", "control_change_pp"], -26.0, "about 26 points", 1)
check("mask k=1 RT-DETR", k1.loc["rtdetr_l", "control_change_pp"], -25.9)
check("mask k=1 YOLO", k1.loc["yolov8x", "control_change_pp"], -9.6, "YOLOv8x 9.6")
det = {d: k1.loc[d, "joint_still_shared_wrong_pct"] / (100 - k1.loc[d, "joint_missed_pct"]) * 100 for d in DETS}
check("still shared among detected, min", min(det.values()), 74, "74 to 93%", 0)
check("still shared among detected, max", max(det.values()), 93, None, 0)

# ---- missing neighbors
g = csv("kaggle_gapcheck/gap_check_summary.csv").iloc[0]
check("J missing neighbor", g.r_J_pct, 44.7, "44.7% of joint failures have a missing")
check("C missing neighbor", g.r_C_pct, 8.5, "against 8.5% of teeth")
check("ratio", g.ratio, 5.3, "ratio 5.3, CI 4.2 to 6.6")
check("ratio lo", g.ratio_lo, 4.2)
check("ratio hi", g.ratio_hi, 6.6)
check("wrong answer is the absent tooth", g.J_wrong_answer_is_absent_tooth_n, 67, "In 67 of the 72", 0)
check("pos-only-wrong missing neighbor", g.r_C_coord_wrong_pct, 10.5, "10.5% of the time")

# ---- space closure
dr = csv("kaggle_drift/drift/drift_summary.csv")
j = dr[(dr.measure == "S") & (dr.outcome == "joint")].iloc[0]
check("slots", j.n_slots, 372, "372 missing teeth", 0)
check("AUROC joint", j.auroc, 0.78, "AUROC 0.78, CI 0.70 to 0.84", 2)
check("AUROC lo", j.auroc_lo, 0.70, None, 2)
check("AUROC hi", j.auroc_hi, 0.84, None, 2)
check("median S unfilled", j.median_unfilled, 0.32, "against 0.32", 2)
slots = csv("kaggle_drift/drift/drift_slots.csv")
f = slots[slots.filled_joint]
check("filled at touching", 100 * (f.S == 0).mean(), 56, "56% of these errors sit at gaps whose neighbors touch", 0)
check("unfilled at touching", 100 * (slots[~slots.filled_joint].S == 0).mean(), 20, "against 20% of other gaps", 0)
oth = dr[(dr.measure == "S") & dr.outcome.isin(DETS)].auroc
check("detector AUROC min", oth.min(), 0.74, "AUROC 0.74 to 0.79", 2)
check("detector AUROC max", oth.max(), 0.79, None, 2)
check("position-only AUROC", dr[(dr.measure == "S") & (dr.outcome == "position_only")].auroc.iloc[0], 0.68, "(0.68)", 2)
check("review rule coverage", j.closed_pct_filled, 97, "covers 97% of the shared errors", 0)
check("review rule precision", j.filled_pct_among_closed, 13, "(precision 13%)", 0)
near = slots[slots.S <= 0.1]
check("22% of near-touching gaps hold a joint failure", 100 * near.filled_joint.mean(), 22, "22% hold a joint failure", 0)
check("about 11% per neighbor", 100 * near.filled_joint.mean() / 2, 11, "about 11% per", 0)
check("near-touching gaps", len(near), 121, None, 0)

# ---- erasure and closure
gi = csv("kaggle_gap_intervention/gap_intervention/gap_intervention_summary.csv").set_index("detector")
check("erasure YOLO", gi.loc["yolov8x", "g"], 2.0, "2.0% (YOLOv8x)")
check("erasure RT-DETR", gi.loc["rtdetr_l", "g"], 4.4, "4.4% (RT-DETR-l)")
check("erasure FRCNN", gi.loc["fasterrcnn", "g"], 5.7, "5.7% (Faster R-CNN)")
check("controls max", max(gi.n2.max(), gi.sham.max()), 0.2, "against 0 to 0.2% for both controls")
ov = csv("kaggle_gap_intervention/gap_intervention/overlap.csv")
o3 = ov[ov.detectors == "yolov8x rtdetr_l fasterrcnn"].iloc[0]
check("all three fill same neighbor", o3.observed, 4, "in only 4 of 2,260 cases", 0)
check("... share", 100 * o3.observed / o3.n, 0.18, "(0.18%)", 2)

cl = csv("kaggle_closure/closure_intervention/closure_summary.csv").set_index("measure").loc["all3"]
check("closure movers", cl.n_movers, 1114, "Of 1,114 moved teeth", 0)
check("open", cl.open, 0.18, "for 0.18% with the space open", 2)
check("half", cl.half, 2.9, "2.9% when moved halfway")
check("closed", cl.closed, 20.8, "20.8% when moved all the way")
check("closed minus open", cl.closed_minus_open, 20.6, "(20.6 points above open, CI 18.2 to 23.1)")
check("diff lo", cl.closed_minus_open_lo, 18.2)
check("diff hi", cl.closed_minus_open_hi, 23.1)
cs = csv("kaggle_closure/closure_intervention/closure_summary.csv").set_index("measure")
fills = [cs.loc[f"fill_{d}", "closed"] for d in DETS]
check("per detector fill min", min(fills), 38, "38 to 42%", 0)
check("per detector fill max", max(fills), 42, None, 0)
c1 = pd.read_csv(R / "kaggle_closure_s1/closure_s1/closure_summary.csv").set_index("measure").loc["all3"]
c2 = pd.read_csv(R / "kaggle_closure_s2/closure_s2/closure_summary.csv").set_index("measure").loc["all3"]
check("closure seed 1", c1.closed, 22.5, "(22.5% and 21.8% at full closure")
check("closure seed 2", c2.closed, 21.8)
check("open max on replications", max(c1.open, c2.open), 0.1, "at most 0.1% open")
tp = csv("kaggle_tip/closure_tip/closure_summary.csv").set_index("measure").loc["all3"]
check("tip", tp.closed, 9.4, "gives 9.4% (CI 7.7")
check("tip lo", tp.closed_lo, 7.7)
check("tip hi", tp.closed_hi, 11.0, "to 11.0;")
tp1 = pd.read_csv(R / "kaggle_tip_s1/closure_tip_s1/closure_summary.csv").set_index("measure").loc["all3"]
check("tip seed 1", tp1.closed, 9.3, "9.3% on the second split's models")
check("repaste", tp.repaste, 0.0, "pasting it back unmoved gives 0.0%")

# ---- gap flag, augmentation, turnover
flag = pd.concat([csv(f"kaggle_gapflag/gap_flag/gap_flag_seed{s}.csv") for s in (0, 1, 2)])
fl = flag[flag.variant == "flag"]
check("flag cost min", fl.cost_pct.min(), 12, "flags 12 to 16% of teeth", 0)
check("flag cost max", fl.cost_pct.max(), 16, None, 0)
check("flag recall min", fl.joint_recall_flag.min(), 41, "catches 41 to 47%", 0)
check("flag recall max", fl.joint_recall_flag.max(), 47, None, 0)
check("conf recall min", fl.joint_recall_conf.min(), 17, "against 17 to 32%", 0)
check("conf recall max", fl.joint_recall_conf.max(), 32, None, 0)
ok_seeds = sum(((fl[fl.seed == s].ratio >= 1.5) & (fl[fl.seed == s].diff_lo > 0)).sum() >= 2 for s in (0, 1, 2))
check("splits meeting the 1.5-fold bar", ok_seeds, 2, "reached on two of three", 0)
au = csv("kaggle_augment/augment/augment_summary.csv")
a = au[(au.comparison == "augmented minus control") & (au["subset"] == "next_to_gap")].iloc[0]
check("augmentation diff", a["diff"], 0.1, "+0.1 points, CI -1.6 to 1.7")
check("augmentation lo", a.diff_lo, -1.6)
check("augmentation hi", a.diff_hi, 1.7)


def joint_set(path):
    t = pd.read_csv(R / path)
    P_ = np.stack([t[f"{d}_pred"].to_numpy(float) for d in DETS])
    c = t.class_id.to_numpy()
    w = np.logical_and.reduce([~np.isnan(P_[k]) & (P_[k] != c) for k in range(3)]) & (t.coord_pred.to_numpy() != c)
    m = w & (P_ == P_[0]).all(0)
    return set(zip(t.image_id[m], t.fdi[m]))


s0 = joint_set("kaggle_tseed_3det/seed0_3det/per_tooth_predictions.csv")
ts1 = joint_set("kaggle_tseed_3det/tseed1_3det/per_tooth_predictions.csv")
sp1 = joint_set("kaggle_seed1_3det/seed1_3det/per_tooth_predictions.csv")
sp2 = joint_set("kaggle_seed2_3det/seed2_3det/per_tooth_predictions.csv")
check("overlap training seed", len(s0 & ts1) / len(s0 | ts1), 0.63, "overlap 0.63, against 0.54 to 0.57", 2)
check("overlap splits min", min(len(s0 & sp1) / len(s0 | sp1), len(s0 & sp2) / len(s0 | sp2)), 0.54, None, 2)
check("overlap splits max", max(len(s0 & sp1) / len(s0 | sp1), len(s0 & sp2) / len(s0 | sp2)), 0.57, None, 2)

# ---- DENTEX
tc = csv("dentex_tseed_compare3/tseed_compare.csv")
v = lambda grp, met, col="point": tc[(tc.group == grp) & (tc.metric == met)][col].iloc[0]
d0 = csv("kaggle_dentex_indomain/dentex_indomain/dentex_summary.csv").set_index("metric")
check("DENTEX position-only", d0.loc["pct_coord_correct", "point"], 66.9, "numbers 66.9% of these teeth")
mg = [v(s, f"margin_{d}_pp") for s in ("seed0",) for d in ("yolov8x", "rtdetr_l", "fasterrcnn")]
check("DENTEX margin min seed 0", min(mg), 22, "by 22 to 25 points", 0)
check("DENTEX margin max seed 0", max(mg), 25, None, 0)
m12 = [v(s, f"margin_{d}_pp") for s in ("seed1", "seed2") for d in ("yolov8x", "rtdetr_l", "fasterrcnn")]
check("later seeds margin min", min(m12), 24, "margins of 24 to 26 points", 0)
check("later seeds margin max", max(m12), 26, None, 0)
check("shared errors seed 0 position share", v("seed0", "position_share_pct"), 85.5, "85.5% of the time (UFBA")
check("shared errors seed 0 n", tc[(tc.group == "seed0") & (tc.metric == "position_share_pct")].n.iloc[0], 303, "Their 303", 0)
check("gap ratio seed 0", v("seed0", "gap_ratio"), 3.5, "3.5 times as often")
check("position share seed 1", v("seed1", "position_share_pct"), 85.5, "match of 85.5% and 84.2%")
check("position share seed 2", v("seed2", "position_share_pct"), 84.2)
check("gap ratio seed 1", v("seed1", "gap_ratio"), 3.3, "ratio of 3.3 and 3.2")
check("gap ratio seed 2", v("seed2", "gap_ratio"), 3.2)
o = tc[tc.group == "overlap3"].iloc[0]
check("shared in all three", o.both, 213, None, 0)
check("shared in any", o.n, 410, None, 0)
gs = csv("gap_strata/gap_strata_standardized.csv")
print("  standardized-ratio file:", gs.to_dict("records")[:2])

dx = csv("kaggle_dentex/dentex_summary.csv").set_index("metric")
check("UFBA-trained shared errors follow position", dx.loc["pos_match_rule3", "point"], 87.5, "still follow position (87.5%)")
mt = csv("missing_teeth/missing_teeth.csv").set_index("dataset")
check("DENTEX has fewer missing teeth", float(mt.loc["DENTEX", "mean_missing"] < mt.loc["UFBA", "mean_missing"]), 1.0, None, 0)
std = csv("gap_strata/gap_strata_standardized.csv").set_index("dataset")
check("standardized ratio", std.loc["DENTEX within", "std_ratio"], 3.9, "mix of missing teeth per X-ray, the ratio is 3.9")

# ---- DENTEX interventions
def im(path):
    return pd.read_csv(R / path).set_index("measure").loc["all3"]
st = {"slide": im("kaggle_dentex_closure_s0n/dentex-closure-s0n/closure_summary.csv"),
      "tip": im("kaggle_dentex_tip_s0n/dentex-tip-s0n/closure_summary.csv")}
check("stored slide", st["slide"].closed, 15.7, "number 15.7% of the time (0.4% open)")
check("stored slide open", st["slide"].open, 0.4)
check("stored tip", st["tip"].closed, 7.0, "tipping it 7.0% (0.4% open)")
check("stored tip open", st["tip"].open, 0.4)
check("stored slide vs UFBA", 100 * st["slide"].closed / cl.closed, 75, "about three quarters", 0)
check("stored tip vs UFBA", 100 * st["tip"].closed / tp.closed, 75, None, 0)
sl = [im(p).closed for p in ("kaggle_dentex_closure_s0/dentex-closure-s0/closure_summary.csv",
                             "kaggle_dentex_closure_t1/dentex-closure-t1/closure_summary.csv",
                             "kaggle_dentex_closure_t2/dentex-closure-t2/closure_summary.csv")]
ti = [im(p).closed for p in ("kaggle_dentex_tip_s0/dentex-tip-s0/closure_summary.csv",
                             "kaggle_dentex_tip_t1/dentex-tip-t1/closure_summary.csv",
                             "kaggle_dentex_tip_t2/dentex-tip-t2/closure_summary.csv")]
check("slide 640 values", float([round(x, 1) for x in sl] == [11.8, 12.0, 12.6]), 1.0, "slide 11.8%, 12.0% and 12.6%", 0)
check("tip 640 values", float([round(x, 1) for x in ti] == [4.8, 5.5, 6.5]), 1.0, "tip 4.8%, 5.5% and 6.5%", 0)
check("tip bar met on", sum(x >= 5 for x in ti), 2, "met on two of the three", 0)
check("full range in abstract low", min(sl + ti[:0]), 12, "12 to 16% of the time", 0)
check("full range in abstract high", st["slide"].closed, 16, None, 0)

# ---- DENTEX fill-set overlap (slide)
def fills(path, cond="closed"):
    d = pd.read_csv(R / path)
    k = ["image_id", "target"]
    o_ = d[d.condition == "open"].assign(r=lambda x: x.intact_pred == x.mover_class).groupby(k).r.agg(lambda s: bool(s.all()) and len(s) == 3)
    mv = set(o_[o_].index)
    c = d[d.condition == cond].assign(f=lambda x: x.pred == x.target_class).groupby(k).f.agg(lambda s: bool(s.all()) and len(s) == 3)
    return {i for i in c[c].index if i in mv}


F = [fills(p.replace("closure_summary", "closure_rows")) for p in
     ("kaggle_dentex_closure_s0/dentex-closure-s0/closure_summary.csv",
      "kaggle_dentex_closure_t1/dentex-closure-t1/closure_summary.csv",
      "kaggle_dentex_closure_t2/dentex-closure-t2/closure_summary.csv")]
check("slide fills any", len(F[0] | F[1] | F[2]), 341, None, 0)
check("slide fills all", len(F[0] & F[1] & F[2]), 57, None, 0)

# data facts quoted in Methods and Results
from PIL import Image
imgs = sorted((ROOT / "Dataset/bb_u_net_dataset/panoramic_x_rays").glob("*.jpg"))
check("UFBA image count", len(imgs), 425, None, 0)
check("UFBA images are 512 x 512", float({Image.open(f).size for f in imgs} == {(512, 512)}), 1.0, "512\\times512 pixels", 0)
bx = pd.read_csv(ROOT / "cpu_repro/cv/boxes.csv")
check("10% shift in tooth widths", 0.10 / bx.width.median(), 2.0, "about two tooth widths", 0)
dbx = pd.read_csv(ROOT / "cpu_repro/cv/dentex_boxes.csv")
check("DENTEX teeth", len(dbx), 18095, "(18,095 teeth)", 0)
check("DENTEX X-rays", dbx.image_id.nunique(), 634, "634 DENTEX X-rays", 0)
gc = csv("kaggle_gapcheck/gap_check_summary.csv").iloc[0]
check("joint failures next to a gap", gc.r_J_pct / 100 * gc.n_J, 72, "In 67 of the 72", 0)

fo = csv("fold_overlap/fold_overlap.csv")
check("fold overlap with first split", fo[fo.split_a == 0].max_share_of_fold_pct.min(), 26, "at most 26% and 32% of any test fold", 0)
check("fold overlap with first split, max", fo[fo.split_a == 0].max_share_of_fold_pct.max(), 32, None, 0)

# ---- supplement: AP table and intervention tables
SUPP = re.sub(r"\s+", " ", (ROOT / "paper/miccai/SUPPLEMENT.md").read_text())


def in_supp(label, text):
    global n_ok
    if re.sub(r"\s+", " ", text) in SUPP:
        n_ok += 1
    else:
        fails.append(f"{label}: not in supplement: {text!r}")


in_supp("shared in all three runs (supplement)", "213 of the 410 teeth that are a shared error in any run")
in_supp("slide fills in all three runs (supplement)", "57 of 341 slide fills")


f1d = lambda x: f"{x:.1f}"
for kind, fn in (("ufba", "ap/ufba/ap_summary.csv"), ("dentex", "ap/dentex/ap_summary.csv")):
    ap_ = csv(fn)
    for _, r_ in ap_.iterrows():
        in_supp(r_.run, f"{f1d(r_.ap50)} ({f1d(r_.ap50_lo)} to {f1d(r_.ap50_hi)}) | {f1d(r_.ap5095)} "
                        f"({f1d(r_.ap5095_lo)} to {f1d(r_.ap5095_hi)}) |")
ufb = csv("ap/ufba/ap_summary.csv").set_index("run")
in_supp("RT-DETR-l boxes", "RT-DETR-l about 290 per X-ray") if 285 <= ufb.loc["ufba_split0_rtdetr_l", "n_dets"] / 425 <= 295 else fails.append("RT-DETR-l boxes per X-ray")
in_supp("YOLO boxes", "YOLOv8x about 29)") if 28 <= ufb.loc["ufba_split0_yolov8x", "n_dets"] / 425 <= 30 else fails.append("YOLO boxes per X-ray")


def ci2(r_, col):
    return f"{r_[col]:.2f} ({r_[col + '_lo']:.2f} to {r_[col + '_hi']:.2f})"


S2 = {("slide", "s0"): "kaggle_dentex_closure_s0/dentex-closure-s0", ("slide", "t1"): "kaggle_dentex_closure_t1/dentex-closure-t1",
      ("slide", "t2"): "kaggle_dentex_closure_t2/dentex-closure-t2", ("tip", "s0"): "kaggle_dentex_tip_s0/dentex-tip-s0",
      ("tip", "t1"): "kaggle_dentex_tip_t1/dentex-tip-t1", ("tip", "t2"): "kaggle_dentex_tip_t2/dentex-tip-t2",
      ("slide", "n"): "kaggle_dentex_closure_s0n/dentex-closure-s0n", ("tip", "n"): "kaggle_dentex_tip_s0n/dentex-tip-s0n",
      ("slide", "ufba"): "kaggle_closure/closure_intervention", ("tip", "ufba"): "kaggle_tip/closure_tip"}
for (mode, run), pth in S2.items():
    r_ = im(pth + "/closure_summary.csv")
    if (mode, run) == ("tip", "ufba"):  # the supplement gives UFBA's tip as 9.4 (7.73 to 11.03), no movers or difference
        in_supp("S2 tip ufba", f"{r_.closed:.1f} ({r_.closed_lo:.2f} to {r_.closed_hi:.2f})")
        continue
    in_supp(f"S2 {mode} {run} closed", ci2(r_, "closed"))
    if (mode, run) != ("slide", "n"):  # the stored-size slide difference is not tabulated in the supplement
        in_supp(f"S2 {mode} {run} closed minus open", ci2(r_, "closed_minus_open"))
    in_supp(f"S2 {mode} {run} movers", f"{int(r_.n_movers):,}")
    if mode == "tip" and run != "ufba":
        in_supp(f"S2 tip {run} paste", f"{r_.repaste:.2f}")

ph = csv("post_hoc_two_detector/post_hoc_two_detector.csv")
g_ = lambda m, sz, dt: ph[(ph["mode"] == m) & (ph["size"] == sz) & (ph.detectors == dt)].j_pct.iloc[0]
in_supp("post hoc tip two-detector", f"the full tip is {g_('tip', 'stored', 'two'):.2f} against {g_('tip', '640', 'two'):.2f}%")
in_supp("post hoc slide two-detector", f"the full slide {g_('slide', 'stored', 'two'):.2f} against {g_('slide', '640', 'two'):.2f}%")
in_supp("seed 0 two-detector 640 slide", f"{g_('slide', '640', 'two'):.1f}% (slide)")
in_supp("UFBA two-detector rates", f"against {g_('slide', 'ufba', 'two'):.1f} and {g_('tip', 'ufba', 'two'):.1f}% on UFBA")

body = TEX[TEX.index("\\begin{abstract}"):TEX.index("\\bibliographystyle")]
check("no em or en dash characters in the paper text", float(not re.search("[\u2013\u2014]", body)), 1.0, None, 0)
check("no -- or --- in the paper text", float(not re.search("--", re.sub(r"\\(?:cite|ref)\{[^}]*\}", "", body))), 1.0, None, 0)
check("data use statement present", 0, 0, "Data use.", 0)

print(f"{n_ok} checks passed, {len(fails)} failed")
for f_ in fails:
    print("FAIL:", f_)
sys.exit(1 if fails else 0)
