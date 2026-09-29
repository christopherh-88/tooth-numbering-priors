"""Recompute every number in the CJSJ paper from committed result files.

Usage: python paper/cjsj/paper_numbers.py

Each check prints the value computed from the repository and the value as it
is written in the paper (HuangChristopher_paper.docx), rounded the same way.
The script exits with status 1 if any printed value disagrees with the paper.

One group of numbers cannot be recomputed from committed files alone and is
listed at the end with its source: the supernumerary-tooth check, which needs
the third-party dual-labeled dataset (RESULTS.md Section 17).
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CB = REPO / "cpu_repro" / "coord_baseline"
YT = REPO / "cpu_repro" / "yolo_training" / "eval_results"
DP = REPO / "cpu_repro" / "boundary_condition" / "denpar_periapical"
sys.path.insert(0, str(CB))
from build_coord_baseline import mean_ci95  # noqa: E402

FAILED = []


def show(label, computed, paper):
    ok = computed == paper
    if not ok:
        FAILED.append(label)
    print(f"  [{'ok' if ok else 'MISMATCH'}] {label}: repo {computed} | paper {paper}")


def p1(x):
    """Percent with one decimal, as in the paper."""
    return f"{100 * x:.1f}"


def p0(x):
    return f"{100 * x:.0f}"


def main():
    # ------------------------------------------------------------------ Claim A
    print("Claim A: position-only model (gradient-boosted tree, seeds 0-4)")
    ufba = pd.read_csv(CB / "summary.csv").set_index("classifier").loc["gradient_boosted_tree"]
    dentex = pd.read_csv(CB / "dentex" / "summary.csv").set_index("classifier").loc["gradient_boosted_tree"]
    show("UFBA-425 top-1 (%)", p1(ufba.top1_acc_mean), "69.5")
    show("UFBA-425 top-1 CI half-width (pp)", p1(ufba.top1_acc_ci95), "0.8")
    show("DENTEX top-1 (%)", p1(dentex.top1_acc_mean), "68.5")
    show("UFBA-425 majority baseline (%)", p1(ufba.majority_baseline_acc_mean), "3.6")
    show("DENTEX majority baseline (%)", p1(dentex.majority_baseline_acc_mean), "3.7")
    show("UFBA-425 quadrant (%)", p1(ufba.quadrant_acc_mean), "96.5")
    show("DENTEX quadrant (%)", p1(dentex.quadrant_acc_mean), "98.0")
    ctl = pd.read_csv(CB / "controls" / "summary.csv")
    ctl_d = pd.read_csv(CB / "dentex" / "controls" / "summary.csv")

    def shuffled(df):
        r = df[(df.condition == "full (shuffled per image)") & (df.classifier == "gradient_boosted_tree")]
        return r.top1_acc_mean.item()

    show("UFBA-425 shuffled control (%)", p1(shuffled(ctl)), "3.2")
    show("DENTEX shuffled control (%)", p1(shuffled(ctl_d)), "3.7")
    abl = pd.read_csv(CB / "feature_ablation.csv").set_index("feature_set").top1_acc_mean
    full = abl["all_six (main baseline)"]
    show("centre coordinates only (%)", p1(abl["position_only (x,y)"]), "62.0")
    show("size and shape features only (%)", p1(abl["shape_only (w,h,area,aspect)"]), "14.3")
    show("cost of removing x (pp)", f"{100 * (full - abl['all_minus_x_center']):.1f}", "47.5")
    show("cost of removing y (pp)", f"{100 * (full - abl['all_minus_y_center']):.1f}", "16.8")
    single_shape = max(abs(full - abl[f"all_minus_{f}"]) for f in ("width", "height", "area", "aspect_ratio"))
    show("any single shape feature changes accuracy by at most 0.3 pp", str(single_shape <= 0.003), "True")
    show("shape features together (pp)", f"{100 * (full - abl['position_only (x,y)']):.1f}", "7.5")

    # --------------------------------------------------------------- datasets
    print("\nDatasets and splits")
    metas = [json.loads((CB / f"image_split_seed{s}.meta.json").read_text()) for s in range(5)]
    n_test = [m["n_test_instances"] for m in metas]
    show("UFBA-425 test teeth per seed", f"{min(n_test):,} to {max(n_test):,}", "4,981 to 5,916")
    show("UFBA-425 source images", str(metas[0]["n_train_images"] + metas[0]["n_test_images"]), "425")

    # ---------------------------------------------------------------- Claim B
    print("\nClaim B: detectors vs. the position-only model (T4 seeds 0-4)")
    det = {
        "YOLOv8x": (pd.read_csv(YT / "multiseed" / "multiseed_summary.csv"), "yolo"),
        "RT-DETR-l": (pd.read_csv(YT / "rtdetr_multiseed" / "rtdetr_multiseed_summary.csv"), "rtdetr"),
        "Faster R-CNN": (pd.read_csv(YT / "fasterrcnn_multiseed" / "fasterrcnn_multiseed_summary.csv"), "fasterrcnn"),
    }
    paper_rows = {
        "YOLOv8x": ("94.3", "24.8 [24.2, 25.4]", "0.18 [0.13, 0.23]"),
        "RT-DETR-l": ("94.5", "25.0 [24.2, 25.8]", "0.19 [0.15, 0.22]"),
        "Faster R-CNN": ("93.3", "23.8 [23.0, 24.7]", "0.18 [0.15, 0.21]"),
    }
    quads, gaps = [], []
    for name, (df, key) in det.items():
        top1 = df[f"{key}_top1"].mean()
        g, gci = mean_ci95(df.gap_pp)
        ph, phci = mean_ci95(df.phi)
        show(f"{name} top-1 (%)", p1(top1), paper_rows[name][0])
        show(f"{name} gap [95% CI] (pp)", f"{g:.1f} [{g - gci:.1f}, {g + gci:.1f}]", paper_rows[name][1])
        show(f"{name} phi [95% CI]", f"{ph:.2f} [{ph - phci:.2f}, {ph + phci:.2f}]", paper_rows[name][2])
        quads.append(df[f"{key}_quadrant"].mean())
        gaps += list(df.gap_pp)
    show("position-only top-1 on the same teeth (%)", p1(det["YOLOv8x"][0].coord_top1.mean()), "69.5")
    show("runs above the 15 pp threshold", f"{sum(g >= 15 for g in gaps)} of {len(gaps)}", "15 of 15")
    frc = det["Faster R-CNN"][0]
    worst = frc.loc[frc.gap_pp.idxmin()]
    show("smallest single-run gap (pp)", f"{min(gaps):.1f}", "22.7")
    show("its bootstrap CI lower end (pp)", f"{worst.gap_ci_lo:.1f}", "19.3")
    show("quadrant on detected teeth (%)", f"{100 * min(quads):.1f} to {100 * max(quads):.1f}", "99.6 to 99.7")
    phis = pd.concat([d.phi for d, _ in det.values()])
    show("phi range over the 15 runs", f"{phis.min():.2f} to {phis.max():.2f}", "0.13 to 0.24")

    # --------------------------------------------------------------- per class
    print("\nPer tooth class")
    h2h = pd.read_csv(YT / "per_tooth_head_to_head.csv")
    for group, paper_wins in (("CUDA seeds 0-4", "96 of 96"), ("MPS seeds 5-9", "96 of 96"),
                              ("CUDA seeds 11-14", "96 of 96")):
        g = h2h[h2h.group == group]
        show(f"{group}: detector-class pairs with CI above zero", f"{int(g.detector_wins.sum())} of {len(g)}", paper_wins)
    g0 = h2h[h2h.group == "CUDA seeds 0-4"]
    r = g0.loc[g0.delta.idxmin()]
    show("narrowest margin, seeds 0-4",
         f"{r.model} FDI {r.fdi} +{100 * r.delta:.1f} [+{100 * r.ci_lo:.1f}, +{100 * r.ci_hi:.1f}]",
         "yolo FDI 38 +7.2 [+3.5, +10.9]")
    rest = h2h[h2h.group != "CUDA seeds 0-4"]
    r = rest.loc[rest.delta.idxmin()]
    show("narrowest margin, replication groups",
         f"{r.model} FDI {r.fdi} +{100 * r.delta:.1f} [+{100 * r.ci_lo:.1f}, +{100 * r.ci_hi:.1f}]",
         "rtdetr FDI 38 +4.2 [+0.7, +7.9]")
    rev = []
    for d, col in (("multiseed", "delta_yolo_minus_coord"), ("rtdetr_multiseed", "delta_rtdetr_minus_coord"),
                   ("fasterrcnn_multiseed", "delta_fasterrcnn_minus_coord")):
        pc = pd.read_csv(YT / d / "per_class_breakdown_multiseed.csv")
        pc = pc[pc.seed.isin(range(5))]
        rev += [(d, s, f, v) for s, f, v in zip(pc.seed, pc.fdi_code, pc[col]) if v < 0]
        n_cells = len(pc)
    show("single-seed reversals", f"{len(rev)} of {3 * n_cells}", "1 of 480")
    d, s, f, v = rev[0]
    show("the reversal", f"{d} seed {s} FDI {f} {100 * v:.1f} pp", "multiseed seed 1 FDI 38 -1.6 pp")

    # ------------------------------------------------------------- case study
    print("\nCase study (seed 0)")
    cs = pd.read_csv(YT / "case_study_yolo_correct_coord_wrong.csv")
    mix = cs.coord_error_type.value_counts(normalize=True)
    show("teeth YOLOv8x right, position-only wrong", f"{len(cs):,}", "1,490")
    show("mirror-image swaps (%)", p1(mix["mirror_quadrant"]), "7.5")
    show("neighbour mix-ups (%)", p1(mix["neighbor"]), "85.4")
    tax = pd.read_csv(YT / "multiseed" / "error_taxonomy_multiseed.csv").set_index("seed").loc[0]
    show("position-only overall error mix at seed 0 (mirror, neighbour %)",
         f"{p1(tax.coord_mirror_frac)}, {p1(tax.coord_neighbor_frac)}", "7.5, 84.2")
    man = pd.read_csv(YT / "case_study_crops" / "manifest.csv")
    fig = man[man.crop_file.str.startswith("case_07_")].iloc[0]
    show("Fig. 1b case", f"{Path(fig.source_image).name.split('_jpg')[0]} true {fig.true_fdi} "
         f"YOLOv8x {fig.yolo_pred_fdi} position-only {fig.coord_pred_fdi}",
         "cate8-00390 true 11 YOLOv8x 11 position-only 21")

    # ----------------------------------------------------------- augmentation
    print("\nAugmentation experiment (YOLOv8x, seed 0)")
    mit = pd.read_csv(YT / "mitigation_bootstrap_ci.csv").set_index("metric").loc["paired_delta_top1"]
    show("paired difference [95% CI] (pp)",
         f"{100 * mit.point:.2f} [{100 * mit.ci_lo:.2f}, +{100 * mit.ci_hi:.2f}]", "-0.18 [-0.74, +0.34]")

    # ------------------------------------------------------ error agreement
    print("\nJoint failures and error agreement")
    ag = pd.read_csv(YT / "cross_architecture_agreement_summary.csv")
    n_by_seed = det["YOLOv8x"][0].set_index("seed").n
    a04 = ag[ag.seed.isin(range(5))].set_index("seed")
    frac = a04.n_joint_triple / n_by_seed
    show("joint-failure share of test teeth (%)", f"{p1(frac.min())} to {p1(frac.max())}", "1.1 to 2.8")
    show("same wrong tooth, seeds 0-4 (%)", p1(a04.triple_rate.mean()), "96.8")
    show("  range", f"{p1(a04.triple_rate.min())} to {p1(a04.triple_rate.max())}", "91.5 to 98.7")
    show("matches position-only guess, seeds 0-4 (%)", p1(a04.coord_match_rate.mean()), "84.5")
    show("  range", f"{p1(a04.coord_match_rate.min())} to {p1(a04.coord_match_rate.max())}", "76.8 to 88.8")
    show("neighbour share of shared errors (%)", p0(a04.adjacent_share_pooled.mean()), "91")
    show("  range", f"{p0(a04.adjacent_share_pooled.min())} to {p0(a04.adjacent_share_pooled.max())}", "87 to 94")
    for label, seeds, paper in (("M3 seeds 5-9", range(5, 10), ("97.4", "87.5")),
                                ("T4 seeds 11-14", range(11, 15), ("97.9", "88.1"))):
        g = ag[ag.seed.isin(seeds)]
        show(f"{label}: same wrong tooth, matches position-only (%)",
             f"{p1(g.triple_rate.mean())}, {p1(g.coord_match_rate.mean())}", f"{paper[0]}, {paper[1]}")

    nulls = pd.read_csv(YT / "cross_architecture_agreement_nulls.csv")
    show("chance agreement across all seeds and comparisons (%)",
         f"{p0(nulls.null_mean.min())} to {p0(nulls.null_mean.max())}", "4 to 11")
    show("largest permutation p", f"{nulls.p.max():.4f}", "0.0000")

    # ---------------------------------------------------------------- DenPAR
    print("\nDenPAR periapical boundary test")
    dp = pd.read_csv(DP / "summary.csv").set_index("classifier").loc["gradient_boosted_tree"]
    boot = pd.read_csv(DP / "bootstrap_ci_section25.csv").set_index("metric")
    show("top-1 (%)", p1(dp.top1_acc_mean), "26.6")
    show("CI half-width (pp)", p1(dp.top1_acc_ci95), "2.8")
    show("quadrant (%)", p1(dp.quadrant_acc_mean), "45.5")
    show("tooth type (%)", p1(dp.tooth_type_acc_mean), "45.0")
    show("majority baseline (%)", p1(dp.majority_baseline_acc_mean), "7.6")
    show("ratio to majority baseline", f"{dp.top1_acc_mean / dp.majority_baseline_acc_mean:.1f}", "3.5")
    show("shuffled control, seed 0 (%)", p1(boot.loc["top1_acc_shuffled_control", "point"]), "4.2")

    print("\nNot recomputable from committed files (quoted from RESULTS.md):")
    print("  Section 17, dual-labeled dataset: 32.9% vs. 3.76% majority (8.8x); 29.5% vs. 33.0% per tooth,"
          " 28.8% vs. 32.3% per image; p = 0.0765 (z-test), p = 0.3615 (Mann-Whitney); 23 images.")

    print(f"\n{'ALL NUMBERS MATCH' if not FAILED else f'{len(FAILED)} MISMATCH(ES): ' + '; '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
