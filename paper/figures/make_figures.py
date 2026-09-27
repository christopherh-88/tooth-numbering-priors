"""Paper figures, built only from committed result CSVs.

Usage (repo root or anywhere): python paper/figures/make_figures.py

Writes fig*.pdf (vector, fonts embedded) and fig*.png (600 dpi) next to this
script. Sized for the LNCS/MICCAI text width (122 mm). Error bars are 95%
t-intervals over seeds (build_coord_baseline.mean_ci95, the same helper
RESULTS.md uses) unless a caption in README.md says otherwise. Key plotted
values are printed so they can be checked against RESULTS.md.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CB = REPO / "cpu_repro" / "coord_baseline"
YT = REPO / "cpu_repro" / "yolo_training" / "eval_results"
DENPAR = REPO / "cpu_repro" / "boundary_condition" / "denpar_periapical"
sys.path.insert(0, str(CB))
from build_coord_baseline import mean_ci95  # noqa: E402

TEXT_WIDTH = 122 / 25.4  # LNCS text width, inches

# Okabe-Ito colours; the coordinate-only prior is always grey.
PRIOR = "#6E6E6E"
LR = "#B0B0B0"
DET = {"yolo": "#0072B2", "rtdetr": "#D55E00", "fasterrcnn": "#009E73"}
DET_NAME = {"yolo": "YOLOv8x", "rtdetr": "RT-DETR-l", "fasterrcnn": "Faster R-CNN"}
QUAD = "#332288"
TYPE = "#CC6677"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "axes.titleweight": "bold",
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def panel(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def pct(ax, axis="y", decimals=0):
    fmt = matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def seeds_and_mean(ax, x, values, color, marker="o", filled=True, dx=0.0, spread=0.07, ms=3.2, label=None):
    """Individual seeds as small dots, mean +/- 95% CI as a larger marker."""
    values = np.asarray(values, float)
    offsets = np.linspace(-spread, spread, len(values)) if len(values) > 1 else [0]
    ax.scatter(x + dx + np.asarray(offsets), values, s=5, color=color, alpha=0.45, linewidths=0, zorder=2)
    m, ci = mean_ci95(values)
    ax.errorbar(x + dx, m, yerr=ci, fmt=marker, color=color, ms=ms, capsize=1.8, elinewidth=0.8,
                capthick=0.8, mfc=color if filled else "white", mew=0.8, zorder=3, label=label)
    return m, ci


def save(fig, name):
    for ext, kw in (("pdf", {"metadata": {"CreationDate": None, "ModDate": None}}), ("png", {"dpi": 600})):
        fig.savefig(HERE / f"{name}.{ext}", **kw)
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


# ---------------------------------------------------------------- Figure 1
DATASETS = [("UFBA-425", "per_seed_results.csv"), ("DENTEX", "dentex/per_seed_results.csv")]


def modality_brackets(ax):
    """'panoramic' under the first two datasets, 'periapical' under the third."""
    for x0, x1, lab in ((-0.3, 1.3, "panoramic"), (1.7, 2.3, "periapical")):
        ax.annotate("", xy=(x0, -0.2), xytext=(x1, -0.2), xycoords=("data", "axes fraction"),
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#555555"}, annotation_clip=False)
        ax.text((x0 + x1) / 2, -0.23, lab, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6, color="#555555", style="italic")


def fig1_coordinate_baseline():
    data = {"UFBA-425": pd.read_csv(CB / "per_seed_results.csv"),
            "DENTEX": pd.read_csv(CB / "dentex" / "per_seed_results.csv"),
            "DenPAR": pd.read_csv(DENPAR / "per_seed_results.csv")}

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.05), gridspec_kw={"wspace": 0.38})

    ax = axes[0]
    print("Fig 1a top-1 (mean, CI):")
    for i, (name, df) in enumerate(data.items()):
        for clf, col, dx in (("gradient_boosted_tree", PRIOR, -0.15), ("logistic_regression", LR, 0.15)):
            m, ci = seeds_and_mean(ax, i, df[df.classifier == clf].top1_acc, col, dx=dx,
                                   label=("Gradient-boosted tree" if clf.startswith("g") else "Logistic regression") if i == 0 else None)
            print(f"  {name:9s} {clf[:3]} {m:.4f} ± {ci:.4f}")
        ax.hlines(df.majority_baseline_acc.mean(), i - 0.32, i + 0.32, color="black", lw=0.8,
                  ls=(0, (2, 1.5)), zorder=1, label="Majority class" if i == 0 else None)
    ax.set_xticks(range(3), data.keys())
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 0.8)
    pct(ax)
    ax.set_ylabel("Top-1 accuracy (32 FDI classes)")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.56), handletextpad=0.3, borderaxespad=0.2)
    modality_brackets(ax)
    ax.set_title("a", loc="left", fontsize=9, x=-0.2, y=1.0)

    ax = axes[1]
    print("Fig 1b quadrant / tooth type (GBT):")
    for i, (name, df) in enumerate(data.items()):
        g = df[df.classifier == "gradient_boosted_tree"]
        mq, _ = seeds_and_mean(ax, i, g.quadrant_acc, QUAD, dx=-0.15, label="Quadrant" if i == 0 else None)
        mt, _ = seeds_and_mean(ax, i, g.tooth_type_acc, TYPE, marker="s", dx=0.15,
                               label="Tooth type (1-8 within quadrant)" if i == 0 else None)
        print(f"  {name:9s} quadrant {mq:.4f}  tooth type {mt:.4f}")
    ax.set_xticks(range(3), data.keys())
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1.05)
    pct(ax)
    ax.set_ylabel("Accuracy (gradient-boosted tree)")
    ax.legend(loc="lower left", handletextpad=0.3, borderaxespad=0.2)
    modality_brackets(ax)
    ax.set_title("b", loc="left", fontsize=9, x=-0.2, y=1.0)

    save(fig, "fig1_coordinate_baseline")


# ---------------------------------------------------------------- Figure 2
def fig2_ablation_noise():
    abl = pd.read_csv(CB / "feature_ablation.csv").set_index("feature_set")
    ctrl = pd.read_csv(CB / "controls" / "summary.csv")
    shuffled = ctrl[(ctrl.classifier == "gradient_boosted_tree") & (ctrl.condition == "full (shuffled per image)")].iloc[0]
    noise = pd.read_csv(CB / "noise_robustness_summary.csv")
    var = pd.read_csv(CB / "positional_variance_by_class.csv")
    majority = noise.majority_baseline_acc_mean.iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH * 0.84, 2.1), gridspec_kw={"width_ratios": [1.0, 1], "wspace": 0.5})

    ax = axes[0]
    rows = [("all_six (main baseline)", "All six features"),
            ("all_minus_x_center", "All but x"),
            ("all_minus_y_center", "All but y"),
            ("all_minus_width", "All but width"),
            ("position_only (x,y)", "x and y only"),
            ("single:x_center", "x only"),
            ("single:y_center", "y only"),
            ("shape_only (w,h,area,aspect)", "Shape only (w, h, area, aspect)")]
    vals = [abl.loc[k, "top1_acc_mean"] for k, _ in rows] + [shuffled.top1_acc_mean]
    cis = [abl.loc[k, "top1_acc_ci95"] for k, _ in rows] + [shuffled.top1_acc_ci95]
    labels = [lab for _, lab in rows] + ["All six, boxes shuffled\namong teeth in each image"]
    colors = [PRIOR] + ["#9A9A9A"] * 4 + ["#C8C8C8"] * 3 + ["white"]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, xerr=cis, color=colors, edgecolor=["none"] * 8 + [PRIOR], linewidth=0.6, height=0.66,
            error_kw={"elinewidth": 0.7, "capsize": 1.5, "capthick": 0.7})
    ax.axvline(vals[0], color=PRIOR, lw=0.6, ls=":")
    ax.axvline(majority, color="black", lw=0.8, ls=(0, (2, 1.5)))
    ax.text(majority + 0.012, y[0] + 0.62, "majority class", fontsize=5.6, va="bottom")
    for yi, v in zip(y, vals):
        ax.text(max(v, majority) + 0.03, yi, f"{100 * v:.1f}", va="center", fontsize=5.8)
    ax.set_yticks(y, labels)
    ax.set_ylim(y[-1] - 0.6, y[0] + 1.2)
    ax.set_xlim(0, 0.85)
    pct(ax, "x")
    ax.set_xlabel("Top-1 accuracy (UFBA-425)")
    ax.set_title("a", loc="left", fontsize=9, x=-0.72, y=1.0)
    print("Fig 2a:", {lab.split(chr(10))[0]: round(v, 4) for lab, v in zip(labels, vals)})

    ax = axes[1]
    for clf, col, mk, lab in (("gradient_boosted_tree", PRIOR, "o", "Gradient-boosted tree"),
                              ("logistic_regression", LR, "s", "Logistic regression")):
        g = noise[noise.classifier == clf].sort_values("noise_std")
        ax.errorbar(g.noise_std, g.top1_acc_mean, yerr=g.top1_acc_ci95, fmt=f"-{mk}", color=col, ms=2.8,
                    capsize=1.5, elinewidth=0.7, lw=0.9, label=lab, mfc=col)
    ax.axhline(majority, color="black", lw=0.8, ls=(0, (2, 1.5)))
    ax.text(0.0005, majority + 0.015, "majority class", fontsize=5.6, ha="left", va="bottom")
    sx, sy = var.x_std.mean(), var.y_std.mean()
    ax.axvspan(sx, sy, color="#E8E8E8", zorder=0, lw=0)
    ax.text(np.sqrt(sx * sy), 0.1, "natural\nper-class\nspread\n(x to y)", fontsize=5.4, ha="center", va="bottom")
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.5)
    ax.set_xticks([0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32], ["0", ".01", ".02", ".04", ".08", ".16", ".32"])
    ax.minorticks_off()
    ax.set_xlim(-0.002, 0.36)
    ax.set_ylim(0, 0.78)
    pct(ax)
    ax.set_xlabel("Test-time box noise (s.d., fraction of image)")
    ax.set_ylabel("Top-1 accuracy (UFBA-425)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.02), handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("b", loc="left", fontsize=9, x=-0.3, y=1.0)
    print(f"Fig 2b: natural per-class spread x {sx:.4f}, y {sy:.4f}")

    save(fig, "fig2_ablation_noise")


# ---------------------------------------------------------------- Figure 3
def load_multiseed():
    out = {}
    for key, folder, col in (("yolo", "multiseed", "yolo_top1"), ("rtdetr", "rtdetr_multiseed", "rtdetr_top1"),
                             ("fasterrcnn", "fasterrcnn_multiseed", "fasterrcnn_top1")):
        df = pd.read_csv(YT / folder / f"{folder}_summary.csv").sort_values("seed")
        out[key] = df.rename(columns={col: "det_top1"})
    return out


def fig3_detectors_vs_prior():
    ms = load_multiseed()
    prior = ms["yolo"].coord_top1
    assert all((d.coord_top1.values == prior.values).all() for d in ms.values()), "prior differs across tables"

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 1.95), gridspec_kw={"width_ratios": [1.25, 1, 1], "wspace": 0.55})

    ax = axes[0]
    m, ci = seeds_and_mean(ax, 0, prior, PRIOR, label="Coordinate-only prior")
    print(f"Fig 3a prior {m:.4f} ± {ci:.4f}")
    for i, k in enumerate(DET, start=1):
        m, ci = seeds_and_mean(ax, i, ms[k].det_top1, DET[k], label=DET_NAME[k])
        print(f"Fig 3a {k} {m:.4f} ± {ci:.4f}")
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0.6, 1.0)
    pct(ax)
    ax.set_ylabel("Top-1 accuracy, all labeled teeth")
    ax.set_title("a", loc="left", fontsize=9, x=-0.3, y=1.0)

    ax = axes[1]
    ax.axhspan(0, 5, color="#F3D9D0", lw=0, zorder=0)
    ax.axhspan(15, 30, color="#D6EBE3", lw=0, zorder=0)
    ax.text(2.4, 2.5, "COMMIT (\u22645 pp)", fontsize=5.6, ha="right", va="center")
    ax.text(2.4, 10, "EXTEND", fontsize=5.6, ha="right", va="center", color="#555555")
    ax.text(2.4, 16.8, "PIVOT (\u226515 pp)", fontsize=5.6, ha="right", va="center")
    for i, k in enumerate(DET):
        m, ci = seeds_and_mean(ax, i, ms[k].gap_pp, DET[k])
        print(f"Fig 3b gap {k} {m:.2f} ± {ci:.2f}")
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 30)
    ax.set_ylabel("Detector \u2212 prior (pp)")
    ax.set_title("b", loc="left", fontsize=9, x=-0.36, y=1.0)

    ax = axes[2]
    for i, k in enumerate(DET):
        m, ci = seeds_and_mean(ax, i, ms[k].phi, DET[k])
        print(f"Fig 3c phi {k} {m:.4f} ± {ci:.4f}")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.02, 0.3)
    ax.set_ylabel("Error correlation with prior (\u03c6)")
    ax.set_title("c", loc="left", fontsize=9, x=-0.36, y=1.0)

    for ax in axes:
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=4, handletextpad=0.2,
               columnspacing=1.2)

    save(fig, "fig3_detectors_vs_prior")


# ---------------------------------------------------------------- Figure 4
UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]


def per_tooth_axes(axes, h2h, group, show_legend=False, ylim=(0, 50)):
    g = h2h[h2h.group == group]
    for ax, teeth, jaw in zip(axes, (UPPER, LOWER), ("Maxilla", "Mandible")):
        x = np.arange(16)
        for j, k in enumerate(DET):
            d = g[g.model == k].set_index("fdi").loc[teeth]
            dx = (j - 1) * 0.24
            ax.errorbar(x + dx, 100 * d.delta, yerr=[100 * (d.delta - d.ci_lo), 100 * (d.ci_hi - d.delta)],
                        fmt="o", color=DET[k], ms=2.3, capsize=0, elinewidth=0.8, label=DET_NAME[k])
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(7.5, color="#999999", lw=0.5, ls=":")
        ax.set_xticks(x, [str(t) for t in teeth])
        ax.set_xlim(-0.6, 15.6)
        ax.set_ylim(*ylim)
        ax.text(0.005, 0.97, jaw, transform=ax.transAxes, fontsize=6.5, fontweight="bold", va="top")
        ax.set_ylabel("Δ top-1 (pp)")
    return g


def det_legend(fig, axes, y):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=3, handletextpad=0.1,
               columnspacing=1.2)


def fig4_per_tooth():
    h2h = pd.read_csv(YT / "per_tooth_head_to_head.csv")
    h2h["fdi"] = h2h.fdi.astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH, 2.7), sharey=True, gridspec_kw={"hspace": 0.35})
    g = per_tooth_axes(axes, h2h, "CUDA seeds 0-4")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.1,
                   columnspacing=1.2)
    axes[1].set_xlabel("FDI tooth (patient's right on the left, as on a panoramic radiograph)")
    print(f"Fig 4: {int(g.detector_wins.sum())}/{len(g)} detector-class CIs above zero; "
          f"delta range {100 * g.delta.min():.1f}-{100 * g.delta.max():.1f} pp; "
          f"min CI low {100 * g.ci_lo.min():.1f} pp")
    save(fig, "fig4_per_tooth")

    fig = plt.figure(figsize=(TEXT_WIDTH, 7.6))
    outer = fig.add_gridspec(3, 1, hspace=0.32)
    axes = []
    for i in range(3):
        inner = outer[i].subgridspec(2, 1, hspace=0.42)
        axes += [fig.add_subplot(inner[0]), fig.add_subplot(inner[1])]
    for i, grp in enumerate(["CUDA seeds 0-4", "MPS seeds 5-9", "CUDA seeds 11-14"]):
        gg = per_tooth_axes(axes[2 * i: 2 * i + 2], h2h, grp)
        axes[2 * i].set_title(f"{'abc'[i]}   {grp}", loc="left", fontsize=7.5, x=-0.09)
        print(f"Fig S1 {grp}: {int(gg.detector_wins.sum())}/{len(gg)} above zero, min CI low {100 * gg.ci_lo.min():.1f} pp")
    axes[-1].set_xlabel("FDI tooth")
    det_legend(fig, axes, 0.9)
    save(fig, "figS1_per_tooth_all_groups")


# ---------------------------------------------------------------- Figure 5
def fig5_agreement():
    df = pd.read_csv(YT / "cross_architecture_agreement_summary.csv").sort_values("seed")
    metrics = [("triple_rate", "All three detectors give\nthe same wrong class"),
               ("adjacent_share_pooled", "Shared errors that are\nadjacent-tooth swaps"),
               ("coord_match_rate", "Shared wrong class matches\nthe coordinate prior")]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 1.75), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (col, title), letter in zip(axes, metrics, "abc"):
        for backend, mk, filled in (("CUDA", "o", True), ("MPS", "D", False)):
            d = df[df.backend == backend]
            ax.scatter(d.seed, d[col], marker=mk, s=11, facecolors=PRIOR if filled else "white",
                       edgecolors=PRIOR, linewidths=0.8, label=f"{backend} seeds", zorder=3)
        means = {}
        for backend, ls in (("CUDA", "--"), ("MPS", ":")):
            means[backend] = df[df.backend == backend][col].mean()
            ax.axhline(means[backend], color=PRIOR, lw=0.7, ls=ls, zorder=1)
        ax.text(14.6, 0.715, f"mean CUDA {100 * means['CUDA']:.1f}%\nmean MPS {100 * means['MPS']:.1f}%",
                fontsize=5.6, ha="right", va="bottom", color="#444444", linespacing=1.3)
        ax.set_xticks([0, 2, 4, 6, 8, 11, 14])
        ax.set_xlim(-0.8, 14.8)
        ax.set_ylim(0.7, 1.01)
        ax.set_xlabel("Seed")
        ax.set_title(title, fontweight="normal", fontsize=6.5)
        panel(ax, letter, x=-0.12 if letter != "a" else -0.3)
        print(f"Fig 5 {col}: range {df[col].min():.4f}-{df[col].max():.4f}; CUDA mean {means['CUDA']:.4f} "
              f"(n={int((df.backend == 'CUDA').sum())}), MPS mean {means['MPS']:.4f} (n={int((df.backend == 'MPS').sum())})")
    pct(axes[0])
    axes[0].set_ylabel("Rate")
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [matplotlib.lines.Line2D([], [], color=PRIOR, lw=0.7, ls="--"),
                matplotlib.lines.Line2D([], [], color=PRIOR, lw=0.7, ls=":")]
    labels += ["CUDA mean", "MPS mean"]
    axes[1].legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.2), ncol=4, handletextpad=0.3,
                   columnspacing=1.2)
    save(fig, "fig5_error_agreement")


# ---------------------------------------------------------------- Figure 6
def fig6_backend():
    df = pd.read_csv(YT / "detector_backend_comparison.csv")
    paired = df[df.seed.between(5, 9) & df.group.isin(["CUDA seeds 5-9", "MPS seeds 5-9"])]
    wide = paired.pivot_table(index=["model", "seed"], columns="backend",
                              values=["top1_all_gt", "missed_rate", "top1_matched_only"]).reset_index()
    panels = [("top1_all_gt", "Top-1, all labeled teeth", (0.90, 0.97), [0.90, 0.92, 0.94, 0.96]),
              ("missed_rate", "Missed-tooth rate", (0.0, 0.042), [0.0, 0.01, 0.02, 0.03, 0.04]),
              ("top1_matched_only", "Top-1, matched teeth only", (0.935, 0.975), [0.94, 0.95, 0.96, 0.97])]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 1.75), gridspec_kw={"wspace": 0.55})
    for ax, (col, title, lim, ticks), letter in zip(axes, panels, "abc"):
        ax.plot(lim, lim, color="#999999", lw=0.6, ls="--", zorder=1)
        for k in DET:
            w = wide[wide.model == k]
            ax.scatter(w[(col, "CUDA")], w[(col, "MPS")], s=11, color=DET[k], label=DET_NAME[k], zorder=3, linewidths=0)
            diff = (w[(col, "MPS")] - w[(col, "CUDA")]).abs().max()
            print(f"Fig 6 {col} {k}: max |MPS-CUDA| {diff:.4f}")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        pct(ax)
        pct(ax, "x")
        ax.set_xlabel("CUDA (Kaggle T4)")
        ax.set_ylabel("MPS (Apple Silicon)")
        ax.set_title(title, fontweight="normal", fontsize=6.5)
        panel(ax, letter, x=-0.42)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3, handletextpad=0.1,
                   columnspacing=1.2)
    save(fig, "fig6_backend")


if __name__ == "__main__":
    fig1_coordinate_baseline()
    fig2_ablation_noise()
    fig3_detectors_vs_prior()
    fig4_per_tooth()
    fig5_agreement()
    fig6_backend()
