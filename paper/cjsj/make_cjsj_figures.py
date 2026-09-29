"""Figures for the CJSJ paper (HuangChristopher_paper.docx), built from
committed files only.

Usage: python paper/cjsj/make_cjsj_figures.py

Writes fig1.png to fig4.png (600 dpi, 3.3 in wide, one journal column) next
to this script. Fonts are Times New Roman at 8 pt when it is installed, as
the CJSJ template asks, with Liberation Serif (same metrics) as the fallback.

Fig. 1  UFBA-425 image cate8-00390 with its 31 labelled boxes, coloured by
        FDI quadrant (Dataset/yolo_train_dataset labels).
Fig. 2  Top-1 accuracy on identical held-out teeth, T4 seeds 0-4
        (multiseed summaries; RESULTS.md Sections 33, 40, 45).
Fig. 3  Close-up of the same image: tooth 11 and its mirror, tooth 21
        (case_study_crops/manifest.csv, RESULTS.md Section 23).
Fig. 4  Error agreement on teeth that every model got wrong, three seed
        groups (cross_architecture_agreement_summary.csv and _nulls.csv;
        RESULTS.md Sections 46, 51, 52, 58).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
YT = REPO / "cpu_repro" / "yolo_training" / "eval_results"
CB = REPO / "cpu_repro" / "coord_baseline"
IMAGES = REPO / "Dataset" / "yolo_train_dataset" / "train"
FIG_IMAGE = "cate8-00390_jpg.rf.eb1ab1028af98aa88e55c7302014dc54"

FDI_CODES = [
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
]
# Okabe-Ito colours, one per quadrant.
QUAD_COLOUR = {"1": "#0072B2", "2": "#E69F00", "3": "#009E73", "4": "#CC79A7"}
QUAD_NAME = {"1": "Quadrant 1 (upper right)", "2": "Quadrant 2 (upper left)",
             "3": "Quadrant 3 (lower left)", "4": "Quadrant 4 (lower right)"}
WIDTH = 3.3  # inches, one CJSJ column

available = {f.name for f in fm.fontManager.ttflist}
SERIF = [f for f in ("Times New Roman", "Liberation Serif", "DejaVu Serif") if f in available]
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": SERIF,
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def save(fig, name):
    fig.savefig(HERE / f"{name}.png", dpi=600)
    plt.close(fig)
    print(f"wrote {name}.png")


def load_boxes():
    """Return {fdi: (x0, y0, x1, y1)} in pixels for the Fig. 1 image."""
    img = Image.open(IMAGES / "images" / f"{FIG_IMAGE}.jpg").convert("L")
    w, h = img.size
    boxes = {}
    for line in (IMAGES / "labels" / f"{FIG_IMAGE}.txt").read_text().split("\n"):
        if not line.strip():
            continue
        c, x, y, bw, bh = line.split()
        x, y, bw, bh = (float(v) for v in (x, y, bw, bh))
        boxes[FDI_CODES[int(c)]] = ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)
    return np.asarray(img), boxes


def midline_x(boxes):
    return (boxes["11"][2] + boxes["21"][0]) / 2


def label(ax, x, y, text, colour, size=7, va="center"):
    ax.text(x, y, text, color=colour, fontsize=size, fontweight="bold", ha="center", va=va,
            bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.9))


def fig1():
    img, boxes = load_boxes()
    xs = [b[0] for b in boxes.values()] + [b[2] for b in boxes.values()]
    ys = [b[1] for b in boxes.values()] + [b[3] for b in boxes.values()]
    x0, x1 = max(min(xs) - 8, 0), min(max(xs) + 8, img.shape[1])
    y0, y1 = max(min(ys) - 26, 0), min(max(ys) + 26, img.shape[0])
    fig, ax = plt.subplots(figsize=(WIDTH, WIDTH * (y1 - y0) / (x1 - x0) + 0.35))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    for code, (bx0, by0, bx1, by1) in boxes.items():
        colour = QUAD_COLOUR[code[0]]
        ax.add_patch(mpatches.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, lw=0.6, ec=colour))
    # Labels above upper teeth and below lower teeth, staggered so neighbours do not collide.
    for q in "1234":
        codes = sorted((c for c in boxes if c[0] == q), key=lambda c: c[1])
        for i, code in enumerate(codes):
            bx0, by0, bx1, by1 = boxes[code]
            cx = (bx0 + bx1) / 2
            step = 9 if i % 2 else 0
            if q in "12":
                label(ax, cx, by0 - 7 - step, code, QUAD_COLOUR[q], va="center")
            else:
                label(ax, cx, by1 + 7 + step, code, QUAD_COLOUR[q], va="center")
    mx = midline_x(boxes)
    ax.plot([mx, mx], [y0, y1], color="white", lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    for q, (x, y, va) in {"1": (x0 + (mx - x0) / 2, y0, "bottom"), "2": (mx + (x1 - mx) / 2, y0, "bottom"),
                          "4": (x0 + (mx - x0) / 2, y1, "top"), "3": (mx + (x1 - mx) / 2, y1, "top")}.items():
        ax.text(x, y + (-3 if va == "bottom" else 3), QUAD_NAME[q], color=QUAD_COLOUR[q], fontsize=8,
                fontweight="bold", ha="center", va=va)
    save(fig, "fig1")


def fig2():
    runs = {
        "YOLOv8x": pd.read_csv(YT / "multiseed" / "multiseed_summary.csv")["yolo_top1"],
        "RT-DETR-l": pd.read_csv(YT / "rtdetr_multiseed" / "rtdetr_multiseed_summary.csv")["rtdetr_top1"],
        "Faster R-CNN": pd.read_csv(YT / "fasterrcnn_multiseed" / "fasterrcnn_multiseed_summary.csv")["fasterrcnn_top1"],
    }
    prior = pd.read_csv(YT / "multiseed" / "multiseed_summary.csv")["coord_top1"]
    majority = pd.read_csv(CB / "summary.csv").set_index("classifier").loc["gradient_boosted_tree",
                                                                           "majority_baseline_acc_mean"]
    names = ["Most common\ntooth (guess)", "Position only"] + list(runs)
    values = [np.array([majority])] + [prior] + list(runs.values())
    colours = ["#D0D0D0", "#9A9A9A", "#2F5B87", "#3F7DB3", "#70A8CC"]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.95))
    y = np.arange(len(names))[::-1]
    for yi, v, c in zip(y, values, colours):
        v = 100 * np.asarray(v, float)
        m = v.mean()
        ax.barh(yi, m, height=0.62, color=c, edgecolor="#808080" if yi == y[0] else c, lw=0.6)
        if len(v) > 1:
            ax.errorbar(m, yi, xerr=[[m - v.min()], [v.max() - m]], fmt="none", ecolor="#1A1A1A",
                        elinewidth=0.8, capsize=1.8, capthick=0.8)
        ax.text(max(m, v.max()) + 1.5, yi, f"{m:.1f}", va="center", ha="left", fontsize=8, fontweight="bold")
    pm = 100 * prior.mean()
    ax.axvline(pm, color="#7A7A7A", lw=0.7, ls=(0, (3, 2)), zorder=0)
    det_lo = min(100 * r.mean() for r in runs.values())
    ay = y[-1] - 0.95
    ax.annotate("", xy=(det_lo, ay), xytext=(pm, ay),
                arrowprops=dict(arrowstyle="<->", color="#B22222", lw=0.9, shrinkA=0, shrinkB=0))
    gaps = [100 * (r.mean() - prior.mean()) for r in runs.values()]
    ax.text(pm + 1.5, ay - 0.45, f"{min(gaps):.0f} to {max(gaps):.0f} points", color="#B22222",
            ha="left", va="center", fontsize=8)
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 108)
    ax.set_ylim(ay - 0.75, y[0] + 0.5)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Correct tooth number (percent)")
    ax.grid(axis="x", color="#E3E3E3", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    save(fig, "fig2")


def fig3():
    img, boxes = load_boxes()
    man = pd.read_csv(YT / "case_study_crops" / "manifest.csv")
    case = man[man.crop_file.str.startswith("case_07_")].iloc[0]
    assert FIG_IMAGE in case.source_image and case.true_fdi == 11 and case.coord_pred_fdi == 21
    b11, b21 = boxes["11"], boxes["21"]
    cx = (b11[0] + b21[2]) / 2
    half = 2.3 * (b21[2] - b11[0])
    x0, x1 = cx - half, cx + half
    y0 = min(b11[1], b21[1]) - 30
    y1 = max(b11[3], b21[3]) + 30
    fig, ax = plt.subplots(figsize=(WIDTH, WIDTH * (y1 - y0) / (x1 - x0)))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    for b, c in ((b11, "#009E4F"), (b21, "#D62728")):
        ax.add_patch(mpatches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=1.2, ec=c))
    mx = midline_x(boxes)
    ax.plot([mx, mx], [y0 + 10, y1 - 12], color="#6A7FDB", lw=0.9, ls=(0, (3, 2)))
    label(ax, mx, y0 + 5, "midline", "#3949AB", size=8)
    ax.text(mx - 3, max(b11[3], b21[3]) + 14, "True tooth: 11", color="#007A3D", fontsize=8, fontweight="bold",
            ha="right", va="center", bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.9))
    ax.text(mx + 3, max(b11[3], b21[3]) + 14, "Its mirror: 21", color="#B71C1C", fontsize=8, fontweight="bold",
            ha="left", va="center", bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.9))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    save(fig, "fig3")


def fig4():
    ag = pd.read_csv(YT / "cross_architecture_agreement_summary.csv")
    nulls = pd.read_csv(YT / "cross_architecture_agreement_nulls.csv")
    groups = [("Kaggle T4\nseeds 0 to 4", range(0, 5)), ("MacBook Air M3\nseeds 5 to 9", range(5, 10)),
              ("Kaggle T4\nseeds 11 to 14", range(11, 15))]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    lo, hi = 100 * nulls.null_mean.min(), 100 * nulls.null_mean.max()
    ax.axhspan(lo, hi, color="#F2C4C4", lw=0, zorder=0.5)
    for yv in (lo, hi):
        ax.axhline(yv, color="#B22222", lw=0.7, ls=(0, (3, 2)), zorder=4)
    w = 0.36
    for i, (name, seeds) in enumerate(groups):
        g = ag[ag.seed.isin(seeds)]
        for dx, col, c in ((-w / 2, "triple_rate", "#2F5B87"), (w / 2, "coord_match_rate", "#8DBF67")):
            v = 100 * g[col]
            m = v.mean()
            ax.bar(i + dx, m, width=w, color=c, zorder=2)
            ax.errorbar(i + dx, m, yerr=[[m - v.min()], [v.max() - m]], fmt="none", ecolor="#1A1A1A",
                        elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=3)
            ax.text(i + dx, v.max() + 2, f"{m:.0f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", zorder=4)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Agreement (percent)")
    ax.grid(axis="y", color="#E3E3E3", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    handles = [mpatches.Patch(color="#2F5B87", label="All three detectors pick the same wrong tooth"),
               mpatches.Patch(color="#8DBF67", label="That tooth is also the position-only guess"),
               mpatches.Patch(fc="#F2C4C4", ec="#B22222", ls=(0, (3, 2)), lw=0.7,
                              label=f"Chance level in permutation tests ({lo:.0f} to {hi:.0f}%)")]
    ax.legend(handles=handles, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 1.02),
              handlelength=1.2, borderaxespad=0.1)
    save(fig, "fig4")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
