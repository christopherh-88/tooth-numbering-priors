"""Figures for the CJSJ paper (HuangChristopher_paper.docx), built from
committed files only.

Usage: python paper/cjsj/make_cjsj_figures.py

Writes fig1.png to fig3.png at 600 dpi next to this script. Fig. 1 and
Fig. 2 are 7.0 in wide (full page width), Fig. 3 is 3.4 in wide (one
column). Text is 8 pt Times (TeX Gyre Termes, or Times New Roman or
Liberation Serif when available), as the CJSJ template asks for figure
labels.

Fig. 1  UFBA-425 image cate8-00390. (a) Its 31 labelled boxes, coloured by
        FDI quadrant. (b) Close-up of tooth 11 and its mirror, tooth 21
        (case_study_crops/manifest.csv, RESULTS.md Section 23). (c) Box
        centres of every labelled tooth, first copy of each source image
        (Dataset/yolo_train_dataset labels).
Fig. 2  (a) Top-1 accuracy on identical held-out teeth, T4 seeds 0-4
        (multiseed summaries; Sections 33, 40, 45). (b) Per-class accuracy,
        pooled over seeds 0-4 (per_tooth_head_to_head.csv; Section 51).
        (c) Position-only accuracy on four datasets (coord_baseline and
        boundary_condition summaries; Sections 2, 10, 17, 25).
Fig. 3  Teeth that every model got wrong, three seed groups
        (cross_architecture_agreement_summary.csv and _nulls.csv;
        Sections 46, 51, 52, 58).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
YT = REPO / "cpu_repro" / "yolo_training" / "eval_results"
CB = REPO / "cpu_repro" / "coord_baseline"
DP = REPO / "cpu_repro" / "boundary_condition" / "denpar_periapical"
IMAGES = REPO / "Dataset" / "yolo_train_dataset" / "train"
FIG_IMAGE = "cate8-00390_jpg.rf.eb1ab1028af98aa88e55c7302014dc54"

# Section 17 (dual-labeled dataset, another clinic). The dataset is third
# party and not committed, so these two values are quoted from RESULTS.md.
DUAL_TOP1, DUAL_MAJORITY = 0.329, 0.0376

FDI_CODES = [
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
]
# Quadrant colours (Fig. 1 only), chosen to stay distinct from the detector colours.
QUAD_COLOUR = {"1": "#E377C2", "2": "#F0E442", "3": "#56B4E9", "4": "#B5E08A"}
QUAD_NAME = {"1": "Quadrant 1 (upper right)", "2": "Quadrant 2 (upper left)",
             "3": "Quadrant 3 (lower left)", "4": "Quadrant 4 (lower right)"}
QUAD_SHORT = {"1": "Upper right (1)", "2": "Upper left (2)", "3": "Lower left (3)", "4": "Lower right (4)"}
# One colour and marker per model, the same in every figure (Okabe-Ito, colour-blind safe).
POS = "#6E6E6E"
DET = {"YOLOv8x": "#0072B2", "RT-DETR-l": "#D55E00", "Faster R-CNN": "#009E73"}
DET_KEY = {"YOLOv8x": "yolo", "RT-DETR-l": "rtdetr", "Faster R-CNN": "fasterrcnn"}
DET_MARK = {"YOLOv8x": "o", "RT-DETR-l": "s", "Faster R-CNN": "^"}
WIDTH = 7.0   # inches, full page width
COLUMN = 3.4  # inches, one column
INK = "#000000"  # pure black, matching the body text
BAND = "#D6E4F0"  # light blue: spread across the three detectors

# Times for all figure text: TeX Gyre Termes (a Times clone) is registered from
# the TeX tree when present; Times New Roman is used first if installed.
for f in Path("/usr/share/texmf/fonts/opentype/public/tex-gyre").glob("texgyretermes-*.otf"):
    fm.fontManager.addfont(str(f))
_available = {f.name for f in fm.fontManager.ttflist}
SERIF = [f for f in ("Times New Roman", "TeX Gyre Termes", "Liberation Serif", "DejaVu Serif") if f in _available]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": SERIF,
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 3,
    "legend.frameon": False,
    "legend.handletextpad": 0.4,
    "savefig.facecolor": "white",
})


def darken_text(fig):
    """Give black text a hairline outline in its own colour so the thin Times strokes
    stay solid black when Word or a PDF viewer scales the image down."""
    for t in fig.findobj(plt.Text):
        if t.get_text() and to_rgba(t.get_color())[:3] == (0.0, 0.0, 0.0):
            t.set_path_effects([pe.withStroke(linewidth=0.12, foreground="black")])


def save(fig, name):
    darken_text(fig)
    fig.savefig(HERE / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {name}.png")


def panel(fig, ax, letter, title, dx=0.0):
    """Bold panel letter at the axes' top-left corner, with a short title beside it."""
    x0 = ax.get_position().x0 - dx
    y1 = ax.get_position().y1
    fig.text(x0, y1 + 0.035, letter, ha="left", va="bottom", fontweight="bold", fontsize=9)
    fig.text(x0 + 0.018, y1 + 0.035, title, ha="left", va="bottom", fontsize=8)


def inch_axes(fig, left, bottom, width, height):
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


# ---------------------------------------------------------------- Figure 1
def load_boxes():
    """Return the image and {fdi: (x0, y0, x1, y1)} in pixels."""
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


def tag(ax, x, y, text, colour, size=6, va="center", ha="center"):
    ax.text(x, y, text, color=colour, fontsize=size, fontweight="bold", ha=ha, va=va,
            bbox=dict(boxstyle="square,pad=0.12", fc="black", ec="none", alpha=0.6))


def all_box_centres():
    """Box centres (normalised x, y) and FDI code of every labelled tooth, one copy per source image."""
    root = REPO / "Dataset" / "yolo_train_dataset"
    first = {}
    for split in ("train", "valid", "test"):
        for f in sorted((root / split / "labels").glob("*.txt")):
            first.setdefault(f.name.split("_jpg")[0], f)
    rows = []
    for f in first.values():
        for line in f.read_text().split("\n"):
            parts = line.split()
            if not parts:
                continue
            v = [float(t) for t in parts[1:]]
            if len(v) == 4:          # box: x_center y_center width height
                x, y = v[0], v[1]
            else:                    # polygon: x1 y1 x2 y2 ... -> centre of its bounding box
                x = (min(v[0::2]) + max(v[0::2])) / 2
                y = (min(v[1::2]) + max(v[1::2])) / 2
            rows.append((x, y, FDI_CODES[int(parts[0])]))
    return pd.DataFrame(rows, columns=["x", "y", "fdi"])


def fig1():
    img, boxes = load_boxes()
    man = pd.read_csv(YT / "case_study_crops" / "manifest.csv")
    case = man[man.crop_file.str.startswith("case_07_")].iloc[0]
    assert FIG_IMAGE in case.source_image and case.true_fdi == 11
    assert case.coord_pred_fdi == 21 and case.yolo_pred_fdi == 11
    assert len(boxes) == 31 and "38" not in boxes

    xs = [b[0] for b in boxes.values()] + [b[2] for b in boxes.values()]
    ys = [b[1] for b in boxes.values()] + [b[3] for b in boxes.values()]
    ax0, ax1 = max(min(xs) - 10, 0), min(max(xs) + 10, img.shape[1])
    ay0, ay1 = max(min(ys) - 30, 0), min(max(ys) + 30, img.shape[0])
    b11, b21 = boxes["11"], boxes["21"]
    mx = midline_x(boxes)
    half = 1.9 * (b21[2] - b11[0])
    bx0, bx1 = mx - half, mx + half
    by0, by1 = min(b11[1], b21[1]) - 34, max(b11[3], b21[3]) + 34

    asp_a = (ax1 - ax0) / (ay1 - ay0)
    asp_b = (bx1 - bx0) / (by1 - by0)
    gap, c_left, c_w, right = 0.1, 0.6, 1.55, 0.04   # inches
    top, bottom = 0.24, 0.52
    h = (WIDTH - gap - c_left - c_w - right) / (asp_a + asp_b)
    H = h + top + bottom
    fig = plt.figure(figsize=(WIDTH, H))
    pa = inch_axes(fig, 0, bottom, h * asp_a, h)
    pb = inch_axes(fig, h * asp_a + gap, bottom, h * asp_b, h)
    c_x = h * (asp_a + asp_b) + gap + c_left
    pc = inch_axes(fig, c_x, bottom, c_w, h)

    # (a) whole test image with its labelled boxes
    pa.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
    for code, (x0, y0, x1, y1) in boxes.items():
        pa.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.6, ec=QUAD_COLOUR[code[0]]))
    for q in "1234":
        codes = sorted((c for c in boxes if c[0] == q), key=lambda c: c[1])
        for i, code in enumerate(codes):
            x0, y0, x1, y1 = boxes[code]
            cx, step = (x0 + x1) / 2, (12 if i % 2 else 0)
            y = y0 - 9 - step if q in "12" else y1 + 9 + step
            tag(pa, cx, y, code, QUAD_COLOUR[q], size=5.5)
    pa.plot([mx, mx], [ay0, ay1], color="white", lw=0.6, ls=(0, (3, 2)))
    pa.set_xlim(ax0, ax1)
    pa.set_ylim(ay1, ay0)
    pa.axis("off")

    # (b) close-up of tooth 11 and its mirror, tooth 21
    pb.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
    for bb, q in ((b11, "1"), (b21, "2")):
        pb.add_patch(mpatches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False, lw=1.1,
                                        ec=QUAD_COLOUR[q]))
    pb.plot([mx, mx], [by0 + 16, by1 - 4], color="white", lw=0.7, ls=(0, (3, 2)))
    tag(pb, mx, by0 + 8, "midline", "white", size=6.5)
    base = max(b11[3], b21[3]) + 16
    tag(pb, (b11[0] + b11[2]) / 2 - 3, base, "11", QUAD_COLOUR["1"], size=7, ha="right")
    tag(pb, (b21[0] + b21[2]) / 2 + 3, base, "21", QUAD_COLOUR["2"], size=7, ha="left")
    pb.set_xlim(bx0, bx1)
    pb.set_ylim(by1, by0)
    pb.axis("off")

    # (c) where every labelled tooth sits: box centres across the whole dataset
    ctr = all_box_centres()
    for q in "1234":
        sub = ctr[ctr.fdi.str[0] == q]
        pc.scatter(sub.x, sub.y, s=0.6, color=QUAD_COLOUR[q], alpha=0.35, lw=0, rasterized=True, zorder=2)
    for code, sub in ctr.groupby("fdi"):
        if code[1] in "18":
            dx = {"1": -0.03, "2": 0.03, "3": 0.03, "4": -0.03}[code[0]] if code[1] == "1" else 0
            ha = "center" if dx == 0 else ("right" if dx < 0 else "left")
            pc.text(sub.x.median() + dx, sub.y.median(), code, ha=ha, va="center", fontsize=6, fontweight="bold",
                    color="white", bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.6),
                    zorder=3)
    pc.axvline(0.5, color="#9E9E9E", lw=0.5, ls=(0, (3, 2)), zorder=1)
    pc.set_xlim(0.05, 0.95)
    pc.set_ylim(0.85, 0.15)
    pc.set_xticks([0.2, 0.5, 0.8])
    pc.set_yticks([0.2, 0.5, 0.8])
    pc.set_xlabel("Horizontal box center\n(fraction of image width)", labelpad=2)
    pc.set_ylabel("Vertical box center\n(fraction of image height)", labelpad=2)
    pc.set_facecolor("#FAFAFA")

    W = WIDTH
    for x, letter, title in ((0, "a", "Held-out test image, 31 labeled teeth"),
                             ((h * asp_a + gap) / W, "b", "Teeth 11 and 21"),
                             ((c_x - 0.55) / W, "c", "Box centers of every labeled tooth")):
        fig.text(x, 1, letter, ha="left", va="top", fontweight="bold", fontsize=9)
        fig.text(x + 0.018, 1, title, ha="left", va="top", fontsize=8)
    handles = [mpatches.Patch(fc=QUAD_COLOUR[q], ec="#555555", lw=0.4, label=QUAD_SHORT[q]) for q in "1234"]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(h * asp_a / 2 / W, 0.0), ncol=2,
               handlelength=1.0, columnspacing=1.0, borderaxespad=0.2, labelspacing=0.3)
    fig.text((h * asp_a + gap + h * asp_b / 2) / W, 0.3 / H, "Position-only model: 21", ha="center", va="center",
             fontsize=8)
    fig.text((h * asp_a + gap + h * asp_b / 2) / W, 0.13 / H, "YOLOv8x: 11 (correct)", ha="center",
             va="center", fontsize=8)
    save(fig, "fig1")


# ---------------------------------------------------------------- Figure 2
def fig2():
    runs = {name: pd.read_csv(YT / d / f"{p}_summary.csv")[f"{DET_KEY[name]}_top1"]
            for name, d, p in (("YOLOv8x", "multiseed", "multiseed"),
                               ("RT-DETR-l", "rtdetr_multiseed", "rtdetr_multiseed"),
                               ("Faster R-CNN", "fasterrcnn_multiseed", "fasterrcnn_multiseed"))}
    prior = pd.read_csv(YT / "multiseed" / "multiseed_summary.csv")["coord_top1"]
    ufba = pd.read_csv(CB / "summary.csv").set_index("classifier").loc["gradient_boosted_tree"]
    dentex = pd.read_csv(CB / "dentex" / "summary.csv").set_index("classifier").loc["gradient_boosted_tree"]
    denpar = pd.read_csv(DP / "summary.csv").set_index("classifier").loc["gradient_boosted_tree"]

    fig = plt.figure(figsize=(WIDTH, 2.72))
    ax_h, ax_b = 1.62, 0.66   # axes height and bottom, inches
    pa = inch_axes(fig, 0.74, ax_b, 1.1, ax_h)
    pb = inch_axes(fig, 2.4, ax_b, 2.5, ax_h)
    pc = inch_axes(fig, 5.62, ax_b, 1.30, ax_h)

    # (a) top-1 accuracy by model on the same teeth: five seeds, mean and 95% CI
    from scipy import stats
    rows = [("Position-only\nmodel", prior, POS, "o")] + [(n, v, DET[n], DET_MARK[n]) for n, v in runs.items()]
    y = np.arange(len(rows))[::-1].astype(float)
    pm = 100 * prior.mean()
    rng = np.random.default_rng(1)
    for xv in (70, 80, 90, 100):
        pa.axvline(xv, color="#EBEBEB", lw=0.5, zorder=0)
    for yi, (name, v, c, m) in zip(y, rows):
        v = 100 * np.asarray(v, float)
        mean = v.mean()
        half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        pa.plot(v, yi + rng.uniform(-0.17, 0.17, len(v)) - 0.0, "o", ms=2.2, color=c, alpha=0.45, mec="none",
                zorder=2)
        pa.errorbar(mean, yi, xerr=half, fmt=m, ms=4.2, color=c, mec=c, elinewidth=0.8, capsize=2,
                    capthick=0.8, zorder=3)
        pa.text(mean + half + 1.8, yi, f"{mean:.1f}", va="center", ha="left", fontsize=8)
    lo = min(100 * r.mean() for r in runs.values())
    gaps = [100 * (r.mean() - prior.mean()) for r in runs.values()]
    yb = y[-1] - 0.8
    pa.plot([pm, pm], [yb + 0.25, y[0] - 0.3], color=POS, lw=0.5, ls=(0, (3, 2)), zorder=1)
    pa.plot([lo, lo], [yb + 0.25, y[-1] - 0.3], color=POS, lw=0.5, ls=(0, (3, 2)), zorder=1)
    pa.annotate("", xy=(lo, yb), xytext=(pm, yb),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.6, shrinkA=0, shrinkB=0, mutation_scale=6))
    pa.text((pm + lo) / 2, yb - 0.15, f"+{min(gaps):.0f} to +{max(gaps):.0f} points", ha="center", va="top",
            fontsize=8)
    pa.text(62.6, y[0] + 0.62, f"Most common tooth: {100 * ufba.majority_baseline_acc_mean:.1f}%", ha="left",
            va="bottom", fontsize=7.5, color=INK)
    pa.set_yticks(y, [r[0] for r in rows])
    pa.tick_params(axis="y", length=0, pad=3)
    pa.spines["left"].set_visible(False)
    pa.set_xlim(62, 104)
    pa.set_ylim(y[-1] - 1.55, y[0] + 1.0)
    pa.set_xticks([70, 80, 90, 100])
    pa.set_xlabel("Correct tooth number (%)")
    panel(fig, pa, "a", "Accuracy on the same test teeth", dx=0.1)

    # (b) accuracy by tooth class, seeds 0-4 pooled
    h2h = pd.read_csv(YT / "per_tooth_head_to_head.csv")
    g = h2h[h2h.group == "CUDA seeds 0-4"]
    assert g.detector_wins.all()
    prior_by = g[g.model == "yolo"].set_index("fdi").prior_acc
    step = 9  # one empty slot between quadrants
    for q in range(4):
        codes = FDI_CODES[8 * q:8 * q + 8]
        x = q * step + np.arange(8)
        pb.plot(x, [100 * prior_by[int(c)] for c in codes], "-o", color=POS, lw=0.8, ms=3.2, mfc="white",
                mew=0.8, zorder=3)
        det = np.array([[100 * g[g.model == DET_KEY[n]].set_index("fdi").detector_acc[int(c)] for c in codes]
                        for n in DET])
        pb.fill_between(x, det.min(0), det.max(0), color=BAND, lw=0, zorder=1)
        for k, name in enumerate(DET):
            pb.plot(x + (k - 1) * 0.18, det[k], ls="none", marker=DET_MARK[name], color=DET[name], ms=2.4,
                    mec="none", zorder=4)
        pb.text(q * step + 3.5, -0.115, QUAD_NAME[str(q + 1)].split(" (")[0], ha="center", va="top",
                transform=pb.get_xaxis_transform(), fontsize=8)
    ticks = [q * step + i for q in range(4) for i in range(8)]
    pb.set_xticks(ticks, [str(i + 1) for q in range(4) for i in range(8)])
    pb.tick_params(axis="x", length=2, pad=1.5)
    pb.set_xlim(-0.8, 3 * step + 7.8)
    pb.set_ylim(40, 101)
    pb.set_yticks([40, 60, 80, 100])
    for yv in (60, 80, 100):
        pb.axhline(yv, color="#EBEBEB", lw=0.5, zorder=0)
    pb.set_ylabel("Correct tooth number (%)")
    pb.set_xlabel("Tooth position from the midline (1 to 8) in each quadrant", labelpad=14)
    panel(fig, pb, "b", "Accuracy for each of the 32 tooth classes", dx=0.055)

    # (c) position-only accuracy by dataset
    ds = [("UFBA-425\n(panoramic)", ufba.top1_acc_mean, ufba.top1_acc_ci95, ufba.majority_baseline_acc_mean),
          ("DENTEX\n(panoramic)", dentex.top1_acc_mean, dentex.top1_acc_ci95, dentex.majority_baseline_acc_mean),
          ("Other clinic\n(panoramic)", DUAL_TOP1, None, DUAL_MAJORITY),
          ("DenPAR\n(periapical)", denpar.top1_acc_mean, denpar.top1_acc_ci95, denpar.majority_baseline_acc_mean)]
    y = np.arange(len(ds))[::-1].astype(float)
    for yi, (name, acc, ci, maj) in zip(y, ds):
        pc.plot([100 * maj, 100 * acc], [yi, yi], color="#BDBDBD", lw=0.8, zorder=1)
        if ci is not None:
            pc.errorbar(100 * acc, yi, xerr=100 * ci, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.8,
                        capthick=0.6, zorder=2)
        pc.plot(100 * maj, yi, "o", ms=4, mfc="white", mec=POS, mew=0.8, zorder=3)
        pc.plot(100 * acc, yi, "o", ms=4.5, color=POS, mec=POS, zorder=3)
        pc.text(100 * (acc + (ci or 0)) + 4, yi, f"{100 * acc:.1f}", va="center", ha="left", fontsize=8)
    pc.set_yticks(y, [d[0] for d in ds])
    pc.tick_params(axis="y", length=0, pad=3)
    pc.spines["left"].set_visible(False)
    pc.set_xlim(0, 90)
    pc.set_ylim(y[-1] - 0.5, y[0] + 0.5)
    pc.set_xticks([0, 20, 40, 60, 80])
    pc.set_xlabel("Correct tooth number (%)")
    panel(fig, pc, "c", "Position-only model by dataset", dx=0.1)

    handles = [mlines.Line2D([], [], ls="-", lw=0.7, marker="o", ms=4, mfc="white", mew=0.7, color=POS,
                             label="Position-only model")]
    handles += [mlines.Line2D([], [], ls="none", marker=DET_MARK[n], ms=4, mec="none", color=DET[n], label=n)
                for n in DET]
    handles.append(mpatches.Patch(color=BAND, label="Detector range"))
    handles += [mlines.Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec=POS, mew=0.8,
                              label="Most common tooth")]
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0), columnspacing=1.5,
               handlelength=1.4)
    save(fig, "fig2")


# ---------------------------------------------------------------- Figure 3
def fig3():
    ag = pd.read_csv(YT / "cross_architecture_agreement_summary.csv")
    nulls = pd.read_csv(YT / "cross_architecture_agreement_nulls.csv")
    lo, hi = 100 * nulls.null_mean.min(), 100 * nulls.null_mean.max()
    groups = [("Seeds 0 to 4 (Kaggle T4)", range(0, 5), "o", "#1A1A1A"),
              ("Seeds 5 to 9 (MacBook Air M3)", range(5, 10), "s", "#5A5A5A"),
              ("Seeds 11 to 14 (Kaggle T4)", range(11, 15), "^", "#8C8C8C")]
    metrics = [("triple_rate", "All three pick\nthe same wrong\ntooth", True),
               ("coord_match_rate", "That tooth is the\nposition-only\nmodel's guess", True),
               ("adjacent_share_pooled", "That tooth is a\nneighbor of the\ntrue tooth", False)]
    fig = plt.figure(figsize=(COLUMN, 2.55))
    ax = inch_axes(fig, 0.5, 0.62, 2.82, 1.83)
    rng = np.random.default_rng(0)
    for k, (col, _, chance) in enumerate(metrics):
        if chance:
            ax.fill_between([k - 0.42, k + 0.42], lo, hi, color="#E3E3E3", lw=0, zorder=0.5)
        for j, (_, seeds, m, c) in enumerate(groups):
            v = 100 * ag[ag.seed.isin(seeds)][col].to_numpy()
            xc = k + (j - 1) * 0.26
            ax.plot(xc + rng.uniform(-0.045, 0.045, len(v)), v, "o", ms=1.8, color="#B0B0B0", mec="none", zorder=2)
            ax.plot([xc + 0.08, xc + 0.08], [v.min(), v.max()], color=c, lw=0.8, zorder=3)
            ax.plot(xc + 0.08, v.mean(), marker=m, ms=4.2, color=c, mec="white", mew=0.4, zorder=4)
    ax.text(1.0, hi + 2, f"Chance in permutation tests ({lo:.0f} to {hi:.0f}%)", ha="center", va="bottom",
            fontsize=8, color=INK)
    ax.set_xticks(range(3), [mm[1] for mm in metrics])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    for yv in (25, 50, 75, 100):
        ax.axhline(yv, color="#EBEBEB", lw=0.5, zorder=0)
    ax.set_ylabel("Share of joint failures (%)")
    handles = [mlines.Line2D([], [], ls="-", lw=0.8, marker=m, ms=4.2, color=c, mec="white", mew=0.4, label=n)
               for n, _, m, c in groups]
    handles.append(mlines.Line2D([], [], ls="none", marker="o", ms=2.5, color="#B0B0B0", mec="none",
                                 label="Single seed"))
    ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.52), ncol=1, handlelength=1.5,
              labelspacing=0.3)
    save(fig, "fig3")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
