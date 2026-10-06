"""Causal diagram of tooth numbering errors, with the evidence for each edge.

    python paper/miccai/make_causal_diagram.py

Writes causal_diagram.png and causal_diagram.pdf next to this script.
Solid edges are tested (RESULTS.md section in the label), dashed edges are
assumed or not yet tested.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
NODES = {
    "patient": (0.10, 0.88, "Patient size and\nhead position"),
    "number": (0.10, 0.60, "True FDI number"),
    "gap": (0.10, 0.28, "Missing neighbor\n(gap in the arch)"),
    "box": (0.45, 0.88, "Box position\nand shape"),
    "drift": (0.45, 0.62, "Neighbors close\nthe space"),
    "look": (0.45, 0.38, "Tooth appearance\n(crown, roots)"),
    "space": (0.45, 0.12, "Visible empty\nspace"),
    "pos": (0.84, 0.88, "Position-only\nmodel"),
    "det": (0.84, 0.45, "Detectors\n(YOLOv8x, RT-DETR-l,\nFaster R-CNN)"),
}
# (from, to, label, label position, tested, curve)
EDGES = [
    ("patient", "box", "", None, False, 0),
    ("number", "box", "", None, True, 0),
    ("number", "look", "", None, True, 0),
    ("gap", "drift", "assumed", (0.385, 0.505), False, 0),
    ("gap", "space", "", None, True, 0),
    ("drift", "box", "closed gaps:\nAUROC 0.78 (72)", (0.53, 0.745), True, 0),
    ("box", "pos", "78.5% top-1 (65)", (0.645, 0.905), True, 0),
    ("box", "det", "absolute position\nbarely used (63, 67)", (0.70, 0.70), True, 0),
    ("look", "det", "nearby context needed\nby 2 of 3 (68)", (0.645, 0.335), True, 0),
    ("space", "det", "erasing a tooth: 2 to 6%\nper detector, rarely\nshared (71)", (0.66, 0.21), True, 0.15),
]
W, H = 0.20, 0.12


def edge_points(a, b):
    (x0, y0), (x1, y1) = a, b
    dx, dy = x1 - x0, y1 - y0
    s = min(W / 2 / abs(dx) if dx else 1e9, H / 2 / abs(dy) if dy else 1e9)
    return (x0 + dx * s, y0 + dy * s), (x1 - dx * s, y1 - dy * s)


def main():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(-0.02, 0.98)
    ax.set_ylim(-0.12, 1.0)
    ax.axis("off")
    for x, y, text in NODES.values():
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.01",
                                    fc="#f4f4f4", ec="#333333", lw=1.2))
        ax.text(x, y, text, ha="center", va="center", fontsize=10)
    for a, b, label, at, tested, curve in EDGES:
        p, q = edge_points(NODES[a][:2], NODES[b][:2])
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.4,
                                     color="#222222" if tested else "#888888",
                                     linestyle="-" if tested else (0, (4, 3)),
                                     connectionstyle=f"arc3,rad={curve}"))
        if label:
            ax.text(*at, label, ha="center", va="center", fontsize=8,
                    color="#222222" if tested else "#666666",
                    bbox=dict(fc="white", ec="none", pad=1))
    ax.text(0.84, -0.04, "Shared wrong number at real gaps:\n44.7% of joint failures sit next to a gap,\n"
            "and 93% of those get the missing tooth's\nnumber (69). It happens at closed gaps (72);\n"
            "erasing a tooth does not reproduce it (71).", ha="center", va="center", fontsize=8.5,
            bbox=dict(fc="#fff8e6", ec="#c9a227", boxstyle="round,pad=0.4"))
    ax.text(-0.01, 0.98, "Solid: tested (RESULTS.md section). Dashed: assumed or not yet tested.",
            fontsize=8.5, color="#444444")
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"causal_diagram.{ext}", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
