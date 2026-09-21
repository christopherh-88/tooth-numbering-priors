"""Which labeled teeth does Faster R-CNN miss? (seeds 0-4 CUDA from joined tables, seeds 5-9 MPS from per_tooth.csv)

A tooth is "missed" when fasterrcnn_pred is empty in joined_seed{N}.csv, i.e. no
predicted box matched the labeled box at ty.MATCH_IOU_THRESHOLD. Breaks misses
down by FDI class, distance of the box to the image edge, box size, and overlap
with the nearest neighboring labeled box.
"""
import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOINED = HERE / "eval_results" / "fasterrcnn_multiseed"
FDI = [f"{q}{n}" for q in (1, 2, 3, 4) for n in range(1, 9)]
EDGE_MARGIN = 0.03  # box within 3% of image width/height of a border


def val_label_dirs(seed):
    prepared = HERE / "prepared" / (f"seed{seed}" if seed else "")
    return {Path(p).stem: Path(p).parents[1] / "labels" for p in (prepared / "val.txt").read_text().splitlines()}


def load_boxes(path):
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls, x, y, w, h = int(p[0]), *map(float, p[1:5])
        out.append((cls, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def rows_cuda(seed):
    """Seeds 0-4: joined table + label files (Kaggle CUDA runs)."""
    dirs = val_label_dirs(seed)
    rows = []
    for r in csv.DictReader(open(JOINED / f"joined_seed{seed}.csv")):
        boxes = load_boxes(dirs[Path(r["label_file"]).stem] / r["label_file"])
        i = int(r["line_idx"])
        rows.append(describe(seed, int(r["true_class"]), r["fasterrcnn_pred"] in ("", "nan"),
                             boxes[i][1:], [o[1:] for j, o in enumerate(boxes) if j != i]))
    return rows


def rows_mps(seed):
    """Seeds 5-9: per_tooth.csv from per_tooth_predictions_mps.py."""
    by_img = defaultdict(list)
    for r in csv.DictReader(open(HERE / "eval_results" / "fasterrcnn" / f"seed{seed}" / "per_tooth.csv")):
        by_img[r["image_id"]].append(r)
    rows = []
    for teeth in by_img.values():
        boxes = [tuple(float(t[k]) for k in ("x1", "y1", "x2", "y2")) for t in teeth]
        for i, t in enumerate(teeth):
            rows.append(describe(seed, int(t["true_class"]), t["pred_class"] == "",
                                 boxes[i], [o for j, o in enumerate(boxes) if j != i]))
    return rows


def describe(seed, cls, missed, b, others):
    return {
        "seed": seed, "cls": cls, "missed": missed,
        "edge": min(b[0], b[1], 1 - b[2], 1 - b[3]) < EDGE_MARGIN,
        "area": (b[2] - b[0]) * (b[3] - b[1]),
        "max_iou": max((iou(b, o) for o in others), default=0.0),
    }


def report(label, rows, q1, q3):
    n, m = len(rows), sum(r["missed"] for r in rows)
    print(f"=== {label}: {m} missed / {n} labeled teeth ({100*m/n:.2f}%) ===\n")

    def show(title, groups):
        print(title)
        for name, rs in groups:
            if rs:
                k = sum(r["missed"] for r in rs)
                print(f"  {name:<22} n={len(rs):>5}  missed={k:>3}  rate={100*k/len(rs):5.2f}%  share_of_misses={100*k/max(m,1):5.1f}%")
        print()

    show("By edge proximity (within 3% of a border):",
         [("near edge", [r for r in rows if r["edge"]]), ("interior", [r for r in rows if not r["edge"]])])
    show("By box area quartile (cutoffs from all 10 seeds):",
         [("smallest quarter", [r for r in rows if r["area"] <= q1]),
          ("middle half", [r for r in rows if q1 < r["area"] < q3]),
          ("largest quarter", [r for r in rows if r["area"] >= q3])])
    show("By overlap with nearest labeled neighbor (IoU):",
         [("no overlap (0)", [r for r in rows if r["max_iou"] == 0]),
          ("light (0-0.1)", [r for r in rows if 0 < r["max_iou"] <= 0.1]),
          ("heavy (>0.1)", [r for r in rows if r["max_iou"] > 0.1])])
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["cls"]].append(r)
    ranked = sorted(by_cls.items(), key=lambda kv: -sum(r["missed"] for r in kv[1]))[:8]
    show("Top 8 FDI classes by number of misses:", [(f"FDI {FDI[c]}", rs) for c, rs in ranked])
    return by_cls


def main():
    cuda = [r for s in range(5) for r in rows_cuda(s)]
    mps = [r for s in range(5, 10) for r in rows_mps(s)]
    areas = sorted(r["area"] for r in cuda + mps)
    q1, q3 = areas[len(areas) // 4], areas[3 * len(areas) // 4]
    by_cls = {}
    for label, rows in (("CUDA seeds 0-4", cuda), ("MPS seeds 5-9", mps), ("all 10 seeds (description only)", cuda + mps)):
        by_cls[label] = report(label, rows, q1, q3)

    dest = HERE / "eval_results" / "missed_tooth_by_class.csv"
    with open(dest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "fdi", "n", "missed", "miss_rate"])
        for label, groups in by_cls.items():
            for c in sorted(groups):
                rs = groups[c]
                k = sum(r["missed"] for r in rs)
                w.writerow([label, FDI[c], len(rs), k, k / len(rs)])
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
