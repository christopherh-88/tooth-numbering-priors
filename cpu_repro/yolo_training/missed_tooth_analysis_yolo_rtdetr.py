"""Which labeled teeth do YOLOv8x and RT-DETR-l miss, on the MPS seeds (5-9)?

Companion to missed_tooth_analysis.py (Faster R-CNN only). Motivated by the
YOLOv8x missed-tooth-rate spikes noted in BACKEND_COMPARISON.md: CUDA seeds 7
and 9 at about 4%, and MPS seed 8 at 3.0%. Only the MPS seeds have per-tooth
data (per_tooth_predictions_ultralytics_mps.py); the two CUDA spikes (seeds 7,
9) cannot be broken down further without a per-tooth rerun on Kaggle, which
has not been done - this script does not explain those two, only MPS seed 8
and the general size/edge/overlap pattern.

Reads eval_results/{yolo_mps,rtdetr_mps}/seed{5-9}/per_tooth.csv directly
(these already carry normalized boxes and true/pred class - no label-file
lookup needed, unlike missed_tooth_analysis.py's seed 0-4 CUDA path).
"""
import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FDI = [f"{q}{n}" for q in (1, 2, 3, 4) for n in range(1, 9)]
EDGE_MARGIN = 0.03
SEEDS = range(5, 10)
MODELS = {"yolo": "yolo_mps", "rtdetr": "rtdetr_mps"}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def rows_for(model):
    subdir = MODELS[model]
    by_img_all_seeds = []
    for seed in SEEDS:
        by_img = defaultdict(list)
        for r in csv.DictReader(open(HERE / "eval_results" / subdir / f"seed{seed}" / "per_tooth.csv")):
            by_img[r["image_id"]].append(r)
        by_img_all_seeds.append((seed, by_img))

    rows = []
    for seed, by_img in by_img_all_seeds:
        for teeth in by_img.values():
            boxes = [tuple(float(t[k]) for k in ("x1", "y1", "x2", "y2")) for t in teeth]
            for i, t in enumerate(teeth):
                b = boxes[i]
                others = [o for j, o in enumerate(boxes) if j != i]
                rows.append({
                    "seed": seed, "cls": int(t["true_class"]), "missed": t["pred_class"] == "",
                    "edge": min(b[0], b[1], 1 - b[2], 1 - b[3]) < EDGE_MARGIN,
                    "area": (b[2] - b[0]) * (b[3] - b[1]),
                    "max_iou": max((iou(b, o) for o in others), default=0.0),
                })
    return rows


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

    show("By seed:", [(f"seed {s}", [r for r in rows if r["seed"] == s]) for s in SEEDS])
    show("By edge proximity (within 3% of a border):",
         [("near edge", [r for r in rows if r["edge"]]), ("interior", [r for r in rows if not r["edge"]])])
    show("By box area quartile (cutoffs from this model's 5 MPS seeds):",
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


def main():
    for model in MODELS:
        rows = rows_for(model)
        areas = sorted(r["area"] for r in rows)
        q1, q3 = areas[len(areas) // 4], areas[3 * len(areas) // 4]
        report(f"{model} MPS seeds 5-9", rows, q1, q3)


if __name__ == "__main__":
    main()
