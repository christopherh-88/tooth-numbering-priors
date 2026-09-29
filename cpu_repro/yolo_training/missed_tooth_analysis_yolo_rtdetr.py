"""Which labeled teeth do YOLOv8x and RT-DETR-l miss, on the MPS seeds (5-9),
the CUDA seeds 10-14, and the two YOLOv8x CUDA spike seeds (7, 9)?

Companion to missed_tooth_analysis.py (Faster R-CNN only). Motivated by the
YOLOv8x missed-tooth-rate spikes noted in BACKEND_COMPARISON.md: CUDA seeds 7
and 9 at about 4%, and MPS seed 8 at 3.0%. Per-tooth data sources:
per_tooth_predictions_ultralytics_mps.py (MPS seeds 5-9),
per_tooth_predictions_cuda_10_14.py (CUDA RT-DETR-l 10-14, YOLOv8x 11-14; no
YOLOv8x seed-10 checkpoint was downloaded), and
per_tooth_predictions_yolo_cuda_7_9.py (YOLOv8x CUDA seeds 7 and 9).

CUDA and MPS runs with the same seed number share the same val split, so the
spike-seed report compares each CUDA spike run against the MPS YOLOv8x run on
the identical set of teeth.

Reads the per_tooth.csv tables directly (they already carry normalized boxes
and true/pred class. No label-file lookup needed, unlike
missed_tooth_analysis.py's seed 0-4 CUDA path):
eval_results/{yolo_mps,rtdetr_mps}/seed{5-9}, eval_results/seed{7,9,11-14},
eval_results/rtdetr/seed{10-14}, and eval_results/fasterrcnn/seed9 (FDI 24
cross-model check). The spike report also reads every run's summary.csv for
seeds 5-9 (same-split table), and its outlier scan reads every per_tooth.csv
under eval_results plus the CUDA seed 0-4 joined_seed{N}.csv tables.
"""
import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FDI = [f"{q}{n}" for q in (1, 2, 3, 4) for n in range(1, 9)]
EDGE_MARGIN = 0.03
SEEDS = range(5, 10)
MODELS = {"yolo": "yolo_mps", "rtdetr": "rtdetr_mps"}
CUDA_SEEDS = {"yolo": [11, 12, 13, 14], "rtdetr": [10, 11, 12, 13, 14]}
CUDA_SUBDIR = {"yolo": None, "rtdetr": "rtdetr"}  # None -> bare eval_results/seed{N}
SPIKE_SEEDS = [7, 9]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def rows_for(model, seeds=None, cuda=False):
    by_img_all_seeds = []
    for seed in (seeds if seeds is not None else SEEDS):
        if cuda:
            subdir = CUDA_SUBDIR[model]
            path = (HERE / "eval_results" / subdir / f"seed{seed}" / "per_tooth.csv" if subdir
                    else HERE / "eval_results" / f"seed{seed}" / "per_tooth.csv")
        else:
            path = HERE / "eval_results" / MODELS[model] / f"seed{seed}" / "per_tooth.csv"
        by_img = defaultdict(list)
        for r in csv.DictReader(open(path)):
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


def report(label, rows, q1, q3, seeds=None, area_cutoff_note="this model's 5 MPS seeds"):
    n, m = len(rows), sum(r["missed"] for r in rows)
    print(f"=== {label}: {m} missed / {n} labeled teeth ({100*m/n:.2f}%) ===\n")

    def show(title, groups):
        print(title)
        for name, rs in groups:
            if rs:
                k = sum(r["missed"] for r in rs)
                print(f"  {name:<22} n={len(rs):>5}  missed={k:>3}  rate={100*k/len(rs):5.2f}%  share_of_misses={100*k/max(m,1):5.1f}%")
        print()

    show("By seed:", [(f"seed {s}", [r for r in rows if r["seed"] == s]) for s in (seeds if seeds is not None else SEEDS)])
    show("By edge proximity (within 3% of a border):",
         [("near edge", [r for r in rows if r["edge"]]), ("interior", [r for r in rows if not r["edge"]])])
    show(f"By box area quartile (cutoffs from {area_cutoff_note}):",
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

        cuda_seeds = CUDA_SEEDS[model]
        cuda_rows = rows_for(model, seeds=cuda_seeds, cuda=True)
        cuda_areas = sorted(r["area"] for r in cuda_rows)
        cq1, cq3 = cuda_areas[len(cuda_areas) // 4], cuda_areas[3 * len(cuda_areas) // 4]
        report(f"{model} CUDA seeds {cuda_seeds}", cuda_rows, cq1, cq3, seeds=cuda_seeds,
               area_cutoff_note=f"this model's CUDA seeds {cuda_seeds}")

    spike_report()


def miss_rate(path, cls=None, exclude_cls=None):
    rs = [r for r in csv.DictReader(open(path))
          if (cls is None or int(r["true_class"]) == cls)
          and (exclude_cls is None or int(r["true_class"]) != exclude_cls)]
    return sum(r["pred_class"] == "" for r in rs), len(rs)


def spike_report():
    spike_rows = rows_for("yolo", seeds=SPIKE_SEEDS, cuda=True)
    areas = sorted(r["area"] for r in spike_rows)
    q1, q3 = areas[len(areas) // 4], areas[3 * len(areas) // 4]
    for s in SPIKE_SEEDS:
        report(f"yolo CUDA spike seed {s}", [r for r in spike_rows if r["seed"] == s], q1, q3, seeds=[s],
               area_cutoff_note=f"YOLOv8x CUDA seeds {SPIKE_SEEDS} pooled")

    print("=== Spike runs, box-area quartile miss rates (each run's own cutoffs) vs. pooled MPS 5-9 ===\n")
    mps_rows = rows_for("yolo")
    spike_runs = [("CUDA seed 7", [r for r in spike_rows if r["seed"] == 7]),
                  ("MPS seed 8", [r for r in mps_rows if r["seed"] == 8]),
                  ("CUDA seed 9", [r for r in spike_rows if r["seed"] == 9]),
                  ("MPS seeds 5-9 pooled", mps_rows)]
    for name, rs in spike_runs:
        a = sorted(r["area"] for r in rs)
        c1, c3 = a[len(a) // 4], a[3 * len(a) // 4]
        cells = []
        for keep in (lambda r: r["area"] <= c1, lambda r: c1 < r["area"] < c3, lambda r: r["area"] >= c3):
            g = [r for r in rs if keep(r)]
            cells.append(f"{100 * sum(r['missed'] for r in g) / len(g):.2f}%")
        print(f"  {name:<21} small {cells[0]:>6}  mid {cells[1]:>6}  large {cells[2]:>6}")
    print()

    print("=== Same-split check: CUDA vs. MPS YOLOv8x per-tooth tables ===\n")
    for s in SPIKE_SEEDS:
        keys = [{(r["image_id"], r["gt_idx"]) for r in csv.DictReader(open(p))}
                for p in (HERE / "eval_results" / f"seed{s}" / "per_tooth.csv",
                          HERE / "eval_results" / "yolo_mps" / f"seed{s}" / "per_tooth.csv")]
        print(f"  seed {s}: identical (image, tooth) sets: {keys[0] == keys[1]} ({len(keys[0])} teeth)")
    print()

    print("=== Spike seeds vs. MPS YOLOv8x on the identical val split ===\n")
    for s in SPIKE_SEEDS:
        for name, path in (("CUDA", HERE / "eval_results" / f"seed{s}" / "per_tooth.csv"),
                           ("MPS", HERE / "eval_results" / "yolo_mps" / f"seed{s}" / "per_tooth.csv")):
            m, n = miss_rate(path)
            print(f"  seed {s} YOLOv8x {name:<5} missed {m:>3}/{n}  ({100*m/n:.2f}%)")
    print()

    fdi24 = FDI.index("24")
    print("=== Seed 9 split, FDI 24 miss rate by model ===\n")
    for name, path in (("YOLOv8x CUDA", HERE / "eval_results" / "seed9" / "per_tooth.csv"),
                       ("YOLOv8x MPS", HERE / "eval_results" / "yolo_mps" / "seed9" / "per_tooth.csv"),
                       ("RT-DETR-l MPS", HERE / "eval_results" / "rtdetr_mps" / "seed9" / "per_tooth.csv"),
                       ("Faster R-CNN MPS", HERE / "eval_results" / "fasterrcnn" / "seed9" / "per_tooth.csv")):
        m, n = miss_rate(path, cls=fdi24)
        print(f"  {name:<17} missed {m:>2}/{n}  ({100*m/n:.1f}%)")
    missed_imgs = {r["image_id"] for r in csv.DictReader(open(HERE / "eval_results" / "seed9" / "per_tooth.csv"))
                   if int(r["true_class"]) == fdi24 and r["pred_class"] == ""}
    m, n = miss_rate(HERE / "eval_results" / "seed9" / "per_tooth.csv", exclude_cls=fdi24)
    print(f"  YOLOv8x CUDA FDI 24 misses fall in {len(missed_imgs)} distinct images")
    print(f"  YOLOv8x CUDA seed 9 excluding FDI 24: missed {m}/{n} ({100*m/n:.2f}%)\n")

    outlier_scan()

    # Same-split control for all three YOLOv8x spikes (MPS 8, CUDA 7 and 9): every run on seeds 5-9.
    # Rates come from summary.csv (n_test + missed = labeled teeth), available for all six runs.
    runs = (("RT-DETR MPS", "rtdetr_mps"), ("RT-DETR CUDA", "rtdetr"), ("FRCNN MPS", "fasterrcnn"),
            ("FRCNN CUDA", "fasterrcnn_cuda"), ("YOLO MPS", "yolo_mps"), ("YOLO CUDA", ""))
    print("=== Missed-tooth rate by run, seeds 5-9 (same seed = same split) ===\n")
    print("  seed  " + "  ".join(f"{name:>12}" for name, _ in runs))
    for seed in SEEDS:
        cells = []
        for _, sub in runs:
            s = next(csv.DictReader(open(HERE / "eval_results" / sub / f"seed{seed}" / "summary.csv")))
            m = int(s["n_unmatched_gt_missed_detections"])
            cells.append(f"{100 * m / (int(s['n_test']) + m):11.2f}%")
        print(f"  {seed:>4}  " + "  ".join(cells))
    print()


def outlier_scan(min_n=30, top=4):
    """Highest per-class miss rates across every per-tooth table and the CUDA seed 0-4 joined tables."""
    rates = []
    per_tooth = sorted(HERE.glob("eval_results/*/seed*/per_tooth.csv")) + sorted(HERE.glob("eval_results/seed*/per_tooth.csv"))
    for path in per_tooth:
        counts = defaultdict(lambda: [0, 0])
        for r in csv.DictReader(open(path)):
            c = counts[int(r["true_class"])]
            c[0] += r["pred_class"] == ""
            c[1] += 1
        rates += [(m / n, m, n, FDI[cls], str(path.relative_to(HERE / "eval_results"))) for cls, (m, n) in counts.items() if n >= min_n]
    joined_max = 0.0
    for sub, col in (("multiseed", "yolo_pred"), ("fasterrcnn_multiseed", "fasterrcnn_pred"), ("rtdetr_multiseed", "rtdetr_pred")):
        for s in range(5):
            counts = defaultdict(lambda: [0, 0])
            for r in csv.DictReader(open(HERE / "eval_results" / sub / f"joined_seed{s}.csv")):
                c = counts[int(r["true_class"])]
                c[0] += r[col] in ("", "nan")
                c[1] += 1
            joined_max = max([joined_max] + [m / n for m, n in counts.values() if n >= min_n])
    rates.sort(reverse=True)
    print(f"=== Per-class miss-rate outlier scan (classes with n >= {min_n}) ===\n")
    print(f"  {len(per_tooth)} per-tooth tables; highest rates:")
    for rate, m, n, fdi, path in rates[:top]:
        print(f"    {100 * rate:5.1f}%  FDI {fdi}  {m}/{n}  {path}")
    print(f"  CUDA seed 0-4 joined tables (all three detectors): highest rate {100 * joined_max:.1f}%\n")


if __name__ == "__main__":
    main()
