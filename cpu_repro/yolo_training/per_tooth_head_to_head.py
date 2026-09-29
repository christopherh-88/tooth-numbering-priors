"""Per-tooth head-to-head: detector vs coordinate-only geometry prior, paired bootstrap.

For each FDI tooth class and each detector (YOLOv8x, RT-DETR-l, Faster R-CNN),
delta = detector top-1 minus coord-prior top-1 over the same labeled teeth
(all-GT: a tooth the detector missed counts as wrong, as in
detector_backend_comparison.py; joined_seed{N}.csv rows are one per labeled
tooth with `*_correct` False for misses). The CI is a paired bootstrap that
resamples whole images (seed, image_id) with replacement, so teeth from the
same image stay together.

Data: eval_results/{multiseed,rtdetr_multiseed,fasterrcnn_multiseed}/joined_seed{N}.csv,
reported as three separate groups, never pooled across backends:
CUDA seeds 0-4 (the original analysis, RESULTS.md Section 50), MPS seeds 5-9
(tables from build_joined_mps_seeds.py) and CUDA seeds 11-14 (tables from
build_joined_cuda_10_14.py; seed 10 has no three-detector table). Every group's
coordinate prior uses the same HistGradientBoostingClassifier recipe, refit on
that seed's split. Pooling seeds within a group reuses images when test sets
overlap, so CIs treat (seed, image) as independent clusters and are
descriptive, not a formal test. Each group's bootstrap uses the same fixed RNG
seed, so the seed 0-4 numbers are unchanged by the added groups.

Writes eval_results/per_tooth_head_to_head.csv (one row per group, model and
FDI class).
"""
import csv
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent / "eval_results"
FDI = [f"{q}{n}" for q in (1, 2, 3, 4) for n in range(1, 9)]
MODELS = {"yolo": ("multiseed", "yolo_correct"),
          "rtdetr": ("rtdetr_multiseed", "rtdetr_correct"),
          "fasterrcnn": ("fasterrcnn_multiseed", "fasterrcnn_correct")}
GROUPS = {"CUDA seeds 0-4": range(0, 5),
          "MPS seeds 5-9": range(5, 10),
          "CUDA seeds 11-14": range(11, 15)}
N_BOOT = 2000
RNG_SEED = 0


def load(model, seeds):
    folder, col = MODELS[model]
    rows = []
    for s in seeds:
        for r in csv.DictReader(open(EVAL / folder / f"joined_seed{s}.csv")):
            rows.append((f"{s}:{r['image_id']}", int(r["true_class"]),
                         r[col] == "True", r["coord_correct"] == "True"))
    return rows


def analyse(model, group, seeds):
    rows = load(model, seeds)
    clusters = sorted({r[0] for r in rows})
    cidx = {c: i for i, c in enumerate(clusters)}
    n = np.zeros((len(clusters), 32))
    det = np.zeros_like(n)
    prior = np.zeros_like(n)
    for cl, cls, d, p in rows:
        n[cidx[cl], cls] += 1
        det[cidx[cl], cls] += d
        prior[cidx[cl], cls] += p
    rng = np.random.default_rng(RNG_SEED)
    boots = rng.integers(0, len(clusters), size=(N_BOOT, len(clusters)))
    out = []
    for c in range(32):
        tot = n[:, c].sum()
        if tot == 0:
            continue
        delta = (det[:, c].sum() - prior[:, c].sum()) / tot
        bn = n[boots, c].sum(1)
        ok = bn > 0
        bd = (det[boots, c].sum(1) - prior[boots, c].sum(1))[ok] / bn[ok]
        lo, hi = np.percentile(bd, [2.5, 97.5])
        out.append({"group": group, "model": model, "fdi": FDI[c], "n_teeth": int(tot),
                    "detector_acc": det[:, c].sum() / tot, "prior_acc": prior[:, c].sum() / tot,
                    "delta": delta, "ci_lo": lo, "ci_hi": hi,
                    "detector_wins": lo > 0, "prior_wins": hi < 0})
    pooled = (det.sum() / n.sum(), prior.sum() / n.sum())
    return out, len(clusters), pooled


def main():
    allrows = []
    for group, seeds in GROUPS.items():
        for model in MODELS:
            allrows += report(model, group, seeds)
    dest = EVAL / "per_tooth_head_to_head.csv"
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(allrows)
    print(f"Saved {dest}")


def report(model, group, seeds):
    rows, n_clusters, (det_acc, prior_acc) = analyse(model, group, seeds)
    print(f"=== {group}, {model}: {len(rows)} tooth classes, {n_clusters} (seed, image) clusters, "
          f"{sum(r['n_teeth'] for r in rows)} teeth ===")
    print(f"  pooled detector acc {det_acc:.4f}  prior acc {prior_acc:.4f}  gap {100*(det_acc - prior_acc):+.1f}pp")
    print(f"  detector significantly better: {sum(r['detector_wins'] for r in rows)}/{len(rows)}; "
          f"prior significantly better: {sum(r['prior_wins'] for r in rows)}/{len(rows)}")
    d = [r["delta"] for r in rows]
    print(f"  per-class delta (points): min {100*min(d):+.1f}  median {100*float(np.median(d)):+.1f}  max {100*max(d):+.1f}")
    for r in sorted(rows, key=lambda r: r["delta"])[:3]:
        print(f"  smallest gain FDI {r['fdi']}: {100*r['delta']:+.1f} [{100*r['ci_lo']:+.1f}, {100*r['ci_hi']:+.1f}] "
              f"det {r['detector_acc']:.3f} vs prior {r['prior_acc']:.3f} (n={r['n_teeth']})")
    print()
    return rows


if __name__ == "__main__":
    main()
