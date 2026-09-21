"""Faster R-CNN backend comparison: Kaggle CUDA (seeds 0-4) vs local MPS (seeds 5-9).

Headline metric: all-GT top-1 (every labeled tooth is in the denominator; a
tooth the detector missed counts as wrong). Secondary: missed-tooth rate and
matched-only top-1 (accuracy over teeth the detector found).

Seed 0 comes from the original joined table; seeds 1+ come from
eval_results/fasterrcnn/seed{N}/summary.csv (all-GT is recovered as
top1_acc * n_test / (n_test + n_unmatched_gt_missed_detections); this
reproduces fasterrcnn_multiseed_summary.csv exactly for seeds 1-4).

The two groups are reported separately and not pooled. Each seed has its own
image split, so backend is confounded with split; no equivalence claim is made.
"""
import csv
import statistics as st
from pathlib import Path

EVAL = Path(__file__).resolve().parent / "eval_results"
CUDA_SEEDS = range(0, 5)
MPS_SEEDS = range(5, 10)


def load_seed(seed: int):
    if seed == 0:
        rows = list(csv.DictReader(open(EVAL / "fasterrcnn_multiseed" / "joined_seed0.csv")))
        found = [r for r in rows if r["fasterrcnn_pred"] not in ("", "nan")]
        n_gt, n_matched = len(rows), len(found)
        correct = sum(r["fasterrcnn_correct"] == "True" for r in rows)
        return n_gt, n_matched, correct
    path = EVAL / "fasterrcnn" / f"seed{seed}" / "summary.csv"
    if not path.exists():
        return None
    r = next(csv.DictReader(open(path)))
    n_matched = int(r["n_test"])
    n_gt = n_matched + int(r["n_unmatched_gt_missed_detections"])
    return n_gt, n_matched, round(float(r["top1_acc"]) * n_matched)


def main():
    out, groups = [], {"CUDA": [], "MPS": []}
    for backend, seeds in (("CUDA", CUDA_SEEDS), ("MPS", MPS_SEEDS)):
        for s in seeds:
            d = load_seed(s)
            if d is None:
                continue
            n_gt, n_matched, correct = d
            row = {
                "seed": s, "backend": backend, "n_gt": n_gt, "n_matched": n_matched,
                "top1_all_gt": correct / n_gt,
                "missed_rate": (n_gt - n_matched) / n_gt,
                "top1_matched_only": correct / n_matched,
            }
            out.append(row)
            groups[backend].append(row)

    print(f"{'seed':>4} {'backend':>7} {'all-GT top1':>12} {'missed':>8} {'matched-only':>13} {'n_gt':>6}")
    for r in out:
        print(f"{r['seed']:>4} {r['backend']:>7} {r['top1_all_gt']:>12.4f} {r['missed_rate']:>8.4f} "
              f"{r['top1_matched_only']:>13.4f} {r['n_gt']:>6}")
    print()
    for b, rs in groups.items():
        if not rs:
            continue
        for k in ("top1_all_gt", "missed_rate", "top1_matched_only"):
            v = [r[k] for r in rs]
            sd = st.stdev(v) if len(v) > 1 else float("nan")
            print(f"{b:>4} n={len(v)} {k:<18} mean {st.mean(v):.4f}  sd {sd:.4f}  range {min(v):.4f}-{max(v):.4f}")

    dest = EVAL / "backend_comparison.csv"
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
