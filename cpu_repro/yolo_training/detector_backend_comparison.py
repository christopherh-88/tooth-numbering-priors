"""YOLOv8x / RT-DETR-l / Faster R-CNN backend comparison: Kaggle CUDA vs local MPS.

Same metric convention as backend_comparison.py: headline is all-GT top-1
(missed teeth count as wrong); secondary are missed-tooth rate and
matched-only top-1.

Sources per model (CUDA seeds 5-14 and MPS seeds 5-9 come from summary.csv;
all-GT is top1_acc * n_test / (n_test + n_unmatched_gt_missed_detections)):
  yolo:        CUDA 0-4 eval_results/multiseed/joined_seed{N}.csv
               CUDA 5-14 eval_results/seed{N}/summary.csv
               MPS 5-9 eval_results/yolo_mps/seed{N}/summary.csv
  rtdetr:      CUDA 0-4 eval_results/rtdetr_multiseed/joined_seed{N}.csv
               CUDA 5-14 eval_results/rtdetr/seed{N}/summary.csv
               MPS 5-9 eval_results/rtdetr_mps/seed{N}/summary.csv
  fasterrcnn:  CUDA 0 eval_results/fasterrcnn_multiseed/joined_seed0.csv
               CUDA 1-4 eval_results/fasterrcnn/seed{N}/summary.csv
               CUDA 5-14 eval_results/fasterrcnn_cuda/seed{N}/summary.csv
               MPS 5-9 eval_results/fasterrcnn/seed{N}/summary.csv
Use the folder (seed) name, not the summary.csv seed column: the Kaggle
Faster R-CNN seeds 5-9 outputs were written by an older script that
hard-coded seed=0 in that column.

Groups are not pooled across backends. Seeds 5-9 CUDA vs MPS share splits, so
that pair is paired by seed. Every expected seed that has no result is listed
as MISSING, not silently skipped. Writes eval_results/detector_backend_comparison.csv.
"""
import csv
import statistics as st
from pathlib import Path

EVAL = Path(__file__).resolve().parent / "eval_results"
SEEDS = range(15)

JOINED = {"yolo": ("multiseed", "yolo_pred", "yolo_correct"),
          "rtdetr": ("rtdetr_multiseed", "rtdetr_pred", "rtdetr_correct"),
          "fasterrcnn": ("fasterrcnn_multiseed", "fasterrcnn_pred", "fasterrcnn_correct")}
MODEL_ORDER = ("yolo", "rtdetr", "fasterrcnn")


def cuda_summary(model, s):
    if model == "yolo":
        return EVAL / f"seed{s}" / "summary.csv"
    if model == "rtdetr":
        return EVAL / "rtdetr" / f"seed{s}" / "summary.csv"
    if s < 5:  # Faster R-CNN seeds 1-4 (CUDA) live in the MPS-named folder
        return EVAL / "fasterrcnn" / f"seed{s}" / "summary.csv"
    return EVAL / "fasterrcnn_cuda" / f"seed{s}" / "summary.csv"


def mps_summary(model, s):
    folder = {"yolo": "yolo_mps", "rtdetr": "rtdetr_mps", "fasterrcnn": "fasterrcnn"}[model]
    return EVAL / folder / f"seed{s}" / "summary.csv"


def from_joined(model, seed):
    d, pred, corr = JOINED[model]
    path = EVAL / d / f"joined_seed{seed}.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    found = [r for r in rows if r[pred] not in ("", "nan")]
    return len(rows), len(found), sum(r[corr] == "True" for r in rows)


def from_summary(path):
    if not path.exists():
        return None
    r = next(csv.DictReader(open(path)))
    n_matched = int(r["n_test"])
    return (n_matched + int(r["n_unmatched_gt_missed_detections"]), n_matched,
            round(float(r["top1_acc"]) * n_matched))


def group_of(s, backend):
    if backend == "MPS":
        return "MPS seeds 5-9"
    return "CUDA seeds 0-4" if s < 5 else "CUDA seeds 5-9" if s < 10 else "CUDA seeds 10-14"


def expected():
    """(seed, backend) cells that should have a result."""
    return [(s, "CUDA") for s in SEEDS] + [(s, "MPS") for s in range(5, 10)]


def collect(model):
    out, missing = [], []
    for s, backend in expected():
        if backend == "CUDA":
            use_joined = s < 5 and (model != "fasterrcnn" or s == 0)
            d = from_joined(model, s) if use_joined else from_summary(cuda_summary(model, s))
        else:
            d = from_summary(mps_summary(model, s))
        if d is None:
            missing.append((s, backend))
            continue
        n_gt, n_matched, correct = d
        out.append({"model": model, "group": group_of(s, backend), "backend": backend, "seed": s,
                    "n_gt": n_gt, "n_matched": n_matched, "top1_all_gt": correct / n_gt,
                    "missed_rate": (n_gt - n_matched) / n_gt, "top1_matched_only": correct / n_matched})
    return out, missing


def summarize(label, rs):
    v = [r["top1_all_gt"] for r in rs]
    sd = st.stdev(v) if len(v) > 1 else float("nan")
    print(f"  {label:<17} n={len(v):>2} all-GT mean {st.mean(v):.4f} sd {sd:.4f}  "
          f"missed {st.mean(r['missed_rate'] for r in rs):.4f}  "
          f"matched-only {st.mean(r['top1_matched_only'] for r in rs):.4f}")


def main():
    allrows = []
    for model in MODEL_ORDER:
        rows, missing = collect(model)
        allrows += rows
        print(f"=== {model} ===")
        for r in rows:
            print(f"  seed {r['seed']:>2} {r['group']:<17} all-GT {r['top1_all_gt']:.4f}  "
                  f"missed {r['missed_rate']:.4f}  matched-only {r['top1_matched_only']:.4f}")
        for s, b in missing:
            print(f"  seed {s:>2} {b} MISSING (not run, not downloaded, or failed)")
        print()
        for g in dict.fromkeys(r["group"] for r in rows):
            summarize(g, [r for r in rows if r["group"] == g])
        cuda_all = [r for r in rows if r["backend"] == "CUDA"]
        summarize("CUDA all seeds", cuda_all)
        cuda = {r["seed"]: r["top1_all_gt"] for r in rows if r["group"] == "CUDA seeds 5-9"}
        mps = {r["seed"]: r["top1_all_gt"] for r in rows if r["group"] == "MPS seeds 5-9"}
        both = sorted(set(cuda) & set(mps))
        if both:
            diffs = [mps[s] - cuda[s] for s in both]
            sd = st.stdev(diffs) if len(diffs) > 1 else float("nan")
            print(f"  paired (same split) MPS-CUDA over seeds {both}: mean {st.mean(diffs):+.4f} sd {sd:.4f}  "
                  f"per-seed {[round(d, 4) for d in diffs]}")
        print()

    if not allrows:
        print("no results found")
        return
    dest = EVAL / "detector_backend_comparison.csv"
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
