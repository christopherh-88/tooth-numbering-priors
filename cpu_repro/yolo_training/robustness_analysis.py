"""CPU-only robustness analyses on already-saved predictions - no new
training or GPU use. Covers three of the six review-driven checks:

1. Per-FDI-class breakdown of the YOLO vs. coordinate-baseline gap
   (Section 21/22 only report aggregates).
2. Leverage-point sensitivity check on Section 26's phi coefficient
   (does one FDI class drive the "1.98x independence" result?).
3. Bootstrap CIs on Section 21/26's headline numbers, resampled at the
   image level (not the instance level) to respect within-image
   correlation between multiple teeth in the same radiograph.

Reads cpu_repro/yolo_training/eval_results/error_correlation_joined.csv
(already computed, Section 26) - does not re-run any model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coord_baseline"))
from build_coord_baseline import FDI_CODES, quadrant_of, tooth_type_of  # noqa: E402

JOINED_CSV = Path(__file__).resolve().parent / "eval_results" / "error_correlation_joined.csv"
OUT_DIR = Path(__file__).resolve().parent / "eval_results"
RNG_SEED = 0
N_BOOTSTRAP = 2000


def phi_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float)
    b = b.astype(float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def task1_per_class_breakdown(df: pd.DataFrame):
    rows = []
    for class_id in range(32):
        sub = df[df["true_class"] == class_id]
        if len(sub) == 0:
            continue
        coord_acc = sub["coord_correct"].mean()
        yolo_acc = sub["yolo_correct"].mean()
        rows.append({
            "fdi_code": FDI_CODES[class_id],
            "class_id": class_id,
            "n_instances": len(sub),
            "coord_acc": coord_acc,
            "yolo_acc": yolo_acc,
            "delta_yolo_minus_coord": yolo_acc - coord_acc,
        })
    out = pd.DataFrame(rows).sort_values("delta_yolo_minus_coord").reset_index(drop=True)
    out.to_csv(OUT_DIR / "per_class_breakdown.csv", index=False)

    print("=" * 70)
    print("TASK 1: per-FDI-class accuracy, coord vs YOLO (seed 0)")
    print("=" * 70)
    print(out.to_string(index=False))

    reversals = out[out["delta_yolo_minus_coord"] < 0]
    print(f"\nClasses where YOLO < coordinate-only: {len(reversals)}")
    if len(reversals):
        print(reversals.to_string(index=False))

    # Share of the aggregate gap contributed by each class: (yolo_correct -
    # coord_correct) summed per class, as a fraction of the total net gain.
    out["net_correct_gain"] = (df.groupby("true_class")["yolo_correct"].sum()
                                - df.groupby("true_class")["coord_correct"].sum()).reindex(out["class_id"]).values
    total_gain = out["net_correct_gain"].sum()
    out["pct_of_total_gap"] = 100 * out["net_correct_gain"] / total_gain
    top5 = out.sort_values("pct_of_total_gap", ascending=False).head(5)
    print(f"\nTotal net correct-count gain (YOLO - coord) across all classes: {total_gain:.0f}")
    print("Top 5 classes by share of that total gain:")
    print(top5[["fdi_code", "net_correct_gain", "pct_of_total_gap"]].to_string(index=False))
    print(f"Top-5 classes account for {top5['pct_of_total_gap'].sum():.1f}% of the total gain.")
    out.to_csv(OUT_DIR / "per_class_breakdown.csv", index=False)
    return out


def task2_leverage_check(df: pd.DataFrame):
    both_wrong = df[~df["yolo_correct"] & ~df["coord_correct"]]
    print("\n" + "=" * 70)
    print("TASK 2: leverage-point check on Section 26's phi coefficient")
    print("=" * 70)
    counts = both_wrong["true_class"].value_counts()
    print("Both-wrong cases by true FDI class (top 10):")
    for cid, n in counts.head(10).items():
        print(f"  {FDI_CODES[cid]:4s} {n:4d}")

    top_class = counts.idxmax()
    top_class_code = FDI_CODES[top_class]
    top_class_n = counts.max()
    print(f"\nMost frequent contributing class: {top_class_code} ({top_class_n} of "
          f"{len(both_wrong)} both-wrong instances, {100*top_class_n/len(both_wrong):.1f}%)")

    def compute_phi_and_ratio(data):
        yolo_wrong = (~data["yolo_correct"]).astype(int).to_numpy()
        coord_wrong = (~data["coord_correct"]).astype(int).to_numpy()
        phi = phi_coefficient(yolo_wrong, coord_wrong)
        p_yolo_wrong = yolo_wrong.mean()
        p_coord_wrong = coord_wrong.mean()
        expected = p_yolo_wrong * p_coord_wrong * len(data)
        observed = int(((yolo_wrong == 1) & (coord_wrong == 1)).sum())
        ratio = observed / expected if expected > 0 else float("nan")
        return phi, ratio, observed, expected, len(data)

    orig = compute_phi_and_ratio(df)
    excl = compute_phi_and_ratio(df[df["true_class"] != top_class])

    print(f"\n{'':30s} {'original':>15s} {'top-class excluded':>20s}")
    labels = ["phi", "obs/expected ratio", "observed both-wrong", "expected both-wrong", "n instances"]
    for label, o, e in zip(labels, orig, excl):
        if isinstance(o, float):
            print(f"{label:30s} {o:15.4f} {e:20.4f}")
        else:
            print(f"{label:30s} {o:15d} {e:20d}")

    return {
        "top_contributing_class": top_class_code,
        "top_class_n_both_wrong": int(top_class_n),
        "top_class_pct_of_both_wrong": 100 * top_class_n / len(both_wrong),
        "phi_original": orig[0], "ratio_original": orig[1],
        "phi_excl_top_class": excl[0], "ratio_excl_top_class": excl[1],
        "n_both_wrong_original": len(both_wrong),
    }


def bootstrap_ci(df: pd.DataFrame, metric_fn, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Resample unique image_ids with replacement (not individual instances),
    take all instances belonging to each sampled image, recompute metric_fn
    on the resulting instance set. Respects within-image correlation between
    multiple teeth annotated on the same radiograph."""
    rng = np.random.RandomState(seed)
    image_ids = df["image_id"].unique()
    n_images = len(image_ids)
    values = []
    # Pre-group for speed.
    groups = {iid: sub for iid, sub in df.groupby("image_id")}
    for _ in range(n_boot):
        sampled_ids = rng.choice(image_ids, size=n_images, replace=True)
        resampled = pd.concat([groups[iid] for iid in sampled_ids], ignore_index=True)
        values.append(metric_fn(resampled))
    values = np.array(values)
    point = metric_fn(df)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return point, lo, hi


def task3_bootstrap_section21_26(df: pd.DataFrame, phi_top_class_excluded: str = None):
    print("\n" + "=" * 70)
    print(f"TASK 3: bootstrap CIs (image-level resampling, n={N_BOOTSTRAP})")
    print("=" * 70)

    results = {}

    def acc_metric(col):
        return lambda d: d[col].mean()

    def quadrant_acc_metric(pred_col, true_col="true_class"):
        def f(d):
            valid = d[pred_col].notna()
            t = d.loc[valid, true_col].to_numpy(dtype=int)
            p = d.loc[valid, pred_col].to_numpy(dtype=int)
            return float(np.mean(quadrant_of(t) == quadrant_of(p)))
        return f

    def tooth_type_acc_metric(pred_col, true_col="true_class"):
        def f(d):
            valid = d[pred_col].notna()
            t = d.loc[valid, true_col].to_numpy(dtype=int)
            p = d.loc[valid, pred_col].to_numpy(dtype=int)
            return float(np.mean(tooth_type_of(t) == tooth_type_of(p)))
        return f

    def top1_metric(pred_col, true_col="true_class"):
        def f(d):
            valid = d[pred_col].notna()
            t = d.loc[valid, true_col].to_numpy(dtype=int)
            p = d.loc[valid, pred_col].to_numpy(dtype=int)
            return float(np.mean(t == p))
        return f

    def phi_metric(d):
        yolo_wrong = (~d["yolo_correct"]).astype(int).to_numpy()
        coord_wrong = (~d["coord_correct"]).astype(int).to_numpy()
        return phi_coefficient(yolo_wrong, coord_wrong)

    metrics = {
        "yolo_top1_acc": top1_metric("yolo_pred"),
        "yolo_quadrant_acc": quadrant_acc_metric("yolo_pred"),
        "yolo_tooth_type_acc": tooth_type_acc_metric("yolo_pred"),
        "coord_top1_acc": top1_metric("coord_pred"),
        "coord_quadrant_acc": quadrant_acc_metric("coord_pred"),
        "coord_tooth_type_acc": tooth_type_acc_metric("coord_pred"),
        "phi_coefficient": phi_metric,
    }

    for name, fn in metrics.items():
        point, lo, hi = bootstrap_ci(df, fn)
        results[name] = (point, lo, hi)
        print(f"  {name:24s} {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    pd.DataFrame([{"metric": k, "point": v[0], "ci_lo": v[1], "ci_hi": v[2]} for k, v in results.items()]
                 ).to_csv(OUT_DIR / "bootstrap_ci_section21_26.csv", index=False)
    return results


def main():
    df = pd.read_csv(JOINED_CSV)
    df["yolo_correct"] = df["yolo_correct"].astype(bool)
    df["coord_correct"] = df["coord_correct"].astype(bool)

    task1_per_class_breakdown(df)
    task2_result = task2_leverage_check(df)
    task3_bootstrap_section21_26(df)

    pd.DataFrame([task2_result]).to_csv(OUT_DIR / "leverage_check_section26.csv", index=False)
    print(f"\nSaved per_class_breakdown.csv, leverage_check_section26.csv, "
          f"bootstrap_ci_section21_26.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
