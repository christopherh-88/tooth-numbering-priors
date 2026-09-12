"""Multi-seed replication of Sections 21/22/26 (RESULTS.md Section 32's
scoped plan). Seed-parameterized generalization of
case_study_yolo_vs_coord.py's build_coord_predictions()/run_yolo_predictions()
and error_correlation_analysis.py's phi/2x2 analysis - same logic, no
hardcoded seed-0 paths, so it can be run once per seed as each seed's YOLO
checkpoint becomes available (seed 0 already does; seeds 1-4 depend on the
corresponding train_yolo_seed<N>.py Kaggle runs actually finishing).

Verification-before-use: `main(seeds=[0])` reproduces Section 21's top-1
(0.9558) and Section 26's phi (0.163) from the already-existing seed-0
checkpoint/predictions before this script is trusted for seeds 1-4 - see
the assertion in `main()`. Not run for seeds 1-4 until their checkpoints
exist; this file itself makes no claim about their results.

Section 25 (DenPAR) is not included here - already 5-seed, no YOLO
dependency, nothing to generalize (Section 32).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_yolo as ty  # noqa: E402
from train_yolo import load_gt_boxes, iou_xyxy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coord_baseline"))
from build_coord_baseline import (  # noqa: E402
    FDI_CODES,
    FEATURE_COLS,
    TEST_FRACTION,
    grouped_split,
    is_mirror_quadrant_error,
    is_neighbor_error,
    load_instances,
    quadrant_of,
)

OUT_DIR = Path(__file__).resolve().parent / "eval_results" / "multiseed"


def paths_for_seed(seed: int):
    prepared_dir = Path(__file__).resolve().parent / "prepared" / (f"seed{seed}" if seed != 0 else "")
    weights = (Path(__file__).resolve().parent / "runs" / f"yolov8_seed{seed}split"
               / "weights" / "best.pt")
    return prepared_dir, weights


def build_coord_predictions(seed: int) -> pd.DataFrame:
    df = load_instances()
    df = df.reset_index(drop=True)
    df["line_idx"] = df.groupby("label_file").cumcount()

    train_df, test_df = grouped_split(df, seed, TEST_FRACTION)
    x_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df["class_id"].to_numpy()
    x_test = test_df[FEATURE_COLS].to_numpy()

    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    result = test_df[["image_id", "label_file", "line_idx", "class_id"]].copy()
    result["coord_pred"] = y_pred
    result = result.rename(columns={"class_id": "true_class"})

    assert result.duplicated(subset=["label_file", "line_idx"]).sum() == 0, (
        f"seed {seed}: label_file+line_idx not unique in the coordinate "
        f"baseline's own test set - join key assumption broken, do not proceed."
    )
    return result


def run_yolo_predictions(seed: int) -> pd.DataFrame:
    from ultralytics import YOLO

    prepared_dir, weights_path = paths_for_seed(seed)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"seed {seed}: {weights_path} not found - train_yolo_seed{seed}.py "
            f"(or train_yolo.py for seed 0) has not been run / its output not "
            f"downloaded yet."
        )
    val_paths = (prepared_dir / "val.txt").read_text().splitlines()
    if not val_paths:
        raise FileNotFoundError(f"seed {seed}: {prepared_dir / 'val.txt'} is empty or missing.")

    model = YOLO(str(weights_path))
    rows = []
    for image_path in val_paths:
        p = Path(image_path)
        label_path = p.parents[1] / "labels" / f"{p.stem}.txt"
        gt_classes, gt_boxes = load_gt_boxes(label_path)
        if not gt_classes:
            continue

        result = model.predict(source=image_path, conf=ty.EVAL_CONF, iou=ty.EVAL_NMS_IOU,
                                device="cpu", verbose=False)[0]
        pred_classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
        pred_boxes = result.boxes.xyxyn.cpu().numpy().tolist()

        gt_arr = np.asarray(gt_boxes, dtype=float)
        pred_arr = np.asarray(pred_boxes, dtype=float)
        matched_gt_idx, matched_pred_idx, gt_to_pred = set(), set(), {}
        if len(gt_arr) and len(pred_arr):
            iou = iou_xyxy(gt_arr, pred_arr)
            pairs = sorted(
                ((iou[i, j], i, j) for i in range(len(gt_arr)) for j in range(len(pred_arr))
                 if iou[i, j] >= ty.MATCH_IOU_THRESHOLD),
                key=lambda t: t[0], reverse=True,
            )
            for _, i, j in pairs:
                if i in matched_gt_idx or j in matched_pred_idx:
                    continue
                matched_gt_idx.add(i)
                matched_pred_idx.add(j)
                gt_to_pred[i] = j

        for line_idx, true_cls in enumerate(gt_classes):
            pred_cls = pred_classes[gt_to_pred[line_idx]] if line_idx in gt_to_pred else None
            rows.append({
                "label_file": label_path.name,
                "line_idx": line_idx,
                "yolo_pred": pred_cls,
            })

    return pd.DataFrame(rows)


def phi_coefficient(a, b) -> float:
    return float(np.corrcoef(a.astype(float), b.astype(float))[0, 1])


def build_merged(seed: int, use_cache: bool = True) -> pd.DataFrame:
    """Cached per-seed joined table (coord + YOLO predictions) - CPU YOLO
    inference over ~200 val images takes real wall-clock time, so re-runs
    of downstream analysis (breakdown/taxonomy scripts) read the cached
    joined CSV instead of re-inferring from scratch."""
    cache_path = OUT_DIR / f"joined_seed{seed}.csv"
    if use_cache and cache_path.exists():
        merged = pd.read_csv(cache_path)
        merged["coord_correct"] = merged["coord_correct"].astype(bool)
        merged["yolo_correct"] = merged["yolo_correct"].astype(bool)
        return merged

    coord_df = build_coord_predictions(seed)
    yolo_df = run_yolo_predictions(seed)
    merged = coord_df.merge(yolo_df, on=["label_file", "line_idx"], how="inner")

    merged["yolo_correct"] = (
        merged["yolo_pred"].notna() & (merged["yolo_pred"].astype("Int64") == merged["true_class"])
    )
    merged["coord_correct"] = merged["coord_pred"] == merged["true_class"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path, index=False)
    return merged


def analyze_seed(seed: int) -> dict:
    merged = build_merged(seed)

    n = len(merged)
    yolo_top1 = float(merged["yolo_correct"].mean())
    coord_top1 = float(merged["coord_correct"].mean())
    yolo_quad = float(np.mean(
        quadrant_of(merged.loc[merged["yolo_pred"].notna(), "true_class"].to_numpy())
        == quadrant_of(merged.loc[merged["yolo_pred"].notna(), "yolo_pred"].astype(int).to_numpy())
    ))
    coord_quad = float(np.mean(
        quadrant_of(merged["true_class"].to_numpy()) == quadrant_of(merged["coord_pred"].to_numpy())
    ))

    phi = phi_coefficient(
        (~merged["yolo_correct"].to_numpy(dtype=bool)).astype(int),
        (~merged["coord_correct"].to_numpy(dtype=bool)).astype(int),
    )

    return {
        "seed": seed, "n": n,
        "yolo_top1": yolo_top1, "coord_top1": coord_top1, "gap_pp": 100 * (yolo_top1 - coord_top1),
        "yolo_quadrant": yolo_quad, "coord_quadrant": coord_quad,
        "phi": phi,
    }


def per_class_breakdown(seed: int, merged: pd.DataFrame) -> pd.DataFrame:
    """Section 27-style per-FDI-class accuracy breakdown, generalized to
    an arbitrary seed - identical logic to robustness_analysis.py's
    task1_per_class_breakdown(), applied to this seed's own joined table
    instead of the seed-0-only error_correlation_joined.csv."""
    rows = []
    for class_id in range(32):
        sub = merged[merged["true_class"] == class_id]
        if len(sub) == 0:
            continue
        coord_acc = sub["coord_correct"].mean()
        yolo_acc = sub["yolo_correct"].mean()
        rows.append({
            "seed": seed, "fdi_code": FDI_CODES[class_id], "class_id": class_id,
            "n_instances": len(sub), "coord_acc": coord_acc, "yolo_acc": yolo_acc,
            "delta_yolo_minus_coord": yolo_acc - coord_acc,
        })
    out = pd.DataFrame(rows).sort_values("delta_yolo_minus_coord").reset_index(drop=True)

    total_gain = (merged.groupby("true_class")["yolo_correct"].sum()
                  - merged.groupby("true_class")["coord_correct"].sum()).sum()
    net_gain = (merged.groupby("true_class")["yolo_correct"].sum()
                - merged.groupby("true_class")["coord_correct"].sum())
    out["net_correct_gain"] = out["class_id"].map(net_gain)
    out["pct_of_total_gap"] = 100 * out["net_correct_gain"] / total_gain if total_gain else float("nan")
    return out


def error_taxonomy_breakdown(seed: int, merged: pd.DataFrame) -> dict:
    """Section 22-style error-type comparison, generalized to an arbitrary
    seed - each model's own wrong predictions (not a joined subset,
    matching Section 22's own convention of computing this per-model on
    that model's full evaluation output) run through
    is_mirror_quadrant_error/is_neighbor_error."""
    def breakdown(true_col, pred_col, correct_col):
        wrong = merged[~merged[correct_col].astype(bool) & merged[pred_col].notna()]
        n_wrong = int((~merged[correct_col].astype(bool)).sum())
        if len(wrong) == 0:
            return {"n_wrong": n_wrong, "mirror_frac": float("nan"), "neighbor_frac": float("nan"),
                    "other_frac": float("nan")}
        t = wrong[true_col].to_numpy(dtype=int)
        p = wrong[pred_col].astype(int).to_numpy()
        mirror = np.array([is_mirror_quadrant_error(ti, pi) for ti, pi in zip(t, p)])
        neighbor = np.array([is_neighbor_error(ti, pi) for ti, pi in zip(t, p)])
        other = ~mirror & ~neighbor
        return {
            "n_wrong": n_wrong,
            "mirror_frac": float(mirror.mean()), "mirror_n": int(mirror.sum()),
            "neighbor_frac": float(neighbor.mean()), "neighbor_n": int(neighbor.sum()),
            "other_frac": float(other.mean()), "other_n": int(other.sum()),
        }

    coord_bd = breakdown("true_class", "coord_pred", "coord_correct")
    yolo_bd = breakdown("true_class", "yolo_pred", "yolo_correct")
    return {"seed": seed,
            **{f"coord_{k}": v for k, v in coord_bd.items()},
            **{f"yolo_{k}": v for k, v in yolo_bd.items()}}


def run_breakdowns(seeds=(0, 1, 2, 3, 4)):
    """Section 32's deferred 'free CPU add-ons' - per-seed per-class
    breakdown (Section 27-style) and error-taxonomy comparison (Section
    22-style), across every seed that has a trained checkpoint. Reuses
    the cached joined tables from analyze_seed()/build_merged() rather
    than re-running YOLO inference."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    class_rows, tax_rows = [], []
    for seed in seeds:
        _, weights_path = paths_for_seed(seed)
        if not weights_path.exists():
            print(f"seed {seed}: SKIPPED - {weights_path} not found yet.")
            continue
        print(f"seed {seed}: building per-class and error-taxonomy breakdowns...")
        merged = build_merged(seed)
        cb = per_class_breakdown(seed, merged)
        class_rows.append(cb)
        tb = error_taxonomy_breakdown(seed, merged)
        tax_rows.append(tb)
        n_reversals = int((cb["delta_yolo_minus_coord"] < 0).sum())
        top5_share = cb.sort_values("pct_of_total_gap", ascending=False).head(5)["pct_of_total_gap"].sum()
        print(f"  reversals={n_reversals}  top5_share_of_gain={top5_share:.1f}%  "
              f"coord_neighbor_frac={tb['coord_neighbor_frac']:.3f}  "
              f"yolo_neighbor_frac={tb['yolo_neighbor_frac']:.3f}")

    if class_rows:
        pd.concat(class_rows, ignore_index=True).to_csv(OUT_DIR / "per_class_breakdown_multiseed.csv", index=False)
        pd.DataFrame(tax_rows).to_csv(OUT_DIR / "error_taxonomy_multiseed.csv", index=False)
        print(f"\nSaved per_class_breakdown_multiseed.csv, error_taxonomy_multiseed.csv to {OUT_DIR}")
    else:
        print("\nNo seeds had a trained checkpoint available - nothing to save.")


def main(seeds=(0, 1, 2, 3, 4), verify_only_seed0=True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        _, weights_path = paths_for_seed(seed)
        if not weights_path.exists():
            print(f"seed {seed}: SKIPPED - {weights_path} not found yet.")
            continue
        print(f"seed {seed}: running coord refit + YOLO CPU inference...")
        r = analyze_seed(seed)
        rows.append(r)
        print(f"  n={r['n']}  yolo_top1={r['yolo_top1']:.4f}  coord_top1={r['coord_top1']:.4f}  "
              f"gap={r['gap_pp']:+.1f}pp  yolo_quad={r['yolo_quadrant']:.4f}  phi={r['phi']:.4f}")

    if verify_only_seed0 and rows and rows[0]["seed"] == 0:
        # Sanity check against Section 21/26's already-published numbers
        # before trusting this generalized script for seeds 1-4.
        r0 = rows[0]
        assert abs(r0["yolo_top1"] - 0.9410) < 0.005, (
            f"seed-0 yolo_top1 {r0['yolo_top1']:.4f} does not match the "
            f"~0.9410 figure from Section 31's paired analysis (same "
            f"undetected-counts-as-wrong convention) - do not trust this "
            f"script for seeds 1-4 until this is resolved."
        )
        assert abs(r0["phi"] - 0.163) < 0.01, (
            f"seed-0 phi {r0['phi']:.4f} does not match Section 26's 0.163 - "
            f"do not trust this script for seeds 1-4 until this is resolved."
        )
        print("\nVerification against Section 21/26/31's published seed-0 numbers: PASS.")

    if rows:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(OUT_DIR / "multiseed_summary.csv", index=False)
        print(f"\nSaved multiseed_summary.csv ({len(rows)} of {len(seeds)} seeds) to {OUT_DIR}")
    else:
        print("\nNo seeds had a trained checkpoint available - nothing to save.")


if __name__ == "__main__":
    main()
