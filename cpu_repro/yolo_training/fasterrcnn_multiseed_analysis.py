"""Faster R-CNN replication analysis (RESULTS.md Section 42/43's scoped
third-detector-architecture run). Mirrors rtdetr_multiseed_analysis.py's
structure exactly (itself mirroring multiseed_analysis.py for YOLOv8),
with the model swapped to Faster R-CNN, so the analysis methodology is
identical and any difference in the resulting gap is attributable to
architecture, not to a different statistical treatment.

build_coco_dataset.py and train_fasterrcnn.py are now seed-parameterized
the same way train_rtdetr_seed{1,2,3,4}.py override train_rtdetr.py, via
train_fasterrcnn_seed{1,2,3,4}.py thin wrappers - this script's own
paths_for_seed()/main(seeds=...) already generalized to any seed from
the start, so no changes were needed here for the 5-seed run.

Reports, per seed:
  - marginal top-1 accuracy (Faster R-CNN vs. coordinate-only)
  - the gap in percentage points
  - phi coefficient (error correlation, Section 26's convention)
  - paired image-level bootstrap 95% CI on the gap (same bootstrap_ci
    pattern as mitigation_analysis.py / rtdetr_multiseed_analysis.py)

Verification-before-use: read this seed-0 output with the same
scrutiny any first result gets (rtdetr_multiseed_analysis.py's own
docstring note) - there being no prior Faster R-CNN number to sanity-
check against. Explicitly checked here: raw prediction diversity (not
every box classified as one FDI number) and detection recall (not
near-zero detections) before trusting the gap/phi numbers at all - see
main()'s printed diagnostics, which are read BEFORE the gap/phi line by
design.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_yolo as ty  # noqa: E402 - reuse load_gt_boxes/iou_xyxy, not the YOLO class
import train_fasterrcnn as tf  # noqa: E402 - reuse get_model()

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

OUT_DIR = Path(__file__).resolve().parent / "eval_results" / "fasterrcnn_multiseed"
N_BOOTSTRAP = 2000


def paths_for_seed(seed: int):
    prepared_dir = Path(__file__).resolve().parent / "prepared" / (f"seed{seed}" if seed != 0 else "")
    weights = (Path(__file__).resolve().parent / "runs" / f"fasterrcnn_seed{seed}split" / "best.pt")
    return prepared_dir, weights


def build_coord_predictions(seed: int) -> pd.DataFrame:
    """Identical to multiseed_analysis.py/rtdetr_multiseed_analysis.py's
    build_coord_predictions() - the coordinate baseline doesn't depend on
    which detector it's being compared against."""
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


def run_fasterrcnn_predictions(seed: int) -> pd.DataFrame:
    """Mirrors run_rtdetr_predictions()'s structure exactly - same
    ty.load_gt_boxes()/ty.iou_xyxy()-based greedy matching, same
    (label_file, line_idx) join key, no "image_id" column here for the
    identical reason documented in rtdetr_multiseed_analysis.py (pandas
    silently renaming both to image_id_x/image_id_y instead of erroring
    was a real bug there - not reintroducing it here)."""
    import torch

    prepared_dir, weights_path = paths_for_seed(seed)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"seed {seed}: {weights_path} not found - train_fasterrcnn.py's "
            f"train() has not been run / its output not downloaded yet."
        )
    val_paths = (prepared_dir / "val.txt").read_text().splitlines()
    if not val_paths:
        raise FileNotFoundError(f"seed {seed}: {prepared_dir / 'val.txt'} is empty or missing.")

    device = torch.device("cpu")
    model = tf.get_model()
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    transform = tf.train_transforms()

    rows = []
    n_raw_detections = 0
    raw_pred_class_counts = {}

    from PIL import Image

    with torch.no_grad():
        for image_path in val_paths:
            p = Path(image_path)
            label_path = p.parents[1] / "labels" / f"{p.stem}.txt"
            gt_classes, gt_boxes = ty.load_gt_boxes(label_path)
            if not gt_classes:
                continue

            image = Image.open(image_path).convert("RGB")
            img_w, img_h = image.size
            image_tensor = transform(image)
            output = model([image_tensor.to(device)])[0]

            pred_classes = (output["labels"].cpu().numpy() - 1).astype(int).tolist()  # undo bg shift
            pred_boxes_abs = output["boxes"].cpu().numpy()
            pred_boxes = (pred_boxes_abs / np.array([img_w, img_h, img_w, img_h])).tolist()

            n_raw_detections += len(pred_classes)
            for c in pred_classes:
                raw_pred_class_counts[c] = raw_pred_class_counts.get(c, 0) + 1

            gt_arr = np.asarray(gt_boxes, dtype=float)
            pred_arr = np.asarray(pred_boxes, dtype=float)
            matched_gt_idx, matched_pred_idx, gt_to_pred = set(), set(), {}
            if len(gt_arr) and len(pred_arr):
                iou = ty.iou_xyxy(gt_arr, pred_arr)
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
                    "fasterrcnn_pred": pred_cls,
                })

    print(f"  raw diagnostics: {n_raw_detections} total raw detections across "
          f"{len(val_paths)} val images, {len(raw_pred_class_counts)} distinct "
          f"FDI classes predicted (out of 32) - degenerate would be near-0 "
          f"detections or 1 distinct class dominating almost all of them.")
    if raw_pred_class_counts:
        top_class, top_count = max(raw_pred_class_counts.items(), key=lambda kv: kv[1])
        print(f"  most-predicted class: FDI {ty.FDI_CODES[top_class] if hasattr(ty, 'FDI_CODES') else top_class} "
              f"with {top_count}/{n_raw_detections} ({100*top_count/max(n_raw_detections,1):.1f}%) of raw detections.")

    return pd.DataFrame(rows)


def phi_coefficient(a, b) -> float:
    return float(np.corrcoef(a.astype(float), b.astype(float))[0, 1])


def bootstrap_ci(image_ids, per_image_data, metric_fn, n_boot=N_BOOTSTRAP, seed=0):
    """Same image-level-resampling bootstrap as mitigation_analysis.py's /
    rtdetr_multiseed_analysis.py's paired-CI treatment."""
    rng = np.random.RandomState(seed)
    n_images = len(image_ids)
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(image_ids, size=n_images, replace=True)
        rows = pd.concat([per_image_data[iid] for iid in sampled], ignore_index=True)
        values.append(metric_fn(rows))
    point = metric_fn(pd.concat([per_image_data[iid] for iid in image_ids], ignore_index=True))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return point, lo, hi


def build_merged(seed: int, use_cache: bool = True) -> pd.DataFrame:
    cache_path = OUT_DIR / f"joined_seed{seed}.csv"
    if use_cache and cache_path.exists():
        merged = pd.read_csv(cache_path)
        merged["coord_correct"] = merged["coord_correct"].astype(bool)
        merged["fasterrcnn_correct"] = merged["fasterrcnn_correct"].astype(bool)
        return merged

    coord_df = build_coord_predictions(seed)
    fasterrcnn_df = run_fasterrcnn_predictions(seed)
    merged = coord_df.merge(fasterrcnn_df, on=["label_file", "line_idx"], how="inner")

    merged["fasterrcnn_correct"] = (
        merged["fasterrcnn_pred"].notna() & (merged["fasterrcnn_pred"].astype("Int64") == merged["true_class"])
    )
    merged["coord_correct"] = merged["coord_pred"] == merged["true_class"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path, index=False)
    return merged


def analyze_seed(seed: int) -> dict:
    merged = build_merged(seed)

    n = len(merged)
    fasterrcnn_top1 = float(merged["fasterrcnn_correct"].mean())
    coord_top1 = float(merged["coord_correct"].mean())
    fasterrcnn_quad = float(np.mean(
        quadrant_of(merged.loc[merged["fasterrcnn_pred"].notna(), "true_class"].to_numpy())
        == quadrant_of(merged.loc[merged["fasterrcnn_pred"].notna(), "fasterrcnn_pred"].astype(int).to_numpy())
    ))
    coord_quad = float(np.mean(
        quadrant_of(merged["true_class"].to_numpy()) == quadrant_of(merged["coord_pred"].to_numpy())
    ))

    phi = phi_coefficient(
        (~merged["fasterrcnn_correct"].to_numpy(dtype=bool)).astype(int),
        (~merged["coord_correct"].to_numpy(dtype=bool)).astype(int),
    )

    def gap_fn(d):
        return 100 * (float(d["fasterrcnn_correct"].mean()) - float(d["coord_correct"].mean()))

    image_ids = merged["image_id"].dropna().unique()
    per_image = {iid: sub for iid, sub in merged.groupby("image_id")}
    gap_point, gap_lo, gap_hi = bootstrap_ci(image_ids, per_image, gap_fn, seed=seed)

    return {
        "seed": seed, "n": n,
        "fasterrcnn_top1": fasterrcnn_top1, "coord_top1": coord_top1,
        "gap_pp": 100 * (fasterrcnn_top1 - coord_top1),
        "gap_bootstrap_pp": gap_point, "gap_ci_lo": gap_lo, "gap_ci_hi": gap_hi,
        "fasterrcnn_quadrant": fasterrcnn_quad, "coord_quadrant": coord_quad,
        "phi": phi,
    }


def per_class_breakdown(seed: int, merged: pd.DataFrame) -> pd.DataFrame:
    """Section 27/33-style per-FDI-class accuracy breakdown, ported from
    multiseed_analysis.py's per_class_breakdown() with yolo_* renamed to
    fasterrcnn_* - identical logic, applied to this seed's joined table."""
    rows = []
    for class_id in range(32):
        sub = merged[merged["true_class"] == class_id]
        if len(sub) == 0:
            continue
        coord_acc = sub["coord_correct"].mean()
        fasterrcnn_acc = sub["fasterrcnn_correct"].mean()
        rows.append({
            "seed": seed, "fdi_code": FDI_CODES[class_id], "class_id": class_id,
            "n_instances": len(sub), "coord_acc": coord_acc, "fasterrcnn_acc": fasterrcnn_acc,
            "delta_fasterrcnn_minus_coord": fasterrcnn_acc - coord_acc,
        })
    out = pd.DataFrame(rows).sort_values("delta_fasterrcnn_minus_coord").reset_index(drop=True)

    total_gain = (merged.groupby("true_class")["fasterrcnn_correct"].sum()
                  - merged.groupby("true_class")["coord_correct"].sum()).sum()
    net_gain = (merged.groupby("true_class")["fasterrcnn_correct"].sum()
                - merged.groupby("true_class")["coord_correct"].sum())
    out["net_correct_gain"] = out["class_id"].map(net_gain)
    out["pct_of_total_gap"] = 100 * out["net_correct_gain"] / total_gain if total_gain else float("nan")
    return out


def error_taxonomy_breakdown(seed: int, merged: pd.DataFrame) -> dict:
    """Section 22/33-style error-type comparison, ported from
    multiseed_analysis.py's error_taxonomy_breakdown() with yolo_*
    renamed to fasterrcnn_* - each model's own wrong predictions run
    through is_mirror_quadrant_error/is_neighbor_error."""
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
    fasterrcnn_bd = breakdown("true_class", "fasterrcnn_pred", "fasterrcnn_correct")
    return {"seed": seed,
            **{f"coord_{k}": v for k, v in coord_bd.items()},
            **{f"fasterrcnn_{k}": v for k, v in fasterrcnn_bd.items()}}


def run_breakdowns(seeds=(0, 1, 2, 3, 4)):
    """Section 32/33's deferred 'free CPU add-ons', ported for Faster
    R-CNN - per-seed per-class breakdown (Section 27-style) and
    error-taxonomy comparison (Section 22-style), across every seed that
    has a trained checkpoint. Reuses the cached joined tables from
    analyze_seed()/build_merged() rather than re-running inference."""
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
        n_reversals = int((cb["delta_fasterrcnn_minus_coord"] < 0).sum())
        top5_share = cb.sort_values("pct_of_total_gap", ascending=False).head(5)["pct_of_total_gap"].sum()
        print(f"  reversals={n_reversals}  top5_share_of_gain={top5_share:.1f}%  "
              f"coord_neighbor_frac={tb['coord_neighbor_frac']:.3f}  "
              f"fasterrcnn_neighbor_frac={tb['fasterrcnn_neighbor_frac']:.3f}")

    if class_rows:
        pd.concat(class_rows, ignore_index=True).to_csv(OUT_DIR / "per_class_breakdown_multiseed.csv", index=False)
        pd.DataFrame(tax_rows).to_csv(OUT_DIR / "error_taxonomy_multiseed.csv", index=False)
        print(f"\nSaved per_class_breakdown_multiseed.csv, error_taxonomy_multiseed.csv to {OUT_DIR}")
    else:
        print("\nNo seeds had a trained checkpoint available - nothing to save.")


def main(seeds=(0,)):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        _, weights_path = paths_for_seed(seed)
        if not weights_path.exists():
            print(f"seed {seed}: SKIPPED - {weights_path} not found yet.")
            continue
        print(f"seed {seed}: running coord refit + Faster R-CNN CPU inference...")
        r = analyze_seed(seed)
        rows.append(r)
        print(f"  n={r['n']}  fasterrcnn_top1={r['fasterrcnn_top1']:.4f}  coord_top1={r['coord_top1']:.4f}  "
              f"gap={r['gap_pp']:+.1f}pp  [{r['gap_ci_lo']:+.1f}, {r['gap_ci_hi']:+.1f}]  "
              f"fasterrcnn_quad={r['fasterrcnn_quadrant']:.4f}  phi={r['phi']:.4f}")

    if not rows:
        print("\nNo seeds had a trained checkpoint available - nothing to save.")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "fasterrcnn_multiseed_summary.csv", index=False)
    print(f"\nSaved fasterrcnn_multiseed_summary.csv ({len(rows)} of {len(seeds)} seeds) to {OUT_DIR}")


if __name__ == "__main__":
    main()
