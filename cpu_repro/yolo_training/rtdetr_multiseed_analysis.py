"""RT-DETR 5-seed replication analysis (RESULTS.md Section 38/39's
scoped second-detector-architecture run). Mirrors multiseed_analysis.py's
structure (Sections 26/33's YOLO-vs-coord comparison) exactly, with the
model swapped to RT-DETR, so the analysis methodology is identical and
any difference in the resulting gap is attributable to architecture,
not to a different statistical treatment.

Reports, per seed and pooled across 5 seeds:
  - marginal top-1 accuracy (RT-DETR vs. coordinate-only)
  - the gap in percentage points
  - phi coefficient (error correlation, Section 26's convention)
  - paired image-level bootstrap 95% CI on the gap (same bootstrap_ci
    pattern as mitigation_analysis.py's paired-CI treatment for YOLOv8)
  - 5-seed mean +/- 95% CI (mean_ci95, Section 33's convention)

Verification-before-use: none needed here in the way multiseed_analysis.py
verifies against a pre-existing YOLO number, because there is no prior
RT-DETR result to check against - this script's own seed-0 output is the
first RT-DETR result in this project. Read its output with the same
scrutiny any first result gets, not with an existing-number sanity check.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_yolo as ty  # noqa: E402 - reuse load_gt_boxes/iou_xyxy/match_boxes, not the YOLO class

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coord_baseline"))
from build_coord_baseline import (  # noqa: E402
    FEATURE_COLS,
    TEST_FRACTION,
    grouped_split,
    load_instances,
    mean_ci95,
    quadrant_of,
)

OUT_DIR = Path(__file__).resolve().parent / "eval_results" / "rtdetr_multiseed"
N_BOOTSTRAP = 2000


def paths_for_seed(seed: int):
    prepared_dir = Path(__file__).resolve().parent / "prepared" / (f"seed{seed}" if seed != 0 else "")
    weights = (Path(__file__).resolve().parent / "runs" / f"rtdetr_seed{seed}split"
               / "weights" / "best.pt")
    return prepared_dir, weights


def build_coord_predictions(seed: int) -> pd.DataFrame:
    """Identical to multiseed_analysis.py's build_coord_predictions() -
    the coordinate baseline doesn't depend on which detector it's being
    compared against."""
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


def run_rtdetr_predictions(seed: int) -> pd.DataFrame:
    from ultralytics import RTDETR

    prepared_dir, weights_path = paths_for_seed(seed)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"seed {seed}: {weights_path} not found - train_rtdetr_seed{seed}.py "
            f"(or train_rtdetr.py for seed 0) has not been run / its output not "
            f"downloaded yet."
        )
    val_paths = (prepared_dir / "val.txt").read_text().splitlines()
    if not val_paths:
        raise FileNotFoundError(f"seed {seed}: {prepared_dir / 'val.txt'} is empty or missing.")

    model = RTDETR(str(weights_path))
    rows = []
    for image_path in val_paths:
        p = Path(image_path)
        label_path = p.parents[1] / "labels" / f"{p.stem}.txt"
        gt_classes, gt_boxes = ty.load_gt_boxes(label_path)
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
                # No "image_id" column here - coord_df already carries one, and
                # the merge below joins on (label_file, line_idx) which
                # uniquely identifies the row anyway. Including a second
                # "image_id" here caused a real bug: pandas silently renamed
                # both to image_id_x/image_id_y instead of erroring, and
                # analyze_seed()'s merged["image_id"] lookup raised KeyError.
                "label_file": label_path.name,
                "line_idx": line_idx,
                "rtdetr_pred": pred_cls,
            })

    return pd.DataFrame(rows)


def phi_coefficient(a, b) -> float:
    return float(np.corrcoef(a.astype(float), b.astype(float))[0, 1])


def bootstrap_ci(image_ids, per_image_data, metric_fn, n_boot=N_BOOTSTRAP, seed=0):
    """Same image-level-resampling bootstrap as mitigation_analysis.py's
    paired-CI treatment for the YOLOv8 mitigation result (Section 31)."""
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
        merged["rtdetr_correct"] = merged["rtdetr_correct"].astype(bool)
        return merged

    coord_df = build_coord_predictions(seed)
    rtdetr_df = run_rtdetr_predictions(seed)
    merged = coord_df.merge(rtdetr_df, on=["label_file", "line_idx"], how="inner")

    merged["rtdetr_correct"] = (
        merged["rtdetr_pred"].notna() & (merged["rtdetr_pred"].astype("Int64") == merged["true_class"])
    )
    merged["coord_correct"] = merged["coord_pred"] == merged["true_class"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path, index=False)
    return merged


def analyze_seed(seed: int) -> dict:
    merged = build_merged(seed)

    n = len(merged)
    rtdetr_top1 = float(merged["rtdetr_correct"].mean())
    coord_top1 = float(merged["coord_correct"].mean())
    rtdetr_quad = float(np.mean(
        quadrant_of(merged.loc[merged["rtdetr_pred"].notna(), "true_class"].to_numpy())
        == quadrant_of(merged.loc[merged["rtdetr_pred"].notna(), "rtdetr_pred"].astype(int).to_numpy())
    ))
    coord_quad = float(np.mean(
        quadrant_of(merged["true_class"].to_numpy()) == quadrant_of(merged["coord_pred"].to_numpy())
    ))

    phi = phi_coefficient(
        (~merged["rtdetr_correct"].to_numpy(dtype=bool)).astype(int),
        (~merged["coord_correct"].to_numpy(dtype=bool)).astype(int),
    )

    def gap_fn(d):
        return 100 * (float(d["rtdetr_correct"].mean()) - float(d["coord_correct"].mean()))

    image_ids = merged["image_id"].dropna().unique()
    per_image = {iid: sub for iid, sub in merged.groupby("image_id")}
    gap_point, gap_lo, gap_hi = bootstrap_ci(image_ids, per_image, gap_fn, seed=seed)

    return {
        "seed": seed, "n": n,
        "rtdetr_top1": rtdetr_top1, "coord_top1": coord_top1,
        "gap_pp": 100 * (rtdetr_top1 - coord_top1),
        "gap_bootstrap_pp": gap_point, "gap_ci_lo": gap_lo, "gap_ci_hi": gap_hi,
        "rtdetr_quadrant": rtdetr_quad, "coord_quadrant": coord_quad,
        "phi": phi,
    }


def main(seeds=(0, 1, 2, 3, 4)):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        _, weights_path = paths_for_seed(seed)
        if not weights_path.exists():
            print(f"seed {seed}: SKIPPED - {weights_path} not found yet.")
            continue
        print(f"seed {seed}: running coord refit + RT-DETR CPU inference...")
        r = analyze_seed(seed)
        rows.append(r)
        print(f"  n={r['n']}  rtdetr_top1={r['rtdetr_top1']:.4f}  coord_top1={r['coord_top1']:.4f}  "
              f"gap={r['gap_pp']:+.1f}pp  [{r['gap_ci_lo']:+.1f}, {r['gap_ci_hi']:+.1f}]  "
              f"rtdetr_quad={r['rtdetr_quadrant']:.4f}  phi={r['phi']:.4f}")

    if not rows:
        print("\nNo seeds had a trained checkpoint available - nothing to save.")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "rtdetr_multiseed_summary.csv", index=False)
    print(f"\nSaved rtdetr_multiseed_summary.csv ({len(rows)} of {len(seeds)} seeds) to {OUT_DIR}")

    if len(rows) > 1:
        print(f"\n5-seed mean +/- 95% CI (mean_ci95, Section 33's convention):")
        for col in ["gap_pp", "phi", "rtdetr_top1", "coord_top1", "rtdetr_quadrant"]:
            mean, ci = mean_ci95(out_df[col].to_numpy())
            print(f"  {col:16s} {mean:.4f} +/- {ci:.4f}  [{mean - ci:.4f}, {mean + ci:.4f}]")


if __name__ == "__main__":
    main()
