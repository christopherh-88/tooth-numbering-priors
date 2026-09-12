"""Mitigation experiment write-up (Section 21's PIVOT-result follow-up):
paired comparison of the jittered run (Section 21, `train_yolo.py`,
translate=0.1/scale=0.5) against the near-zero-jitter run
(`train_yolo_zerojitter.py`, translate=0.0/scale=0.0), per the analysis
method pre-committed in RESULTS.md Section 30 before these results were
read.

Reuses case_study_yolo_vs_coord.py's load_gt_boxes()/iou_xyxy() and a
parameterized version of its run_yolo_predictions() (only change: takes a
weights path instead of a hardcoded module constant) to run CPU inference
with both checkpoints over the identical seed-0 val split, then joins on
(label_file, line_idx) - the same exact-identity join key used by
Section 23/26's YOLO-vs-coordinate comparisons.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from case_study_yolo_vs_coord import PREPARED_DIR, EVAL_CONF, EVAL_NMS_IOU, MATCH_IOU_THRESHOLD
from train_yolo import load_gt_boxes, iou_xyxy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coord_baseline"))
from build_coord_baseline import quadrant_of

OUT_DIR = Path(__file__).resolve().parent / "eval_results"
JITTERED_WEIGHTS = Path(__file__).resolve().parent / "runs" / "yolov8_seed0split" / "weights" / "best.pt"
ZEROJITTER_WEIGHTS = Path(__file__).resolve().parent / "runs" / "yolov8_seed0split_zerojitter" / "weights" / "best.pt"
N_BOOTSTRAP = 2000


def run_predictions(weights_path) -> pd.DataFrame:
    from ultralytics import YOLO

    val_paths = (PREPARED_DIR / "val.txt").read_text().splitlines()
    model = YOLO(str(weights_path))
    rows = []

    for image_path in val_paths:
        p = Path(image_path)
        label_path = p.parents[1] / "labels" / f"{p.stem}.txt"
        gt_classes, gt_boxes = load_gt_boxes(label_path)
        if not gt_classes:
            continue

        result = model.predict(source=image_path, conf=EVAL_CONF, iou=EVAL_NMS_IOU,
                                device="cpu", verbose=False)[0]
        pred_classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
        pred_boxes = result.boxes.xyxyn.cpu().numpy().tolist()

        gt_arr = np.asarray(gt_boxes, dtype=float)
        pred_arr = np.asarray(pred_boxes, dtype=float)
        matched_gt_idx, matched_pred_idx = set(), set()
        gt_to_pred = {}
        if len(gt_arr) and len(pred_arr):
            iou = iou_xyxy(gt_arr, pred_arr)
            pairs = sorted(
                ((iou[i, j], i, j) for i in range(len(gt_arr)) for j in range(len(pred_arr))
                 if iou[i, j] >= MATCH_IOU_THRESHOLD),
                key=lambda t: t[0], reverse=True,
            )
            for _, i, j in pairs:
                if i in matched_gt_idx or j in matched_pred_idx:
                    continue
                matched_gt_idx.add(i)
                matched_pred_idx.add(j)
                gt_to_pred[i] = j

        image_id = p.stem
        for line_idx, true_cls in enumerate(gt_classes):
            pred_cls = pred_classes[gt_to_pred[line_idx]] if line_idx in gt_to_pred else None
            rows.append({
                "image_id": image_id,
                "label_file": label_path.name,
                "line_idx": line_idx,
                "true_class": true_cls,
                "pred": pred_cls,
            })

    return pd.DataFrame(rows)


def bootstrap_ci(image_ids, per_image_data, metric_fn, n_boot=N_BOOTSTRAP, seed=0):
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


def main():
    print("Running jittered checkpoint (Section 21, translate=0.1/scale=0.5) on val split...")
    jit = run_predictions(JITTERED_WEIGHTS)
    print("Running near-zero-jitter checkpoint (translate=0.0/scale=0.0) on val split...")
    zero = run_predictions(ZEROJITTER_WEIGHTS)

    merged = jit.merge(zero, on=["label_file", "line_idx", "true_class"],
                        suffixes=("_jit", "_zero"), how="outer")
    n_total = len(merged)
    print(f"Joined on (label_file, line_idx): {n_total} instances "
          f"(jit rows={len(jit)}, zero rows={len(zero)}).")

    merged["image_id"] = merged["image_id_jit"].combine_first(merged["image_id_zero"])
    merged["jit_correct"] = merged["pred_jit"].notna() & (merged["pred_jit"].astype("Int64") == merged["true_class"])
    merged["zero_correct"] = merged["pred_zero"].notna() & (merged["pred_zero"].astype("Int64") == merged["true_class"])

    both_correct = merged[merged["jit_correct"] & merged["zero_correct"]]
    both_wrong = merged[~merged["jit_correct"] & ~merged["zero_correct"]]
    jit_only = merged[merged["jit_correct"] & ~merged["zero_correct"]]
    zero_only = merged[~merged["jit_correct"] & merged["zero_correct"]]

    print("\nPaired 2x2 breakdown (jittered vs. near-zero-jitter, same instances):")
    for name, subset in [("both_correct", both_correct), ("both_wrong", both_wrong),
                          ("jittered_only_correct", jit_only), ("zerojitter_only_correct", zero_only)]:
        print(f"  {name:26s} {len(subset):5d}  ({100 * len(subset) / n_total:.2f}%)")

    jit_acc = merged["jit_correct"].mean()
    zero_acc = merged["zero_correct"].mean()
    print(f"\nMarginal top-1 accuracy: jittered={jit_acc:.4f}  zerojitter={zero_acc:.4f}  "
          f"delta={zero_acc - jit_acc:+.4f}")

    def jit_top1(d):
        return float(d["jit_correct"].mean())

    def zero_top1(d):
        return float(d["zero_correct"].mean())

    def paired_delta(d):
        return float(d["zero_correct"].mean() - d["jit_correct"].mean())

    def jit_quad(d):
        p = d[d["pred_jit"].notna()]
        return float(np.mean(quadrant_of(p["true_class"].to_numpy()) == quadrant_of(p["pred_jit"].astype(int).to_numpy())))

    def zero_quad(d):
        p = d[d["pred_zero"].notna()]
        return float(np.mean(quadrant_of(p["true_class"].to_numpy()) == quadrant_of(p["pred_zero"].astype(int).to_numpy())))

    image_ids = merged["image_id"].dropna().unique()
    per_image = {iid: sub for iid, sub in merged.groupby("image_id")}

    print("\nBootstrap 95% CIs (image-level resampling, n=2000):")
    results = {}
    for name, fn in [("jittered_top1", jit_top1), ("zerojitter_top1", zero_top1),
                      ("paired_delta_top1", paired_delta),
                      ("jittered_quadrant", jit_quad), ("zerojitter_quadrant", zero_quad)]:
        point, lo, hi = bootstrap_ci(image_ids, per_image, fn)
        results[name] = (point, lo, hi)
        print(f"  {name:22s} {point:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

    merged.to_csv(OUT_DIR / "mitigation_paired_joined.csv", index=False)
    pd.DataFrame([
        {"variant": "both_correct", "n": len(both_correct)},
        {"variant": "both_wrong", "n": len(both_wrong)},
        {"variant": "jittered_only_correct", "n": len(jit_only)},
        {"variant": "zerojitter_only_correct", "n": len(zero_only)},
    ]).to_csv(OUT_DIR / "mitigation_paired_2x2.csv", index=False)
    pd.DataFrame([{"metric": k, "point": v[0], "ci_lo": v[1], "ci_hi": v[2]} for k, v in results.items()]
                 ).to_csv(OUT_DIR / "mitigation_bootstrap_ci.csv", index=False)
    print(f"\nSaved mitigation_paired_joined.csv, mitigation_paired_2x2.csv, "
          f"mitigation_bootstrap_ci.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
