"""Per-tooth predictions for the MPS-trained Faster R-CNN seeds (5-9).

train_fasterrcnn.evaluate() only saves aggregate metrics, so the per-tooth
miss analysis cannot be run on seeds 5-9 from summary.csv. This reruns
inference on each best.pt with the same protocol as evaluate() (CPU,
val split, ty.match_boxes IoU threshold, greedy one-to-one matching) and
writes eval_results/fasterrcnn/seed{N}/per_tooth.csv: one row per labeled
tooth with its normalized box, class, and matched prediction (empty if
missed). Totals are checked against the existing summary.csv.

Usage: python per_tooth_predictions_mps.py [seed ...]   (default 5 6 7 8 9)
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_fasterrcnn as base
import train_yolo as ty


def run_seed(seed):
    out_dir = HERE / "prepared_coco" / f"seed{seed}"
    val_json = out_dir / "instances_val.json"
    base.SEED = seed
    base.VAL_JSON = val_json

    model = base.get_model()
    model.load_state_dict(torch.load(HERE / "runs" / f"fasterrcnn_seed{seed}split" / "best.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    val_ds = base.CocoTeethDataset(val_json, base.train_transforms())

    rows = []
    with torch.no_grad():
        for idx, iid in enumerate(val_ds.image_ids):
            info = val_ds.images[iid]
            w_img, h_img = info["width"], info["height"]
            anns = val_ds.anns_by_image[iid]
            gt_cls = [a["category_id"] - 1 for a in anns]
            gt_boxes = [[a["bbox"][0] / w_img, a["bbox"][1] / h_img,
                         (a["bbox"][0] + a["bbox"][2]) / w_img, (a["bbox"][1] + a["bbox"][3]) / h_img]
                        for a in anns]

            img, _ = val_ds[idx]
            out = model([img])[0]
            pred_cls = (out["labels"].numpy() - 1).astype(int)
            pred_boxes = out["boxes"].numpy() / np.array([w_img, h_img, w_img, h_img])

            match = {}  # gt index -> predicted class
            if len(gt_boxes) and len(pred_boxes):
                iou = ty.iou_xyxy(np.asarray(gt_boxes, float), np.asarray(pred_boxes, float))
                pairs = sorted(((iou[i, j], i, j) for i in range(len(gt_boxes))
                                for j in range(len(pred_boxes)) if iou[i, j] >= ty.MATCH_IOU_THRESHOLD),
                               key=lambda t: t[0], reverse=True)
                used_p = set()
                for _, i, j in pairs:
                    if i in match or j in used_p:
                        continue
                    match[i] = int(pred_cls[j])
                    used_p.add(j)

            for i, (c, b) in enumerate(zip(gt_cls, gt_boxes)):
                rows.append({"image_id": iid, "gt_idx": i, "true_class": c,
                             "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                             "pred_class": match.get(i, "")})

    dest = HERE / "eval_results" / "fasterrcnn" / f"seed{seed}" / "per_tooth.csv"
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summ = next(csv.DictReader(open(dest.parent / "summary.csv")))
    n_missed = sum(r["pred_class"] == "" for r in rows)
    n_matched = len(rows) - n_missed
    ok = (n_missed == int(summ["n_unmatched_gt_missed_detections"]) and n_matched == int(summ["n_test"]))
    print(f"seed {seed}: {len(rows)} teeth, {n_missed} missed, {n_matched} matched; "
          f"summary.csv says missed={summ['n_unmatched_gt_missed_detections']} n_test={summ['n_test']} "
          f"-> {'OK' if ok else 'MISMATCH'}", flush=True)
    return ok


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [5, 6, 7, 8, 9]
    results = [run_seed(s) for s in seeds]
    sys.exit(0 if all(results) else 1)
