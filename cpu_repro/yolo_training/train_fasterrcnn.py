"""Faster R-CNN training run (RESULTS.md Section 42's scoping, third
detector architecture for Claim B). Written for a Kaggle GPU session,
same convention as train_yolo.py/train_rtdetr.py - do NOT run the real
30-epoch train() on CPU. smoke_test_forward() (one batch, forward pass
only) IS meant to run locally on CPU - that's the Section 42 checkpoint
4 check, and needs no GPU.

Reads cpu_repro/yolo_training/prepared_coco/instances_{train,val}.json
(build_coco_dataset.py's output - run that first). Does NOT re-derive
the train/val split - those JSONs already encode the same
image_split_seed0.csv split every other detector in this project uses.

Model: fasterrcnn_resnet50_fpn_v2, COCO-pretrained backbone, box
predictor head replaced for this project's 33-class problem (32 FDI
classes + background at label 0 - see build_coco_dataset.py's docstring
for why the shift is +1, not a free 33rd class).

Fair-comparison protocol (Section 42, not identical hyperparameters to
YOLOv8x/RT-DETR-l - matching the disclosed "use the framework's own
default recipe" stance already applied to both of those):
  - Optimizer/LR: SGD lr=0.02 momentum=0.9 weight_decay=1e-4,
    MultiStepLR gamma=0.1 - torchvision's own detection reference
    training script defaults (references/detection/train.py), not
    hand-tuned for this dataset.
  - LR milestones scaled proportionally to this project's 30-epoch
    budget: torchvision's reference recipe decays at epochs 16/22 of a
    26-epoch run (61.5%/84.6% of the way through); scaled to EPOCHS=30
    that is epochs 18/25. This preserves the reference recipe's *shape*
    at the epoch count this project standardizes on (Sections 21/38),
    rather than either using the reference's own 26-epoch total (breaking
    cross-architecture epoch comparability) or its raw milestone numbers
    unscaled (which would decay far too early relative to a 30-epoch
    budget - a genuine hand-tune, not a default-recipe transplant).
  - Batch size: 2 - torchvision's own reference default for this model,
    used unmodified (not YOLOv8's BATCH=10, which is a YOLO-specific
    default and not this framework's).
  - Eval-time operating point: box_score_thresh=0.5, box_nms_thresh=0.7,
    matching EVAL_CONF/EVAL_NMS_IOU in train_yolo.py/train_rtdetr.py, so
    the three architectures are compared at the same confidence/NMS
    point rather than each one's own untuned default.
  - Epoch count: EPOCHS=30, matching Sections 21/38. Two-stage detectors
    commonly need more epochs than single-stage/DETR models to converge
    (two forward passes per image: region-proposal + per-region
    classification) - if the real run's training loss is still dropping
    steeply at epoch 30, that must be reported as a caveat on the
    cross-architecture comparison, not silently fixed by adding epochs
    after the fact.

Transform pipeline - deliberately excludes horizontal AND vertical flip
(Section 42's MIRROR_MAP-risk analysis): FDI tooth numbers encode
left-right quadrant in their class id (11-18 vs 21-28, 31-38 vs 41-48).
Mirroring the image without remapping the class id teaches a physically
wrong tooth-to-position association - the exact bug fixed once already
in notebooks/yolov8+unet/yolov8+unet_training.ipynb (see HANDOFF.md) and
avoided in train_yolo.py/train_rtdetr.py via fliplr=0.0. Torchvision has
no single equivalent flag - augmentation here is whatever transforms are
explicitly composed - so RandomHorizontalFlip/RandomVerticalFlip are
simply never added to TRAIN_TRANSFORMS below, not toggled off via a
parameter.

Other-transforms audit (per Section 42's "confirm and flag any other
transform that touches box coordinates" instruction): torchvision's own
reference training script's default preset for Faster R-CNN
("hflip", references/detection/presets.py::DetectionPresetTrain) contains
exactly RandomHorizontalFlip + a tensor/dtype conversion - no crop, zoom,
or photometric transform. The "ssd"/"ssdlite"/"multiscale" preset
variants (built for SSD-family models, not part of Faster R-CNN's own
reference command, and NOT used here) add RandomIoUCrop/RandomZoomOut/
RandomPhotometricDistort/RandomShortestSize; noted for completeness that
none of those mirror the image, so even if a future edit adopted one, it
would reposition/rescale a box's coordinates together with the image
content without swapping which physical side of the image it's on - no
class remap would be needed the way flips need one. Only flip-type
transforms carry the MIRROR_MAP risk.
"""

import json
import sys
import time
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coord_baseline"))
from build_coord_baseline import FDI_CODES  # noqa: E402

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
# Seed-parameterized like train_rtdetr.py (SEED=0 here, train_fasterrcnn_seed{1,2,3,4}.py
# override SEED/COCO_DIR/TRAIN_JSON/VAL_JSON/RUN_NAME/EVAL_RESULTS_DIR before
# calling train()/evaluate() - same thin-wrapper pattern as train_rtdetr_seed*.py).
# Seed 0's COCO_DIR is left unsuffixed (matches build_coco_dataset.py's own
# seed-0 OUT_DIR) so the already-trained seed-0 run's paths are untouched.
SEED = 0
COCO_DIR = Path(__file__).resolve().parent / "prepared_coco" / (f"seed{SEED}" if SEED != 0 else "")
TRAIN_JSON = COCO_DIR / "instances_train.json"
VAL_JSON = COCO_DIR / "instances_val.json"

NUM_CLASSES = len(FDI_CODES) + 1  # +1 for background at label 0 (build_coco_dataset.py's shift)

PROJECT_DIR = Path(__file__).resolve().parent / "runs"
RUN_NAME = f"fasterrcnn_seed{SEED}split"
EVAL_RESULTS_DIR = Path(__file__).resolve().parent / "eval_results" / "fasterrcnn" / f"seed{SEED}"

EPOCHS = 30
BATCH = 2                  # torchvision reference default, not YOLO's BATCH=10
LR = 0.02                  # torchvision reference default
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_MILESTONES = [18, 25]   # torchvision's 16/22-of-26 schedule, scaled to EPOCHS=30 (see module docstring)
LR_GAMMA = 0.1
DEVICE = 0                  # GPU device index for Kaggle; smoke_test_forward() forces "cpu" regardless
TRAIN_DEVICE = "cuda"        # "cuda" (Kaggle, uses DEVICE as the index) or "mps" (local Apple Silicon)

EVAL_CONF = 0.5              # matches EVAL_CONF in train_yolo.py/train_rtdetr.py
EVAL_NMS_IOU = 0.7           # matches EVAL_NMS_IOU in train_yolo.py/train_rtdetr.py
# --------------------------------------------------------------------------


def train_transforms():
    """No RandomHorizontalFlip/RandomVerticalFlip - see module docstring's
    MIRROR_MAP-risk section for why."""
    return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])


class CocoTeethDataset(Dataset):
    """Minimal COCO-format detection dataset - reads build_coco_dataset.py's
    instances_{train,val}.json directly (not torchvision.datasets.CocoDetection,
    to avoid a pycocotools dependency for what's a single small JSON file)."""

    def __init__(self, json_path: Path, transforms):
        coco = json.loads(json_path.read_text())
        self.images = {im["id"]: im for im in coco["images"]}
        self.image_ids = list(self.images.keys())
        self.anns_by_image = {iid: [] for iid in self.image_ids}
        for ann in coco["annotations"]:
            self.anns_by_image[ann["image_id"]].append(ann)
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        image = Image.open(img_info["file_name"]).convert("RGB")

        anns = self.anns_by_image[image_id]
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])  # COCO xywh -> xyxy for torchvision detection API
            labels.append(ann["category_id"])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
        }
        image = self.transforms(image)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes: int = NUM_CLASSES) -> torchvision.models.detection.FasterRCNN:
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        box_score_thresh=EVAL_CONF,
        box_nms_thresh=EVAL_NMS_IOU,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_dataloaders():
    if not TRAIN_JSON.exists() or not VAL_JSON.exists():
        raise FileNotFoundError(
            f"{TRAIN_JSON} / {VAL_JSON} not found - run build_coco_dataset.py first."
        )
    train_ds = CocoTeethDataset(TRAIN_JSON, train_transforms())
    val_ds = CocoTeethDataset(VAL_JSON, train_transforms())
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    return train_loader, val_loader


def smoke_test_forward():
    """Section 42 checkpoint 4: run one real batch through the model's
    forward pass (train mode, computing a loss dict) with no shape/dtype/
    label-range error. Runs on CPU - no GPU needed for a single batch."""
    train_loader, _ = get_dataloaders()
    images, targets = next(iter(train_loader))

    device = torch.device("cpu")
    model = get_model()
    model.to(device)
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    assert all(t["labels"].max().item() <= NUM_CLASSES - 1 for t in targets if len(t["labels"])), \
        "a label exceeds NUM_CLASSES-1 - background-shift invariant broken"
    assert all(t["labels"].min().item() >= 1 for t in targets if len(t["labels"])), \
        "a label is 0 (background) or negative - background-shift invariant broken"

    model.train()
    loss_dict = model(images, targets)
    total_loss = sum(loss_dict.values())
    print(f"Forward pass (train mode) succeeded on a real batch of {len(images)}.")
    for k, v in loss_dict.items():
        print(f"    {k}: {v.item():.4f}")
    print(f"  total_loss: {total_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        outputs = model(images)
    print(f"Forward pass (eval mode) succeeded. Output keys per image: "
          f"{list(outputs[0].keys())}, boxes shape: {outputs[0]['boxes'].shape}")

    return loss_dict, outputs


def train():
    """NOT for CPU - Kaggle GPU only, same convention as train_yolo.py.
    Resumable across a killed Kaggle session (Section 39's 12-hour-limit
    concern): last.pt is a full training-state checkpoint (model +
    optimizer + scheduler + epoch), not just weights, so a rerun of this
    function picks up mid-schedule rather than restarting the LR
    schedule from scratch."""
    train_loader, val_loader = get_dataloaders()
    device = torch.device(f"cuda:{DEVICE}") if TRAIN_DEVICE == "cuda" else torch.device(TRAIN_DEVICE)
    model = get_model().to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA)

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = PROJECT_DIR / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = run_dir / "last.pt"

    start_epoch = 0
    if last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from {last_ckpt} at epoch {start_epoch}/{EPOCHS}.")

    n_batches = len(train_loader)
    print(f"Starting training: {n_batches} batches/epoch (BATCH={BATCH}), {EPOCHS} epochs, "
          f"starting at epoch {start_epoch}.")

    # Per-step progress logging - added after a real seed-0 run sat silent
    # for 65+ minutes with zero output (train() previously printed only
    # once per COMPLETED epoch), and `kaggle kernels logs -f` confirmed
    # live-streaming showed no new lines either - genuinely no way to
    # distinguish "slow" from "stuck" without this. Prints every
    # PROGRESS_EVERY steps with a running avg step time, so a future run's
    # first real number arrives within roughly a minute, not an hour.
    PROGRESS_EVERY = 20
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        for step, (images, targets) in enumerate(train_loader):
            step_start = time.time()
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if step == 0 or (step + 1) % PROGRESS_EVERY == 0:
                elapsed = time.time() - epoch_start
                avg_step_time = elapsed / (step + 1)
                print(f"  epoch {epoch+1}/{EPOCHS}  step {step+1}/{n_batches}  "
                      f"loss={loss.item():.4f}  avg_step_time={avg_step_time:.2f}s  "
                      f"last_step_time={time.time() - step_start:.2f}s")

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        epoch_time = time.time() - epoch_start
        print(f"epoch {epoch+1}/{EPOCHS}  avg_loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}  "
              f"epoch_time={epoch_time:.1f}s")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }, last_ckpt)

    torch.save(model.state_dict(), run_dir / "best.pt")  # weights-only, for evaluate()/downstream analysis
    return model


def evaluate(weights_path):
    """Identical protocol to train_yolo.evaluate()/train_rtdetr.evaluate() -
    same coord_evaluate() call, same ty.match_boxes() IoU-matching
    convention (normalized xyxy) - only the model class and the box-
    coordinate conversion (absolute pixel -> normalized xyxy) differ, so
    results are directly comparable to Sections 21/33/40."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_yolo as ty  # noqa: E402 - reuse load_gt_boxes/iou_xyxy/match_boxes

    from sklearn.metrics import confusion_matrix
    import numpy as np
    import pandas as pd

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = get_model()
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    val_ds = CocoTeethDataset(VAL_JSON, train_transforms())
    train_ds = CocoTeethDataset(TRAIN_JSON, train_transforms())

    train_y = []
    for iid in train_ds.image_ids:
        for ann in train_ds.anns_by_image[iid]:
            train_y.append(ann["category_id"] - 1)  # undo background shift
    train_y = np.array(train_y)

    all_true, all_pred = [], []
    total_unmatched_gt, total_unmatched_pred, total_gt = 0, 0, 0

    with torch.no_grad():
        for idx, iid in enumerate(val_ds.image_ids):
            img_info = val_ds.images[iid]
            img_w, img_h = img_info["width"], img_info["height"]
            anns = val_ds.anns_by_image[iid]

            gt_classes = [a["category_id"] - 1 for a in anns]
            gt_boxes = []
            for a in anns:
                x, y, w, h = a["bbox"]
                gt_boxes.append([x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h])
            total_gt += len(gt_classes)

            image_tensor, _ = val_ds[idx]
            output = model([image_tensor.to(device)])[0]

            pred_classes = (output["labels"].cpu().numpy() - 1).astype(int).tolist()  # undo shift
            pred_boxes_abs = output["boxes"].cpu().numpy()
            pred_boxes = (pred_boxes_abs / np.array([img_w, img_h, img_w, img_h])).tolist()

            matched_true, matched_pred, n_unmatched_gt, n_unmatched_pred = ty.match_boxes(
                gt_boxes, gt_classes, pred_boxes, pred_classes
            )
            all_true.extend(matched_true)
            all_pred.extend(matched_pred)
            total_unmatched_gt += n_unmatched_gt
            total_unmatched_pred += n_unmatched_pred

    y_true = np.array(all_true, dtype=int)
    y_pred = np.array(all_pred, dtype=int)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coord_baseline"))
    from build_coord_baseline import evaluate as coord_evaluate

    metrics = coord_evaluate(y_true, y_pred, train_y)
    metrics["seed"] = SEED
    metrics["classifier"] = "fasterrcnn"
    metrics["detection_recall"] = (
        (total_gt - total_unmatched_gt) / total_gt if total_gt > 0 else float("nan")
    )
    metrics["n_unmatched_gt_missed_detections"] = total_unmatched_gt
    metrics["n_unmatched_pred_spurious_detections"] = total_unmatched_pred

    pd.DataFrame([metrics]).to_csv(EVAL_RESULTS_DIR / "summary.csv", index=False)

    if len(y_true) == 0:
        print("WARNING: zero matched detections across the whole val split - skipping confusion matrix.")
    else:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(32)))
        pd.DataFrame(cm, index=FDI_CODES, columns=FDI_CODES).to_csv(
            EVAL_RESULTS_DIR / "confusion_matrix_fasterrcnn_seed0.csv"
        )

    print("\n" + "=" * 70)
    print("Faster R-CNN evaluation (matched detections vs. ground truth, seed 0 split)")
    print("=" * 70)
    for key in ["top1_acc", "quadrant_acc", "tooth_type_acc", "majority_baseline_acc",
                "mirror_quadrant_error_frac", "neighbor_error_frac", "detection_recall"]:
        print(f"  {key:32s} {metrics[key]}")
    print(f"  missed detections (unmatched GT)  {total_unmatched_gt} / {total_gt}")
    print(f"  spurious detections (unmatched pred) {total_unmatched_pred}")
    print(f"\nSaved summary.csv to {EVAL_RESULTS_DIR}")
    return metrics


def main():
    train()
    best_weights = PROJECT_DIR / RUN_NAME / "best.pt"
    evaluate(best_weights)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                         help="run the real 30-epoch training + evaluate() - Kaggle GPU only. "
                              "Default (no flag) runs smoke_test_forward() instead, which is "
                              "safe on CPU and does not touch runs/ or eval_results/.")
    args = parser.parse_args()
    if args.train:
        main()
    else:
        smoke_test_forward()
