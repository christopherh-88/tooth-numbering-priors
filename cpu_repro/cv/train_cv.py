"""Grouped 5-fold CV on uncropped UFBA-425 (cpu_repro/cv/PHASE2_CV_RULES.md).

    python cpu_repro/cv/train_cv.py --detector yolov8x [--folds 0 1 2 3 4] [--cv-seed 0]
    python cpu_repro/cv/train_cv.py --detector fasterrcnn --smoke   # CPU, 1 epoch, 4 images

Per fold: train on the fold's training X-rays, pick the epoch with the best
validation fitness (0.1 x mAP50 + 0.9 x mAP50-95, Ultralytics' rule, also
used for Faster R-CNN), then save every test-fold detection with confidence
>= 0.05. Scoring (confidence >= 0.5, IoU >= 0.5 matching, gaps, CIs) is done
afterwards by score_cv.py, from these raw detections, so the
IoU/confidence sweeps need no retraining.

Outputs under OUT_ROOT/<detector>_cvseed<s>/fold<f>/:
  test_detections.csv   image_id, class_id, conf, x1, y1, x2, y2 (normalized)
  epoch_log.csv         per-epoch validation fitness (Faster R-CNN; the
                        Ultralytics run dir has its own results.csv)
  best.pt               selected weights
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_uncropped_cv as prep  # noqa: E402

OUT_ROOT = HERE / "runs"
EPOCHS = 30
IMGSZ = 640
RAW_CONF = 0.05
NMS_IOU = 0.7

# Same settings as yolo_training/train_yolo.py and train_rtdetr.py.
YOLO_ARGS = dict(epochs=EPOCHS, batch=10, imgsz=IMGSZ, dropout=0.6, close_mosaic=0, cos_lr=True,
                 warmup_epochs=10, lrf=0.005, translate=0.1, scale=0.5, fliplr=0.0, val=True)
RTDETR_ARGS = dict(epochs=EPOCHS, batch=10, imgsz=IMGSZ, fliplr=0.0, val=True)

# Faster R-CNN: torchvision reference recipe as in yolo_training/train_fasterrcnn.py,
# except the learning rate is scaled to a batch of 2 (reference: 0.02 for 16).
FRCNN_BATCH = 2
FRCNN_LR = 0.02 * FRCNN_BATCH / 16
FRCNN_MILESTONES = [18, 25]


def fold_lists(work: Path, fold: int):
    return {k: [Path(p) for p in (work / f"fold{fold}_{k}.txt").read_text().split()]
            for k in ("train", "val", "test")}


def prepare(work: Path, cv_seed: int, augment_gaps: bool = False):
    boxes = pd.read_csv(HERE / "boxes.csv")
    folds = pd.read_csv(prep.folds_path(cv_seed))  # folds.csv for seed 0, folds_seed<s>.csv otherwise
    prep.write_yolo(work, boxes, folds)
    if augment_gaps:
        write_gap_augmented(work, boxes)


def write_gap_augmented(work: Path, boxes):
    """GAP_AUGMENT_RULES.md: each X-ray gets, with probability 0.5 (seed 0), an altered
    copy with one tooth erased, and in half of those its distal neighbor moved fully into
    the space. Training lists point to the copy; validation and test lists are unchanged."""
    import cv2
    import closure_intervention as ci
    from gap_check import NEIGHBORS
    from gap_intervention import dilate, load_mask
    mask_path = {p.name.replace(".ome.tiff", ""): p for p in (prep.SRC / "labels").glob("*/*.ome.tiff")}
    img_dir, lab_dir = work / "aug" / "images", work / "aug" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    altered, log = set(), []
    for image_id, g in boxes.astype({"fdi": str}).groupby("image_id"):
        present = set(g["fdi"])
        if rng.random() >= 0.5:
            continue
        ok = sorted(f for f in present if f[1] != "8" and all(n in present for n in NEIGHBORS[f])
                    and len(NEIGHBORS[f]) == 2)
        close = rng.random() < 0.5
        if not ok:
            continue
        T = str(rng.choice(ok))
        img = cv2.imread(str(prep.SRC / "panoramic_x_rays" / f"{image_id}.jpg"))
        t_region = dilate(load_mask(mask_path[f"{image_id}_{T}"], img.shape[:2])).astype(bool)
        g = g[g["fdi"] != T].copy()
        s = 0
        if close:
            N, M, direction = ci.neighbors_of(T)
            n_mask = load_mask(mask_path[f"{image_id}_{N}"], img.shape[:2])
            full = ci.touch_shift(n_mask, load_mask(mask_path[f"{image_id}_{M}"], img.shape[:2]), direction)
            if full is not None:
                s = full * direction
                img = ci.move_tooth(img, t_region, dilate(n_mask).astype(bool), s)
                g.loc[g["fdi"] == N, "x_center"] += s / img.shape[1]
        if s == 0:
            img = cv2.inpaint(img, t_region.astype(np.uint8), 5, cv2.INPAINT_TELEA)
        cv2.imwrite(str(img_dir / f"{image_id}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        lines = [f"{r.class_id} {r.x_center:.6f} {r.y_center:.6f} {r.width:.6f} {r.height:.6f}"
                 for r in g.itertuples()]
        (lab_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n")
        altered.add(image_id)
        log.append(dict(image_id=image_id, erased=T, closed=bool(s), shift_px=abs(s)))
    pd.DataFrame(log).to_csv(work / "aug" / "augment_log.csv", index=False)
    for f in range(prep.N_FOLDS):
        p = work / f"fold{f}_train.txt"
        paths = [str(img_dir / Path(x).name) if Path(x).stem in altered else x for x in p.read_text().split()]
        p.write_text("\n".join(paths) + "\n")
    print(f"gap augmentation: {len(altered)} altered X-rays, {sum(r['closed'] for r in log)} with a closed gap",
          flush=True)


def label_path(img: Path) -> Path:
    return img.parent.parent / "labels" / f"{img.stem}.txt"


def read_labels(img: Path):
    a = np.loadtxt(label_path(img), ndmin=2)
    cls, xc, yc, w, h = a.T
    return cls.astype(int), np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], 1)


def save_detections(rows, out: Path):
    pd.DataFrame(rows, columns=["image_id", "class_id", "conf", "x1", "y1", "x2", "y2"]).to_csv(
        out / "test_detections.csv", index=False, float_format="%.6f")


# ---------------------------------------------------------------- Ultralytics
def run_ultralytics(detector, fold, work, out, device, smoke, train_seed=0):
    from ultralytics import RTDETR, YOLO
    lists = fold_lists(work, fold)
    if smoke:
        for k in lists:
            lists[k] = lists[k][:4]
    for k, paths in lists.items():
        (out / f"{k}.txt").write_text("\n".join(map(str, paths)) + "\n")
    yaml = out / "data.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(prep.FDI_CODES))
    yaml.write_text(f"train: {out / 'train.txt'}\nval: {out / 'val.txt'}\nnc: 32\nnames:\n{names}\n")

    cls, base, args = ((YOLO, "yolov8x.pt", YOLO_ARGS) if detector == "yolov8x"
                       else (RTDETR, "rtdetr-l.pt", RTDETR_ARGS))
    args = dict(args, seed=train_seed)
    if smoke:
        args.update(epochs=1, batch=2, imgsz=320)
        base = "yolov8n.pt" if detector == "yolov8x" else base
    run_dir = out / "ultralytics"
    last = run_dir / "weights" / "last.pt"
    if last.exists():
        try:
            cls(str(last)).train(resume=True)
        except AssertionError as e:  # Ultralytics: "... is finished, nothing to resume"
            if "nothing to resume" not in str(e):
                raise
            print(f"fold already trained: {e}")
    else:
        cls(base).train(data=str(yaml), device=device, project=str(out), name="ultralytics",
                        exist_ok=True, **args)
    best = run_dir / "weights" / "best.pt"

    model = cls(str(best))
    rows = []
    for img in lists["test"]:
        r = model.predict(source=str(img), conf=RAW_CONF, iou=NMS_IOU, imgsz=args["imgsz"],
                          device=device, verbose=False)[0]
        for c, p, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                           r.boxes.xyxyn.cpu().numpy()):
            rows.append([img.stem, int(c), float(p), *map(float, b)])
    save_detections(rows, out)


# ---------------------------------------------------------------- Faster R-CNN
def run_fasterrcnn(fold, work, out, device, smoke, train_seed=0):
    import torch
    if train_seed:  # the original runs set no seed; only seed when asked (PHASE3_BATCH2_RULES.md, 2026-10-06)
        import random
        random.seed(train_seed)
        np.random.seed(train_seed)
        torch.manual_seed(train_seed + fold)
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                              fasterrcnn_resnet50_fpn_v2)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.transforms import v2 as T

    lists = fold_lists(work, fold)
    if smoke:
        for k in lists:
            lists[k] = lists[k][:4]
    to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])  # no flips

    class Teeth(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            w, h = img.size
            cls, boxes = read_labels(self.paths[i])
            boxes = boxes * np.array([w, h, w, h])
            return to_tensor(img), {"boxes": torch.as_tensor(boxes, dtype=torch.float32),
                                    "labels": torch.as_tensor(cls + 1, dtype=torch.int64)}

    def collate(b):
        return tuple(zip(*b))

    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                       box_nms_thresh=NMS_IOU)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, len(prep.FDI_CODES) + 1)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=FRCNN_LR, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=FRCNN_MILESTONES, gamma=0.1)
    epochs = 1 if smoke else EPOCHS
    train_dl = DataLoader(Teeth(lists["train"]), batch_size=FRCNN_BATCH, shuffle=True,
                          collate_fn=collate, num_workers=0 if device == "cpu" else 2)

    state, log = out / "last.pt", []
    start, best_fit = 0, -1.0
    if state.exists():
        ck = torch.load(state, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best_fit, log = ck["epoch"] + 1, ck["best_fit"], ck["log"]
        print(f"resuming fold {fold} at epoch {start}")

    def predict(paths, score_thresh):
        model.eval()
        model.roi_heads.score_thresh = score_thresh
        res = []
        with torch.no_grad():
            for p in paths:
                img = Image.open(p).convert("RGB")
                w, h = img.size
                o = model([to_tensor(img).to(device)])[0]
                b = o["boxes"].cpu().numpy() / np.array([w, h, w, h])
                res.append((p, (o["labels"].cpu().numpy() - 1).astype(int),
                            o["scores"].cpu().numpy(), b))
        return res

    for epoch in range(start, epochs):
        model.train()
        t0, total = time.time(), 0.0
        for images, targets in train_dl:
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        m50, m5095 = val_map(predict(lists["val"], 0.001))
        fit = 0.1 * m50 + 0.9 * m5095
        log.append(dict(epoch=epoch + 1, train_loss=total / len(train_dl), map50=m50,
                        map50_95=m5095, fitness=fit, seconds=time.time() - t0))
        print(json.dumps(log[-1]), flush=True)
        if fit > best_fit:
            best_fit = fit
            torch.save(model.state_dict(), out / "best.pt")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=epoch, best_fit=best_fit, log=log), state)
    pd.DataFrame(log).to_csv(out / "epoch_log.csv", index=False)

    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    rows = []
    for p, cls, conf, b in predict(lists["test"], RAW_CONF):
        rows += [[p.stem, int(c), float(s), *map(float, bb)] for c, s, bb in zip(cls, conf, b)]
    save_detections(rows, out)


def iou_matrix(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def val_map(preds):
    """mAP50 and mAP50-95 exactly as Ultralytics' DetectionValidator: its
    greedy match_predictions (copied below) and ap_per_class."""
    from ultralytics.utils.metrics import ap_per_class
    iouv = np.linspace(0.5, 0.95, 10)
    tp, conf, pcls, tcls = [], [], [], []
    for path, cls, scores, boxes in preds:
        gt_cls, gt_boxes = read_labels(path)
        tcls.append(gt_cls)
        correct = np.zeros((len(cls), 10), bool)
        if len(cls) and len(gt_cls):
            iou = iou_matrix(gt_boxes, boxes) * (gt_cls[:, None] == cls[None, :])
            for i, t in enumerate(iouv):
                m = np.array(np.nonzero(iou >= t)).T
                if m.shape[0]:
                    if m.shape[0] > 1:
                        m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                        m = m[np.unique(m[:, 1], return_index=True)[1]]
                        m = m[np.unique(m[:, 0], return_index=True)[1]]
                    correct[m[:, 1].astype(int), i] = True
        tp.append(correct)
        conf.append(scores)
        pcls.append(cls)
    tp, conf, pcls, tcls = (np.concatenate(x) for x in (tp, conf, pcls, tcls))
    if not len(conf):
        return 0.0, 0.0
    ap = ap_per_class(tp, conf, pcls, tcls)[5]
    return float(ap[:, 0].mean()), float(ap.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=["yolov8x", "rtdetr_l", "fasterrcnn"])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(prep.N_FOLDS)))
    ap.add_argument("--cv-seed", type=int, default=0)
    ap.add_argument("--work", type=Path, default=HERE / "prepared")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--device", default="0")
    ap.add_argument("--smoke", action="store_true", help="CPU, 1 epoch, 4 images per split")
    ap.add_argument("--augment-gaps", action="store_true", help="GAP_AUGMENT_RULES.md training copies")
    ap.add_argument("--train-seed", type=int, default=0, help="Ultralytics training seed (default 0, as before)")
    a = ap.parse_args()
    if a.smoke:
        a.device = "cpu"
    prepare(a.work, a.cv_seed, a.augment_gaps)
    tag = ("_aug" if a.augment_gaps else "") + (f"_tseed{a.train_seed}" if a.train_seed else "")
    for fold in a.folds:
        out = a.out_root / f"{a.detector}_cvseed{a.cv_seed}{tag}{'_smoke' if a.smoke else ''}" / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        if (out / "test_detections.csv").exists():
            print(f"fold {fold}: test_detections.csv exists, skipping")
            continue
        t0 = time.time()
        if a.detector == "fasterrcnn":
            dev = "cpu" if a.device == "cpu" else f"cuda:{a.device}"
            run_fasterrcnn(fold, a.work, out, dev, a.smoke, a.train_seed)
        else:
            run_ultralytics(a.detector, fold, a.work, out, a.device, a.smoke, a.train_seed)
        print(f"fold {fold} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
