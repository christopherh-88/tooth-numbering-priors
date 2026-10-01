# Phase 2 CV: decision rules (frozen 2026-10-01)

Written 2026-10-01, before any CV training run and before the position-only
model is fit on these folds, and approved by the user the same day before
any kernel was pushed (code at commit 6ea25a5). This file is frozen:
results are graded against it, and any later change is added as a dated
note at the end, not an edit above.

## Why

Two weaknesses in the current headline (24 to 25 pp), from RESULTS.md
Section 59 and the roadmap:
1. About 70% of each seed's test X-rays are Roboflow copies randomly cropped
   0 to 20%, which weakens the position cue. On uncropped originals the gap
   was 18.5 to 19.9 pp (seeds 0-4), but from only about 25 X-rays per seed.
2. YOLOv8x and RT-DETR-l kept the epoch that scored best on the test images.

## Design

- **Data:** all 425 UFBA-425 X-rays, uncropped (`Dataset/bb_u_net_dataset`,
  512x512, byte-identical to the FigShare release). Boxes are the pixel
  extent of each tooth's FDI mask (`cpu_repro/cv/boxes.csv`, 11,602 teeth).
  No Roboflow copies anywhere, in training or test.
- **Folds:** 5 folds, balanced by UFBA category, CV seed 0
  (`cpu_repro/cv/folds.csv`). Each X-ray is tested exactly once. Inside
  each fold, 15% of the training X-rays (52) form a validation set used only
  to pick the epoch; about 288 X-rays train.
- **Epoch selection:** the epoch with the best validation fitness,
  0.1 x mAP50 + 0.9 x mAP50-95 (Ultralytics' own rule). Faster R-CNN uses
  the same formula, computed on the same validation set. The test fold is
  never used for selection. (Faster R-CNN's validation mAP is computed by
  `train_cv.val_map`, a copy of Ultralytics' matching plus its
  `ap_per_class`. Checked 2026-10-01 on a trained YOLOv8x checkpoint and
  fold 0's 52 validation X-rays: mAP50 0.976 vs. Ultralytics' 0.981,
  mAP50-95 0.614 vs. 0.617; the small gap comes from Ultralytics' validator
  preprocessing images slightly differently from `predict`.)
- **Detectors:** YOLOv8x, RT-DETR-l, Faster R-CNN with the same settings as
  before (30 epochs, COCO-pretrained, flips off), except Faster R-CNN's
  learning rate is scaled to its batch of 2: 0.02 x 2/16 = 0.0025.
- **Position-only model:** the same 6 features and classifier
  (HistGradientBoostingClassifier, defaults) trained on each fold's
  training and validation X-rays, tested on the fold's test X-rays, using
  the same mask boxes.
- **Scoring:** unchanged. A detection counts at confidence >= 0.5 and
  IoU >= 0.5; a missed tooth counts as wrong. Raw detections down to
  confidence 0.05 are saved so the IoU/confidence sweeps can run later
  without retraining.
- **Repeats:** CV seed 0 first. A second repeat (CV seed 1, new folds) is
  run if GPU quota allows. That is decided now, not after seeing results.

## Primary result and decision rule

Per detector: the gap = detector top-1 minus position-only top-1, pooled
over all 425 test X-rays (each tested once), with a 95% CI from 10,000
bootstrap draws of X-rays.

1. **Lower bound >= 15 pp:** "substantial image-based recognition" (the
   GO_NO_GO threshold) holds on uncropped X-rays without test-set epoch
   selection.
2. **Lower bound between 5 and 15 pp:** detectors still add beyond
   position, but the paper can no longer say the pre-set threshold was
   cleared; the wording changes to the measured gap and CI.
3. **Lower bound <= 5 pp:** Claim B fails on clean data. Stop and review
   before any writing.

The CV gap becomes the MICCAI headline in every case, since it is the
cleaner measurement. Whether the CJSJ revision uses it is the user's call.

**Consistency with Section 59.** If the CV gap is within 3 pp of the
uncropped-originals estimate for the same detector (YOLOv8x 19.7,
RT-DETR-l 19.9, Faster R-CNN 18.5), the two agree. If it is more than 3 pp
lower, check whether less training data (about 288 X-rays, against 820
image files before) explains it, by comparing detector accuracy to the old
runs, before reading it as a change in the finding.

## Guard against a broken run

A fold counts as a failed training run, to be rerun or investigated before
any result is reported, if that detector's matched-only top-1 is below 85%
or its missed rate is above 10% on that fold. The old runs reached 94 to 96%
matched-only and 1 to 2% missed, so these limits catch a broken run, not a
weaker but working one.

## Secondary analyses (same rules as before)

- **Missed vs. misnumbered, kappa:** same method and rules as
  `yolo_training/PHASE2_BATCH1_RULES.md` B and C, pooled over the 425 X-rays.
- **Joint failures (the mechanism):** teeth all three detectors and the
  position-only model get wrong. The mechanism replicates if the three
  detectors pick the same wrong tooth on at least 80% of them and that
  tooth matches the position-only guess above the 95th percentile of the
  permutation null (10,000 shuffles, as in Section 52). If either fails,
  the tie-breaker claim is reported as not replicated on uncropped data.
  If there are fewer than 30 joint failures, report them descriptively
  only.

## What this does not settle

One dataset; mask-derived boxes differ slightly from the annotators' boxes
(median 0.8% of image width against the uncropped X-rays' label files, checked 2026-10-01); 30
epochs with no hyperparameter search, as before.
