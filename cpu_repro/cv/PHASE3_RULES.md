# Phase 3 batch 1: decision rules (frozen 2026-10-01)

Written 2026-10-01, before any of these is run, and approved by the user
the same day before running. This file is frozen; later changes go in a dated note at the end.

All three use the CV models and outputs already made (RESULTS.md Section
60). No training. Everything runs on Kaggle, reading the training kernels'
outputs there: the shift test on a T4 (inference only, all three
detectors), the other two in a CPU kernel. Nothing runs locally.

## 1. Shift test: does the detector's answer follow position? (causal)

The observational results (Sections 46-60) show detectors' errors line up
with position. An intervention tests it directly: move every tooth in the
image and see whether the tooth numbers move with it.

**Method.** For each of the 425 X-rays and each detector (YOLOv8x,
RT-DETR-l, Faster R-CNN), using that detector's model from the fold that
tests the X-ray: shift the image content horizontally by dx = -10%, -5%,
+5%, +10% of the width, and vertically by dy = -5%, +5% of the height,
filling the uncovered strip with black (the zero padding a detector
already sees at image borders). Labeled boxes move with the content;
teeth pushed partly outside the frame (any box edge outside [0, 1]) are
dropped from that condition. Score at confidence >= 0.5, IoU >= 0.5 against
the shifted boxes, as in Section 60.

Per condition, report:
- top-1 accuracy and the change from the unshifted image (paired, 95% CI
  by X-ray bootstrap);
- **follow rate:** among teeth whose detector answer changes between the
  unshifted and shifted image (and is matched in both), the share whose new
  answer equals the position-only model's prediction for the shifted box.
  Chance level: the same share with the position-only predictions shuffled
  among those teeth (10,000 shuffles).

For scale: a 10% horizontal shift moves a tooth about 2 tooth-widths
(median box width 5.1% of the image), so a model that read only position
would renumber most teeth.

**Rule** (judged on the +-10% horizontal shifts, the strongest):
Applied to each detector separately.
1. Accuracy drops by less than 2 pp, and the follow rate is not above
   its permutation 95th percentile: the detector does not use absolute position
   to number teeth; the tie-breaker reading is about relative, not
   absolute, position.
2. Accuracy drops by 5 pp or more and the follow rate is above 50% and
   above its permutation 95th percentile: absolute position causally
   drives some of its answers. That is direct evidence for the shortcut on
   the teeth that change.
3. Anything else: report the numbers as measured, with no causal claim
   either way.

The +-5% horizontal and +-5% vertical shifts are reported alongside, with
no separate rule.

Limits: shifting
with black fill changes the image border, which is itself a cue; this tests
reliance at test time, not what the model learned in training (the
retraining arm of item 8 needs GPU time and comes later).

## 2. Confidence on joint failures (calibration)

**Method.** From `per_tooth_predictions.csv`: for each detector, compare its
confidence on the 151 joint failures with its confidence on correctly
numbered teeth. Report the mean confidence of each group and the AUROC of
confidence for telling the two apart (95% CI by X-ray bootstrap).

**Rule.**
1. AUROC >= 0.8: confidence separates these errors well; a confidence
   threshold would catch most of them.
2. AUROC < 0.7: these errors are made confidently; a confidence threshold
   would not catch them, which motivates a different review rule (item 11,
   flag look-alike neighbors).
3. In between: report as measured.

## 3. Sensitivity to the confidence and IoU cutoffs

**Method.** Recompute the Section 60 pooled gap for each detector at
confidence cutoffs 0.25, 0.5, 0.75 and IoU cutoffs 0.3, 0.5, 0.7 (9
combinations), from the raw detections saved down to confidence 0.05.

**Rule.** If every gap stays within 2 pp of the Section 60 value, the
headline does not depend on the cutoffs, and the paper says so with the
range. Otherwise the paper reports the range and names the cutoff that
moves it.


## Note 2026-10-01 (after items 2 and 3 ran): matcher-order check

Added after the sweep (RESULTS.md Section 62), approved by the user before
running. RT-DETR-l at confidence 0.25 lost about 16 pp while its missed rate
fell, which points at the matcher: `score_cv.match` pairs teeth with boxes
by IoU alone. Rescore all 9 cutoffs with confidence-first matching (pairs
sorted by detector confidence, then IoU), next to the IoU-first result.

**Rule.**
- If the RT-DETR-l confidence-0.25 rows come back to within 2 pp of their
  confidence-0.5 value, the drop is a matcher artifact and the paper says so.
- If the 0.5 / 0.5 gap of any detector changes by more than 0.5 pp,
  Section 60 is reported under both matchers.
