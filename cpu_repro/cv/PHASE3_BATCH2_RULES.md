# Phase 3 batch 2: decision rules (frozen 2026-10-01)

Written 2026-10-01, before any of these is run, and approved by the user
the same day ("gpu plan as written"). Frozen; later changes go in a dated
note at the end. All scoring uses confidence-first matching (RESULTS.md
Section 64), confidence >= 0.5, IoU >= 0.5, unless stated.

## 1. Shift test v2 (T4, inference only)

Same conditions and models as PHASE3_RULES.md item 1, with three changes:
confidence-first matching; every raw detection (confidence >= 0.05) is
saved; and the follow rate is computed only on changed teeth whose
position-only prediction for the shifted box is wrong, so a change to the
correct number never counts as following position. The unrestricted
follow rate is reported alongside.

**Rule** (on dx = +-10%, per detector, using the restricted follow rate):
the same three outcomes as PHASE3_RULES.md item 1. If fewer than 20
changed teeth qualify, the follow rate is reported without a grade.

## 2. Context masking (T4, inference only, same kernel)

**Sample.** All joint same-wrong teeth from the confidence-first rescore
(`tooth-numbering-cv-confmatch-s0`) and 1,000 control teeth drawn with
seed 0 from teeth all three detectors number correctly.

**Method.** For each sampled tooth and k = 0.5, 1, 2: black out every pixel
outside the tooth's labeled box widened by k box widths on each side and
k box heights above and below, then run the detector from the fold that
tests the X-ray and match that tooth alone. The full-image answer is the
k = infinity reference.

**Rule** (control teeth, k = 1, per detector; change in top-1 from the
full image, 95% CI by X-ray bootstrap):
1. Drop of 20 pp or more: the detector numbers teeth from wider context
   (arch order, neighbors), consistent with the relative-position reading.
2. Drop under 5 pp: it numbers from the tooth and its immediate
   surroundings; the relative-position reading is not supported for
   correct teeth.
3. Otherwise: report as measured.

Joint failures, reported with no separate rule: at each k, the share
still given the shared wrong answer, the share now correct, and missed.

## 3. CV seed 1 (T4 training, YOLOv8x and RT-DETR-l)

New split: `prepare_uncropped_cv.py --split-seed 1` gives
`folds_seed1.csv` (same stratified procedure, seed 1). Same training
settings as seed 0. Scored with confidence-first matching.

**Rule** (per detector): if the seed 1 gap and the two-detector joint
same-wrong position-match rate fall inside the seed 0 95% CIs (seed 0
recomputed with confidence-first matching and the same two detectors),
seed 1 is reported as a replication of seed 0. Otherwise both are
reported with the spread, and the paper's headline uses the mean of the
two seeds with the wider of the two CIs.
