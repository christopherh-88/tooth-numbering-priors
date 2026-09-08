# Mitigation experiment design (prepared, not yet run)

**Status: design only.** No code has been executed, no numbers exist yet.
This document exists so the mitigation step can be run immediately once a
trained detector is available (see `RESULTS.md` Section 11.5, "Error-pattern
correlation check") rather than designed under time pressure.

## Why this is needed

Section 11.2 of `RESULTS.md` draws the line between Claim A (geometry alone
predicts FDI identity - well-supported) and Claim B (a real detector
exploits this as a shortcut - untested). Measuring Claim B (the open error-
pattern correlation check) tells you whether the problem exists in a real
model. It does not, by itself, show the problem is fixable. A mitigation
that is proposed *and* evaluated is what closes the gap to a main-conference
contribution rather than a diagnostic-only one - this is the structure used
by the closest published comparable (Lin et al., "Shortcut Learning in
Medical Image Segmentation," MICCAI 2024): identify the shortcut, propose a
concrete intervention, re-measure.

## Two candidate interventions

### Primary: geometry-jitter augmentation

During detector training, apply a translation/scale perturbation to each
training crop or tile that is larger than the acquisition-protocol variance
naturally present in the dataset (panoramic radiographs are already fairly
canonicalized - see the two-precondition hypothesis in Section 11.4 - so the
natural jitter is small). Concretely: randomly translate/rescale the input
crop window per instance by an amount exceeding the dataset's natural
positional variance (to be measured empirically from the training split
before picking a jitter magnitude, not guessed), so that the canonical
pixel-space slot for a given FDI class is no longer reliably present at
train time. If the detector's accuracy holds up under this perturbation,
that's evidence it was relying on appearance rather than position; if
accuracy collapses, that itself is further (and more direct) evidence for
Claim B.

Implementation hook: goes into the training-time augmentation pipeline in
`notebooks/yolov8+unet/yolov8+unet_training.ipynb` (or the plain YOLOv8
notebook), as an additional transform applied before/alongside existing
augmentations - not yet located precisely in the notebook, needs a read-
through of the current augmentation block before wiring this in.

### Secondary / fallback: decorrelation auxiliary loss

Train the detector's classification head jointly with a penalty term that
discourages its predictions from agreeing with a frozen coordinate-only
reference classifier (the existing logistic-regression/GBT model from
Section 2) beyond what accuracy requires - a confound-removal-style loss,
similar in spirit to a gradient-reversal position-predictor. More invasive
to implement than the augmentation approach; keep as fallback if jitter
augmentation turns out to not move the needle.

## Evaluation protocol

Train the real detector twice on the same split (with and without the
chosen intervention). For each of the two resulting models:

1. Overall top-1 / quadrant / tooth-type accuracy (same metrics as
   Section 2), to check the intervention doesn't just destroy performance.
2. Fraction of the detector's predictions that agree with the coordinate-
   only baseline's predictions, specifically on the subset where the
   detector and the coordinate-only model disagree with ground truth -
   this is the direct measure of shortcut reliance.
3. Run both models' predictions through the existing
   `is_mirror_quadrant_error` / `is_neighbor_error` taxonomy
   (`build_coord_baseline.py`) and compare the error-type distribution
   before vs. after the intervention.

## Falsification threshold (stated in advance, per the discipline used in
`RESULTS.md` Section 2)

The intervention will be treated as having **failed to demonstrate
mitigation** if the post-intervention agreement-with-geometry-only-model
rate (metric 2 above) drops by less than half the drop seen in the
shuffled-geometry negative control's collapse (Section 4 / `controls/`) -
i.e., if the intervention barely moves the number relative to how far a
model *could* be pushed away from the geometry correlation, it should be
reported as a negative/inconclusive result, not reframed as a partial
success after the fact.

## Open items before this can run

- ~~Read `notebooks/yolov8+unet/yolov8+unet_training.ipynb` to find the
  exact augmentation pipeline location~~ **Done (static read, 2026-09-08,
  no execution).** Framework is plain NumPy/OpenCV, not
  Albumentations/torchvision. The only existing augmentation is a single
  `augment()` function (cell 21): a 50%-probability `np.fliplr` applied
  identically to the image, the segmentation mask, and `input_bb`. This
  is the natural hook point for the geometry-jitter intervention - it's
  the only place per-sample spatial augmentation happens in this
  notebook.
- **Flip/label-mismatch bug: CONFIRMED, 2026-09-08.** Static read +
  minimal CPU-only reproduction (no training, no model) - a scratch
  verification script, not part of this repo (logic and full run log
  below, so this is reproducible without the original file).

  **Exact source.** `notebooks/yolov8+unet/yolov8+unet_training.ipynb`,
  cell 13, line 12:
  ```
  binary_map[int(class1), y1:y2, x1:x2] = 1
  ```
  (channel index = `class1` = FDI class id, set once per box, never
  touched again after this line) and cell 21, line 4:
  ```
  input_bb = np.fliplr(input_bb)
  ```
  (applied to `input_bb` after `resize_img`'s `np.transpose(input_bb,
  axes=[1, 2, 0])`, i.e. to a `(H, W, C)` array - `np.fliplr` flips axis
  1, the `W` axis, and leaves axis 2, the channel/class axis, untouched).
  Grepped the full notebook for every reference to `class1` / `class_id`
  / `channel` between cells 13 and 21 inclusive: **no remapping code
  exists anywhere in that span.**

  **Reproduction.** Took a real UFBA-425 label file with a matched
  same-tooth-type, opposite-quadrant pair -
  `Dataset/yolo_train_dataset/test/labels/cate1-00026_jpg.rf.365e2e2d1d708d69d19a27a09f0b05de.txt`,
  class 0 (FDI 11, `CODE_TO_IDX` convention from `build_coord_baseline.py`)
  and class 8 (FDI 21) - and reproduced cell 13's `binary_map`
  construction and cell 21's `fliplr` call verbatim, line-for-line, on a
  32x640x640 array (no image pixels, no model - box placement and the
  flip are the entire test). Result:

  ```
  Before flip:
    channel 0 (FDI 11): x range [319, 343]
    channel 8 (FDI 21): x range [346, 370]

  After fliplr (exact cell-21 call, channel axis untouched):
    channel 0: x range [296, 320]
    channel 8: x range [269, 293]

  Post-flip channel 0 == mirror of pre-flip channel 0
    (channel index NOT remapped): True
  ```
  Channel 0's content moved to the opposite side of the canvas center
  (640/2 = 320) exactly as a spatial mirror predicts (319-343 -> 296-320,
  consistent with reflection about x=320), while remaining stored under
  channel index 0 the entire time. Channel identity tracks nothing; only
  pixel content moves.

  **Scope.** This is not a messy or random misalignment - it is a fully
  deterministic, systematic swap. Because `MIRROR_QUADRANTS` in
  `build_coord_baseline.py` (`{("1","2"),("2","1"),("3","4"),("4","3")}`)
  already defines the exact anatomical mirror-pairing (upper-left <->
  upper-right, lower-left <-> lower-right, same tooth-type digit), the
  affected classes on any flipped view are precisely the 32 classes
  paired by that same relation - e.g. every 1x<->2x and 3x<->4x pair,
  applied symmetrically. **Fraction of training exposure affected: the
  flip's own trigger probability, `np.random.uniform() > 0.5` -> ~50% of
  augmented training views**, on every box in the image simultaneously
  (not per-box independent).

  **Remediation: IMPLEMENTED and verified, 2026-09-08.** Only
  `notebooks/yolov8+unet/yolov8+unet_training.ipynb` cell 21 was touched
  (nothing else in the notebook). New cell 21 source:
  ```python
  FDI_CODES = [
      "11", "12", "13", "14", "15", "16", "17", "18",
      "21", "22", "23", "24", "25", "26", "27", "28",
      "31", "32", "33", "34", "35", "36", "37", "38",
      "41", "42", "43", "44", "45", "46", "47", "48",
  ]
  MIRROR_QUADRANTS = {("1", "2"), ("2", "1"), ("3", "4"), ("4", "3")}
  MIRROR_MAP = [
      next(j for j, c2 in enumerate(FDI_CODES)
           if c2[1] == c[1] and (c[0], c2[0]) in MIRROR_QUADRANTS)
      for c in FDI_CODES
  ]

  def augment(input_image,input_mask,input_bb):
      if np.random.uniform() > 0.5:
          input_image = np.fliplr(input_image)
          input_mask = np.fliplr(input_mask)
          input_bb = np.fliplr(input_bb)
          input_bb = input_bb[:, :, MIRROR_MAP]

      return input_image,input_mask,input_bb
  ```
  `FDI_CODES`/`MIRROR_QUADRANTS` are copied verbatim from
  `build_coord_baseline.py` (confirmed identical to this dataset's own
  `Dataset/yolo_train_dataset/data.yaml` `names:` ordering - same 32
  codes, same index order - so reusing that class-id assumption here is
  justified, not assumed blind). `MIRROR_MAP` is a 32-entry involution;
  spot-checked `MIRROR_MAP[0] == 8` (FDI 11 -> 21) and
  `MIRROR_MAP[8] == 0` (FDI 21 -> 11) in the verification run below.

  **Correction to this document's own earlier spec:** the original
  remediation note here said to check
  `flipped_by_channel[0] == pre-flip binary_map[8]` as the post-fix
  invariant. That was imprecise and, on reflection, wrong as literally
  stated - two independent real teeth (FDI 11 and FDI 21 in this same
  test image) are not exact pixel-for-pixel mirrors of each other, so
  that exact equality would not hold even with a fully correct fix. The
  invariant that actually must hold, and the one checked below, is:
  post-fix, channel `MIRROR_MAP[c]` contains `fliplr(pre-flip channel c)`
  - the (correctly mirrored) *pixel content* moves to the class-correct
  channel, rather than coincidentally matching a different tooth's
  independent box.

  **Re-verification (same real UFBA-425 box data as the bug-confirmation
  run, plus the fix applied):**
  ```
  MIRROR_MAP[0] (FDI 11 -> ?) = 8 (FDI 21)
  MIRROR_MAP[8] (FDI 21 -> ?) = 0 (FDI 11)

  post-fix channel 8 == fliplr(pre-flip channel 0)  [class-correct]: True
  post-fix channel 0 == fliplr(pre-flip channel 8)  [class-correct]: True
  (sanity) post-fix channel 0 still == fliplr(pre-flip channel 0)
    [old buggy behavior, should now be False]: False

  PASS
  ```
  Confirms the fix: mirrored pixel content now lands under the
  class-correct channel, and the old buggy behavior (content staying
  under its original channel) no longer occurs.
- Measure the dataset's natural positional variance (not yet done) to pick
  a jitter magnitude that is deliberately larger than it, rather than an
  arbitrary number.
- Requires GPU (Kaggle or equivalent) - blocked until compute is available.
  Task 2 (error-pattern correlation check) now depends on this fix being
  in place before any detector is trained on this notebook - training on
  the pre-fix pipeline would confound Claim-B evidence with this labeling
  artifact.

## Related audit: error-taxonomy correctness (`build_coord_baseline.py`)

Static-read audit (2026-09-08, no execution) of `is_mirror_quadrant_error`
and `is_neighbor_error` (both used by the mitigation evaluation protocol
above and already in production use for Section 2/10's error breakdowns).
No bugs found. Specifically checked: `MIRROR_QUADRANTS` correctly pairs
upper-with-upper (1<->2) and lower-with-lower (3<->4), not all four
quadrants mutually; neighbor-detection correctly requires same quadrant
(a same-tooth-type error across the midline, e.g. 11 vs. 21, is correctly
classified as mirror-quadrant, not neighbor, since they're anatomically on
opposite sides despite adjacent numbering - a deliberate and correct
definitional choice, not an edge-case gap); both functions are only ever
called on the wrong-prediction subset (`y_true[wrong]`), so the undefined
true_id==pred_id case never actually executes. This taxonomy code can be
trusted as-is for Task 2 once real detector predictions exist.
