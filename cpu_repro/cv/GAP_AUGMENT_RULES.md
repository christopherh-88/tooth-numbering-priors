# Gap augmentation training: decision rule (frozen 2026-10-05)

The user approved running this and left the thresholds to be set on
measured anchors (2026-10-05). Change from the draft, made before any
run: a control model (same recipe, no augmentation, training seed 1) is
added, so the comparison is between two fresh retrains and the size of
retraining noise is measured instead of assumed.

Can training with simulated missing teeth reduce the gap errors? First
YOLOv8x only (seed 0 split, about 3 GPU-hours); the other two detectors
only if the result is rule 1 or close to it.

**Augmentation.** Offline, before training, with seed 0. Each training
X-ray (validation and test untouched) is replaced, with probability 0.5,
by an altered copy: one tooth T, drawn at random among teeth that are not
third molars and have both neighbors labeled, is erased (mask dilated
3 px, Telea inpainting) and its label removed; in half of these the
distal neighbor is also moved fully into the space (as in the closure
intervention) and its label moved with it. The training set keeps the
same number of images, so the number of training steps does not change.
Everything else is the seed 0 YOLOv8x recipe.

**Models.** Three YOLOv8x runs on the seed 0 split: the existing one
(original); a control retrained with the same recipe and data but
training seed 1 (no augmentation); and the augmented model (training
seed 0). The control's difference from the original measures retraining
noise. Paired by tooth, confidence-first matching, 95% CIs from 10,000
bootstrap draws of X-rays.

**Measures.**
- Top-1 on teeth next to a gap (1,364 teeth), new minus old.
- Overall top-1, new minus old.
- Of the 161 joint failures, the share the new YOLOv8x numbers right.

**Anchors.** YOLOv8x's top-1 next to a gap is 80.0% against 95.0%
overall (Section 74 data), so closing a fifth of that 15-point shortfall
is 3 points; the bar is 2 points, above the about 1.5-point
seed-to-seed noise of Section 66 D. The overall bar is 0.5 points, the
minimum detectable difference between detectors in Section 66 D.

**Rule** (augmented minus control, paired).
1. Top-1 next to a gap rises by at least 2 points with a CI excluding 0,
   and overall top-1 falls by less than 0.5 points: the augmentation
   works; train RT-DETR-l and Faster R-CNN the same way.
2. The CI of the next-to-gap change includes 0: it does not help.
3. Otherwise: report as measured.

Also reported: augmented minus original and control minus original (the
noise), for both measures.

**Limits.** The CIs cover the choice of X-rays, not training randomness;
retraining the same recipe also moves top-1 (Section 66 D puts seed-to-
seed differences of under about 1.5 points out of reach). A 2-point bar
next to gaps is chosen to stand above that. The altered copies are
artificial (see the closure intervention's limits).

**Cost.** Two YOLOv8x 5-fold trainings (control and augmented), about
3 hours each on a T4, then a CPU scoring kernel.
