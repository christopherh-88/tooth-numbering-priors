# Space closure at gaps: decision rule (frozen 2026-10-05)

Approved by the user 2026-10-05, asking for the most accurate measure.
Change from the draft, made before any run: the primary space measure uses
the tooth masks instead of the boxes (box version kept as secondary).

Follows RESULTS.md Sections 69 and 71. At real gaps all three detectors
and the position-only model give a tooth the missing tooth's number;
erasing a tooth from the image (which leaves a full-width empty space)
seldom makes them do it together. The idea to test: the shared failure
happens where the teeth on either side have moved into the space, so the
space is partly or fully closed and the arch reads as if nothing were
missing.

**Data.** The confidence-first per-tooth file (uncropped, seed 0) and
the tooth masks (`Dataset/bb_u_net_dataset/labels`). No X-ray pixels, no
GPU. Kaggle CPU kernel.

**Unit: a gap slot.** A position M with no labeled tooth in an X-ray,
where M is not a third molar and both arch neighbors of M (X and Z, in
the arch order of `gap_check.py`, crossing the midline where needed) are
labeled. Slots in longer gaps (a neighbor also missing) are left out and
counted.

**Measure: space S (primary, masks).** The shortest distance in pixels
between any pixel of X's mask and any pixel of Z's mask (0 if they touch
or overlap), divided by the median box width in pixels of tooth M over
all X-rays where M is labeled. This follows the arch curve and is not
inflated by tilted teeth. S = 1 is a space the size of the missing tooth;
S = 0 is a closed space.

**Secondary: S_box.** The horizontal distance between the facing edges of
X's and Z's boxes over the same width (negative when the boxes overlap).
Reported with the same statistics, not graded.

**Outcome: slot filled.** X or Z is a joint failure (all three detectors
and the position-only model wrong, same answer, none missed, as in
Section 65) whose shared answer is M's number.

**Rule** (on the primary S; CIs from 10,000 bootstrap draws of X-rays).
1. Filled slots have smaller S: the X-ray bootstrap 95% CI of the median
   S (unfilled minus filled) excludes 0, and the AUROC of S for telling
   filled from unfilled slots (lower S = filled) is at least 0.75. Space
   closure explains the shared failure at gaps.
2. The AUROC CI includes 0.5: space closure does not explain it.
3. Otherwise: report as measured.

**Also reported, no rule.**
- Distribution of S for filled and unfilled slots; share with S < 0.5.
- The same AUROC with each model alone as the outcome (that model gives X
  or Z the number M, whatever the others do): YOLOv8x, RT-DETR-l, Faster
  R-CNN, position-only.
- Of all slots with S < 0.5, the share filled. This is the precision of a
  "closed space" review flag.

**Limits.** The filled outcome requires the position-only model to be
wrong, and closing a space also moves X's absolute position toward M's,
which is what the position-only model reads. The per-model AUROCs show
how far the detectors, which do not use box positions, track S on their
own. Masks are 2D outlines on a panoramic projection, so overlap in the
image can be projection, not contact. The box measure ignores the arch
curve and tilt.

**Cost.** A few minutes on a Kaggle CPU kernel (reading the masks).
