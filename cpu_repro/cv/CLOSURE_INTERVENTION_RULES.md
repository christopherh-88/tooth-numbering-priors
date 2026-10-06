# Closure intervention: decision rule (frozen 2026-10-05)

The user approved running this and left the thresholds to be set on
measured anchors (2026-10-05). Changes from the draft, made before any
run: rule 2 is now a statistical null (CI includes 0) instead of a fixed
1% floor, and the 5% bar is tied to the real-data rate below.

Follows RESULTS.md Sections 71 and 72. Erasing a tooth (open gap) seldom
makes all three detectors give a neighbor the missing number (4 of 2,260,
0.18%); at real gaps the shared errors sit where the space has closed
(AUROC 0.78), which is correlational. This test closes the gap on purpose.

**Targets.** The same targets as Section 71 (up to 3 teeth T per X-ray,
not third molars, both arch neighbors labeled, drawn with seed 0), so the
erasure and closure results are paired tooth by tooth.

**Mover.** The distal neighbor N of T (the one farther from the midline;
in real mouths it is usually the distal tooth that tips or drifts into the
space). Only movers that all three detectors number right on the intact
image are scored.

**Conditions** (each its own inference, seed 0 fold models,
confidence-first):
- open: T erased (mask dilated 3 px, Telea inpainting), as Section 71.
- half: T erased, then N cut out (its mask dilated 3 px), its old place
  inpainted, and N pasted back shifted toward T by half the distance d.
- closed: the same with the full d.
d is the horizontal distance between N's mask and the mesial neighbor M's
mask after T is erased, so in "closed" N's outline touches M. N's labeled
box moves with it. The position-only model is also run on the moved box.

**Measure.** j = share of movers that all three detectors give T's
number (a shared error), with 95% CIs from 10,000 bootstrap draws of
X-rays; and the same with the position-only model added (all four).
Per-detector fill rates are reported too.

**Anchor.** At real gaps closed to S of 0.1 or less, 22% of gaps hold a
joint failure (Section 72); a gap has two neighbors, so about 11% of
neighbors. The bar for "closing reproduces the real effect" is half of
that, 5%.

**Rule** (on j, closed against open, paired by mover).
1. The CI of j(closed) minus j(open) excludes 0 and j(closed) is at least
   5%: closing a gap causes shared errors at about the real rate.
2. The CI of j(closed) minus j(open) includes 0: closing the space alone
   does not produce them; the Section 72 link then comes from what goes
   with real closure (tipping, rotation, bone change) that a sideways move
   does not copy.
3. The CI excludes 0 but j(closed) is under 5%: closing causes shared
   errors, at well under the real rate; reported as a partial cause.

**Also reported, no rule.** j(half), to see whether the effect grows
with closure; per-detector fill rates in all three conditions; the share
of movers missed.

**Limits.** A sideways paste is not drift: real teeth tip and rotate, and
the paste leaves edges an inpainting cannot hide. Only the distal
neighbor moves. Examples are saved for a visual check before reading the
numbers.

**Cost.** About 1,250 targets x 4 images x 3 detectors: about 1 hour on
a T4.

## Dated note 2026-10-06: replication on the seed 1 models

Approved by the user. The same procedure on the seed 1 split models
(YOLOv8x, RT-DETR-l, Faster R-CNN trained on `folds_seed1.csv`), the same
targets (the seed 0 draw over the same teeth) and the seed 1
position-only models. Graded: replicated if the CI of j(closed) minus
j(open) excludes 0 and j(closed) falls inside the seed 0 95% CI (18.35 to
23.30%, RESULTS.md Section 76); otherwise both are reported with the
spread.
