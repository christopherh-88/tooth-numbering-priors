# Phase 3 batch 3 (CPU only): decision rules (frozen 2026-10-01)

Approved by the user as written on 2026-10-01, before any of it was run.
Frozen; later changes go in a dated note at the end. Script `batch3.py`. All four use the saved
seed 0 raw detections (confidence >= 0.05) and confidence-first matching
(RESULTS.md Section 64), and run in one Kaggle CPU kernel.

## A. Position-only model on predicted boxes

Section 60 feeds the position-only model the labeled boxes, which a
deployed system would not have. Here the same fold models are applied to
each detector's matched predicted box instead.

**Rule** (per detector): if the position-only top-1 on predicted boxes is
within 2 pp of its top-1 on labeled boxes (78.5%), the Section 60 gap
stands as stated. Otherwise the paper reports both gaps and calls the
labeled-box gap an upper bound on what position alone gives.

## B. Tooth-order post-processor

Per X-ray and detector: group raw detections that overlap (IoU >= 0.5)
into tooth candidates, each with a score per class (the highest
confidence of any detection of that class in the group). Split candidates
into upper and lower arch by the class with the highest score; sort each
arch left to right in the image. Relabel with dynamic programming
(Viterbi): pick one class per candidate to maximize the summed log score,
subject to the image-left-to-right order being strictly increasing in arch
position (18 ... 11, 21 ... 28 upper; 48 ... 41, 31 ... 38 lower), gaps
allowed for missing teeth. Score the relabeled candidates as in Section 60.

**Rule** (per detector, compared with the same detector without the
post-processor):
1. Top-1 rises by 0.5 pp or more (95% CI by X-ray bootstrap above 0) and
   the joint same-wrong count falls by 30% or more: an order constraint
   fixes a real share of position-consistent errors; reported as a
   practical fix.
2. Top-1 changes by less than 0.5 pp either way: the detectors already
   obey arch order; the errors that remain are order-consistent (a whole
   run of teeth shifted by one), which an order constraint cannot see.
3. Top-1 falls by 0.5 pp or more: reported as a negative result.

## C. Risk-coverage and a look-alike review rule

Risk-coverage curve per detector: teeth sorted by match confidence,
error rate (missed or misnumbered) against coverage; area under the curve
(AURC) with an X-ray bootstrap CI.

Look-alike flag: a tooth is flagged when its tooth candidate (as in B)
also scores an adjacent tooth in the same quadrant (FDI position +-1) at
confidence >= 0.1. Compared with a confidence threshold set to flag the
same number of teeth.

**Rule** (per detector): if, at equal numbers of flagged teeth, the
look-alike flag catches at least 1.5 times as many joint failures as the
confidence threshold, the paper proposes it as the review rule. Otherwise
the paper reports both and recommends neither.

## D. Power for the seed and detector comparisons

From the seed 0 X-ray bootstrap: the smallest gap difference between two
detectors (and between two seeds) that a paired comparison over 425
X-rays would detect with 80% power at alpha 0.05. No rule: it sets what
the paper may call a difference and what it must call a tie.

## Clarifications written at freezing, before the run

- A: graded on matched teeth, comparing the position-only model on the
  predicted box with the same model on the labeled box of the same teeth
  (missed teeth have no predicted box). The all-teeth numbers, with missed
  teeth counted wrong, are reported alongside.
- B: the Viterbi step uses only candidates whose top confidence is >= 0.5
  (the ones that get scored); the per-class scores inside a candidate come
  from all its detections down to 0.05. "Collapse only" (one box per
  candidate, its top class, no order constraint) is reported too, so the
  effect of the order constraint is separate from that of removing
  duplicates. The joint count per detector swaps in that detector's
  post-processed answers and keeps the other two as they are; all three
  post-processed together is reported alongside. A rise of 0.5 pp or more
  without a 30% fall in joint failures is outside rules 1 to 3 and is
  reported as measured.
- B and C use the confidence-first joint set (RESULTS.md Section 65, 161 teeth).
- D: the seed comparison treats the two splits as independent, which
  overstates its standard error (both seeds test the same X-rays), so the
  minimum detectable seed difference is conservative.
