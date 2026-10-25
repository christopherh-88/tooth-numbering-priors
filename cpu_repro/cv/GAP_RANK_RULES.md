# Closure-ranked review flag: can the closed-gap mechanism order the review?

Approved 2026-10-25 with the thresholds below, frozen before any of it is run.
The thresholds were set from the power simulation in `gap_rank_power.py`
(see "Why these thresholds"), not from any ranked result.

## Question

Section 84's gap flag catches 41 to 47% of the joint failures but marks 12 to
16% of teeth (3 to 4 per X-ray), so most flagged teeth are fine. The paper
found that a gap with closed neighbors is where the shared errors sit (space
closure S, AUROC 0.78 from the masks). Can the same idea, computed from the
detector's own boxes with no labels, put the flagged teeth in a useful order,
so that a reviewer who checks only about one tooth per X-ray finds more
of the shared errors than confidence would?

## Ranking (per detector, per X-ray, from its output only)

Start from the Section 84 flag (gap-adjacent or duplicate, same candidates,
confidence 0.5, third-molar positions not counted as gaps).

- For a gap-adjacent tooth, take the empty position it borders and the two
  teeth numbered as that position's arch neighbors. The estimated space is the
  horizontal distance between the boxes of those two teeth (0 if they overlap
  or touch), divided by the mean of their two box widths.
- If a tooth borders several empty positions, use the smallest space.
- Rank flagged teeth by estimated space, smallest first. Teeth flagged only as
  duplicates, and gap-adjacent teeth whose space cannot be computed (a
  neighbor missing), come after all others. Ties are broken by the tooth's
  position in the X-ray, left to right, then by image id (not by confidence,
  so confidence does not leak into the ranking).

No labels, masks or other detectors are used.

## Measures (confidence-first matching; 10,000 bootstrap draws of X-rays)

Review budget B = 4% of the detector's matched teeth (about 1 per X-ray, a
third of the Section 84 flag), filled from the top of each list over the whole
test set. Compare, at the same number of teeth:

1. Ranked flag: joint-failure recall.
2. Confidence: the least-confident matched teeth (as in Sections 66 and 84).
3. Unranked flag cut to the same count: its recall scaled by B over the flag's
   size, which is what a random subset of the flag gives on average.

Paired X-ray bootstrap CIs for ranked minus confidence and ranked minus
unranked. Check that the joint-failure sets reproduce 161, 166 and 165.

Also reported, no rule: recall at 2%, 8% and the full flag; AUROC of the
estimated space for separating joint-failure teeth from other flagged teeth;
precision at B; the same ranking with the true box space from the labels, as a
ceiling.

## Graded

1. **Better than confidence (seed 0):** for at least 2 of 3 detectors, ranked
   recall at B is at least 1.5 times confidence's, and the CI of the
   difference excludes 0. Same bar as Section 84, so the two are judged alike.
2. **Ranking helps:** for the same detectors, ranked recall at B is at least
   1.5 times the unranked flag cut to the same count, with the CI of the
   difference excluding 0. Without this the ordering adds nothing over the flag
   and the paper reports the flag only.
3. **Replicated:** rules 1 and 2 also hold on the seed 1 and seed 2 split
   models, each for at least 2 of 3 detectors.

A detector counts toward "at least 2 of 3" only if it meets rule 1 and rule 2
together (`gap_rank.py --grade` applies this).

Rule 1 alone is not enough to claim a new contribution, because Section 84's
flag already beat confidence on most pairs.

## Why these thresholds

They were chosen from the statistics of the test, not from any ranked result
(none exists). The Section 84 flag, seed 0, YOLOv8x, marks 1,327 teeth and
holds 73 of the 161 joint failures. A random subset of it catches 31% at
B = 8% but only 16% at B = 4%, and the best possible ranking caps at 45%, so
at 8% there is little room to show an effect and at 4% there is a lot.
`gap_rank_power.py` simulates a ranking of a given strength (AUROC among the
flagged teeth) on those counts and asks how often a rule passes. Teeth are
treated as independent, so the pass rates are optimistic.

| True AUROC of the ranking | B 8%, 1.25 times | B 4%, 1.5 times | B 2%, 1.5 times |
|---|---|---|---|
| 0.5 (no signal) | 0.00 | 0.00 | 0.02 |
| 0.6 (weak) | 0.11 | 0.28 | 0.52 |
| 0.7 | 0.83 | 0.96 | 0.99 |
| 0.8 | 1.00 | 1.00 | 1.00 |

B = 4% with 1.5 times never passes a ranking with no signal, rarely passes a
weak one, and nearly always passes an AUROC of 0.7. B = 2% passes a coin flip
2% of the time and a weak ranking half the time, so it is too lenient; B = 8%
at 1.25 times misses real rankings (0.83 at AUROC 0.7). The true AUROC of the
box-based space is unknown; for reference, the space from the masks separates
gaps with and without a joint failure at 0.78 (gap level, not tooth level).
Because the real teeth cluster by X-ray, the real pass rates will be lower
than these.

## Paper use

- All three rules met: one result paragraph in Discussion replaces the
  unranked-flag sentence, with its numbers, and the Limitations keeps that the
  ranking was tested on this data only.
- Rule 1 or 2 not met: the paper keeps the Section 84 sentence unchanged and
  the failed result goes in RESULTS.md only. No wording is added.

## Implementation

`gap_rank.py` (asserts that its flag equals Section 84's on every X-ray, and
that the joint-failure sets are 161, 166 and 165) and the Kaggle CPU kernel
`tooth-numbering-cv-gaprank` from `kaggle/make_phase3_kernels.py`. It was
checked before the run on synthetic detections only (the real ones are on
Kaggle), for crashes and for the space estimate on hand-made cases.

## Cost

One Kaggle CPU kernel reading the saved raw detections of seeds 0, 1 and 2
(no GPU, no training), 20 to 40 minutes, like Section 84. It ran after
sign-off, on 2026-10-25.

## After the run

Graded in RESULTS.md Section 98: rules 1 and 2 met on seed 0, rule 3 not met,
so the paper is unchanged. The reported-only ceiling in this file's measures
turned out to be invalid by construction (see Section 98) and is not used.
