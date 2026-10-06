# Gap flag: can the detector's own output point a reviewer at the shared errors?

Approved 2026-10-06 with the thresholds below (asked to set them as well
as possible), frozen before any of it is run.

## Question

Section 69 found joint failures next to missing teeth 5 times as often as
correct teeth, but it read the gaps from the labels, which a deployed
system does not have, and did not count the review cost. Section 66 found
the detectors' own confidence catches 8% of joint failures. Can a flag
built only from one detector's output do better at the same cost?

## Flag (per detector, per X-ray, from its output only)

The detector's raw detections are grouped into one candidate per tooth
(`batch3.candidates`, the Section 66 collapse, which changes top-1 by under
0.04 pp), each with its top class; candidates below confidence 0.5 are
dropped.

- **Gap-adjacent:** an FDI position 1 to 7 in a quadrant has no candidate
  (third-molar positions are never counted as gaps, since they are often
  absent); every candidate numbered as an arch neighbor of that position,
  in image order (`batch3.ORDER`, so 11 and 21 are neighbors), is flagged.
- **Duplicate:** two or more candidates share a number; all are flagged.
- **Flag** = gap-adjacent or duplicate.

A labeled tooth is flagged when its matched detection (confidence-first,
as scored) belongs to a flagged candidate.

## Measures (confidence-first; 10,000 bootstrap draws of X-rays)

- Review cost: share of the detector's matched teeth flagged.
- Recall on the joint failures (all three detectors and the position-only
  model wrong with the same answer), and on the detector's own
  misnumbered teeth.
- Baseline: the confidence flag at the same count (the least confident
  matched teeth, as in Section 66).
- Paired difference in joint-failure recall, flag minus confidence.

Check: the joint-failure sets reproduce 161 (seed 0, Section 65), 166
(seed 1, Section 75) and 165 (seed 2, Section 78).

## Graded

Thresholds and why:

1. **Useful (seed 0):** for at least 2 of the 3 detectors the flag catches
   at least 1.5 times as many joint failures as confidence at the same
   cost, and the CI of the difference excludes 0. 1.5 times is the bar
   frozen for the Section 66 review flag, so the two flags are judged
   the same way; the CI makes it a tested difference, not a ratio of
   small counts.
2. **Practical:** the review cost is at most 15% of teeth for those
   detectors, about 4 of the 27 teeth on an average X-ray. If rule 1 holds
   but the cost is above 15%, it is reported as useful but costly.
3. **Replicated:** rule 1 also holds on the seed 1 and seed 2 split models,
   each against its own joint-failure set.

Otherwise the flag is reported as not better than confidence, and the paper
says that.

Also reported, no rule: each part alone (gap-adjacent, duplicate); the
same flag with third-molar positions counted as gaps; recall on each
detector's own misnumbered teeth; the Section 69 label-gap figure (44.7%)
as a ceiling for a gap rule that knew the true gaps.

## Cost

One Kaggle CPU kernel reading the saved raw detections of the seed 0, 1
and 2 models (no GPU, no training), about 20 to 40 minutes.
