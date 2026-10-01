# Label-error check on the CV joint failures: decision rules (frozen 2026-10-01)

Written 2026-10-01, before any of this is run, and approved by the user
the same day before running. This file is frozen; later changes go in a dated note at the end.

## The question

RESULTS.md Section 60: on 152 teeth that all three detectors and the
position-only model got wrong, the detectors chose the same wrong tooth
99.3% of the time, and it matched the position-only guess 92.7% of the
time. Two readings predict that pattern:

- **Tie-breaker:** the label is right; the detectors fall back on position
  between look-alike neighbors.
- **Label error:** the mask's FDI number is wrong; every model, position-only
  included, names the true tooth, which then counts as a shared error.

## Check A: do the two annotation files disagree on these teeth?

UFBA-425 has two sets of FDI labels for the same X-rays: the tooth masks
(used in the CV) and the Roboflow boxes (`Dataset/yolo_train_dataset`,
used in the original runs). For each labeled tooth, find the Roboflow box
on the same X-ray that overlaps its mask box best (IoU >= 0.5 after mapping
Roboflow coordinates back to the uncropped frame; for cropped copies the
mapping is fitted by least squares on teeth whose labels agree, and an
X-ray is skipped if the fit leaves a median residual above 1% of the image
width). Use the uncropped copy where one exists, otherwise all three copies,
and take the label they agree on (skip the tooth if they disagree).

Compare two groups:
- **J:** the 152 joint failures.
- **C:** every other tooth where all three detectors agreed with the mask
  label (the control: how often the two files disagree when nothing is wrong).

Let r_J = share of J where the Roboflow label equals the detectors' shared
answer (not the mask label), and r_C = share of C where the Roboflow label
differs from the mask label.

**Rule.**
1. If r_J >= 50%: label errors explain at least half the joint failures.
   The tie-breaker claim is withdrawn in its current form. Section 60's
   joint-failure statistics are recomputed on J minus these teeth and
   reported as the mechanism result, with this check stated.
2. If r_J <= 2 x r_C: the two annotation files show no excess disagreement
   on these teeth, so this check gives no support to the label-error
   reading. The tie-breaker reading stands pending the dentist's review.
3. In between: report r_J, recompute the joint-failure statistics without
   those teeth, and report both versions.

Limit: both label sets may come from the same annotators, so agreement
between them does not prove a label is right. Disagreement is the
informative direction.

## Check B: cleanlab

cleanlab's `find_label_issues` needs out-of-fold class probabilities for
every tooth. The only model in the CV with full probabilities is the
position-only model, and using it to judge the labels would be circular,
because the position-only model is part of the pattern under test. The
detectors' saved outputs give one class and one confidence per box, not a
probability for each class. So cleanlab is run only as a secondary,
labeled check: on the position-only model's out-of-fold probabilities,
reporting how many of the 152 it flags against its flag rate on C. No
claim depends on it.

## Check C: dentist review (prepared here, run by a person)

A blinded sheet: all 152 joint-failure teeth plus 152 control teeth drawn
at random from C (seed 0), shuffled together. For each, a crop of the
uncropped X-ray with the tooth outlined and its neighbors visible, and no
label shown. The dentist writes the FDI number, or "can't tell".

**Rule (applied once the review is back).** Error rate = share of teeth
where the dentist's number differs from the mask label, among those the
dentist could number.
1. Error rate on J > 50%: the joint failures are mostly label errors. The
   tie-breaker claim is withdrawn, and the result is reported as these
   models flagging label errors.
2. Error rate on J <= 20% and no more than 10 points above C's: the
   tie-breaker reading stands.
3. In between: the joint-failure statistics are recomputed on teeth the
   dentist confirmed, and both versions are reported.

## Where it runs

Check A and the cleanlab check need `per_tooth_predictions.csv`, which is
on Kaggle. They run in a CPU kernel that reads it there and writes small
outputs: a per-tooth flag table for J and the control rates. The dentist
sheet needs the X-ray images, which are in the repo, so it is built
locally from the flag table.
