# Average precision of the detectors (appendix table, reported only)

**Written 2026-10-10 before the numbers were computed; run 2026-10-10 on the instruction "do the cheap add on". Results: RESULTS.md Section 89.**

## Why

Metrics Reloaded (`paper/miccai/METRICS_RELOADED.md`) recommends a
multi-threshold metric, AP or FROC, next to a counting metric for
detectors that give scores. The paper reports top-1 and numbered F1 at
one cutoff plus cutoff sweeps and does not report AP (decision of
2026-10-05; the position-only model has no scores to compare with). A
reviewer used to detection papers will ask for AP. This computes it from
detections already saved, so it costs no GPU time, and puts it in an
appendix table. Nothing in the paper's claims depends on it.

## What

`ap_eval.py`, from the raw detections the training kernels saved
(`test_detections.csv`, confidence at least 0.05, normalized boxes), for
every set of three detectors we have:

- UFBA, splits 0, 1 and 2 (Sections 65, 75, 78), 9 sets;
- DENTEX, trained within DENTEX, training seeds 0 and 1 (Sections 86, 88),
  6 sets.

AP50 and AP50:95 (IoU 0.50 to 0.95 in steps of 0.05), COCO's matching and
101-point interpolation, **class-aware**: a detection counts for the tooth
number it gives, so a misnumbered box is a false positive of that number
and leaves the labeled tooth unmatched. Mean over the 32 numbers. 95% CIs
from 1,000 bootstrap draws of X-rays.

## Checks

- The implementation matches `pycocotools` on two synthetic cases to 3
  decimals (done 2026-10-10: AP50:95 32.026 and 31.836, AP50 63.879 and
  63.796 from both).
- A perfect detector scores 100.

## Reading, fixed in advance

No bars. The table is reported as measured. The paper states in one
sentence that AP is in the appendix and that top-1 stays the headline
because it puts the detectors and the position-only baseline, which has no
scores, on the same teeth.

## Limits

Detections below confidence 0.05 were not saved, so the tail of each
precision-recall curve is cut there (AP is a little lower than with all
detections). The position-only model has no scores and no AP.
