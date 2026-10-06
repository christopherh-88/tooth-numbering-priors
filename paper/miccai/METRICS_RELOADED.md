# Metrics Reloaded check

Metric choices checked against the Metrics Reloaded framework (Maier-Hein
et al., Nature Methods 2024). The framework's questions are paraphrased
here from memory, not quoted; check the wording against the paper and its
online tool before submission.

## Problem fingerprint

| Question | This study |
|---|---|
| Problem category | Object detection with a class per object: find each tooth, give it one of 32 FDI numbers. Numbering is the quantity of interest, not localization. |
| Reference annotation | Boxes from tooth masks (UFBA-425), one annotator set, no repeat labels. A confident-learning check flags about 10% of the joint failures as possible label errors (RESULTS.md Section 65). An expert re-read was dropped (2026-10-05). |
| Class balance | Balanced: 289 to 417 teeth per class (third molars fewest). Macro and micro top-1 differ by under 0.4 pp for every model. |
| Object size and spacing | Small, touching or overlapping objects in a row. Neighbors are the main confusion (92.8 to 94.8% of shared errors are one place off, Sections 65, 70). |
| Stratification needed | By X-ray: teeth in one X-ray are not independent. All CIs resample X-rays. Folds are grouped by X-ray. |
| Is a missed object a failure | Yes. A tooth with no matching box counts as wrong. Missed and misnumbered are also reported apart. |
| Do predicted scores matter | Yes for the review-flag analyses (risk-coverage, Section 66 C; calibration on joint failures, Section 65). |

## Localization and assignment

| Choice | This study | Status |
|---|---|---|
| Localization criterion | Box IoU at least 0.5 | Done. Sweeps over IoU 0.3 to 0.7 and confidence 0.25 to 0.75 (Sections 62, 65). |
| Assignment | Greedy one-to-one, highest confidence first (COCO order), class-agnostic so a misnumbered box is still matched | Done. The IoU-first order used before Section 64 inflated RT-DETR-l's errors at confidence 0.25; the paper uses confidence-first. |
| Confidence threshold | 0.5 | Done, with the same sweep. |

## Metrics

| Metric | Why | Status |
|---|---|---|
| Top-1 over all labeled teeth (missed = wrong) | Headline, directly comparable with the position-only model, which always answers | Reported with X-ray bootstrap CIs |
| Gap over position-only (pp, paired) | The study's question | Reported |
| Missed and misnumbered rates | Separates detection from numbering | Reported |
| Top-1 on matched teeth only | Numbering given detection | Reported |
| Macro top-1 (per-class mean) | Class balance check | Computed in `benchmark/score.py`; equals micro within 0.4 pp |
| False positives and numbered F1 | Top-1 ignores extra boxes; the framework expects a metric that counts them for detection | Done (RESULTS.md Section 73): F1 95.0 / 92.3 / 89.6 for YOLOv8x / RT-DETR-l / Faster R-CNN; RT-DETR-l leaves about 1.9 extra boxes per X-ray |
| Per-class (per-FDI) top-1 | Shows where errors sit | Done (RESULTS.md Section 74, `benchmark/per_fdi.csv`): every detector beats position-only on all 32 numbers |
| AP / mAP | Standard detection ranking metric | Not reported (decided 2026-10-05). It averages over confidence thresholds, and the position-only model has no scores to compare with; numbered F1 covers the false positives mAP would add. |
| Risk-coverage (AURC), calibration | For review flags | Reported (Sections 65, 66) |
| Agreement between models (kappa, kappa max, phi, permutation nulls) | The shared-error claim | Reported (Sections 59, 65, 70) |

## Aggregation and uncertainty

- Pooled over the five test folds, so every X-ray is scored once by a
  model that did not train on it. Fold-level values are also reported.
- 95% CIs from 10,000 bootstrap draws of X-rays; paired for differences.
- Second split (seed 1) as replication (Section 70). Minimum detectable
  effects given in Section 66 D.

## To do before submission

Nothing open on metrics. Check the paraphrased questions against the
paper.
