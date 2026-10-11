# Metrics Reloaded check

Metric choices checked against the Metrics Reloaded framework (Maier-Hein
et al., Nature Methods 2024; open-access text PMC11182665, main text read
2026-10-09). What was checked: the object detection (ObD) path described
in the main text (localization criterion, assignment strategy, counting
metric, multi-threshold metric, per-class application). What was not: the
supplementary notes, the extended data figures (S8, S9) and the online
tool, so the fingerprint item numbers (FP) are the paper's, but each
mapping below is this study's reading and should be re-run in the tool
before submission.

**One departure from the framework, now closed in the supplement.** For
detectors that give continuous class scores (FP5.1), the paper recommends
complementing a counting metric with a multi-threshold metric, AP or the
FROC score, rather than relying on a single score cutoff. The main text
reports numbered F1 and top-1 at one cutoff (0.5), cutoff sweeps (0.25 to
0.75; Sections 62, 65) and risk-coverage and calibration for the review
flags (Sections 65, 66). AP50 and AP50:95 (class-aware, COCO matching) are
computed for every detector set and go in the supplement (RESULTS.md
Section 89, `AP_RULES.md`); FROC is not computed. The main text says that
top-1 stays the headline because the position-only baseline has no scores.

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
| Top-1 over all labeled teeth (missed = wrong) | Headline, directly comparable with the position-only model, which always answers. It is a per-reference-tooth recall (detected and correctly numbered over labeled), so it needs no true negatives, which the framework says rules out accuracy and specificity for detection | Reported with X-ray bootstrap CIs |
| Gap over position-only (pp, paired) | The study's question | Reported |
| Missed and misnumbered rates | Separates detection from numbering | Reported |
| Top-1 on matched teeth only | Numbering given detection | Reported |
| Macro top-1 (per-class mean) | Class balance check | Computed in `benchmark/score.py`; equals micro within 0.4 pp |
| False positives and numbered F1 | Top-1 ignores extra boxes; the framework expects a metric that counts them for detection | Done (RESULTS.md Section 73): F1 95.0 / 92.3 / 89.6 for YOLOv8x / RT-DETR-l / Faster R-CNN; RT-DETR-l leaves about 1.9 extra boxes per X-ray |
| Per-class (per-FDI) top-1 | Shows where errors sit | Done (RESULTS.md Section 74, `benchmark/per_fdi.csv`): every detector beats position-only on all 32 numbers |
| AP | The framework's multi-threshold metric for scored detectors (FP5.1) | **Supplement only (2026-10-10, RESULTS.md Section 89).** Headline stays top-1 because the position-only model has no scores to compare with. FROC not computed. |
| Risk-coverage (AURC), calibration | For review flags | Reported (Sections 65, 66) |
| Agreement between models (kappa, kappa max, phi, permutation nulls) | The shared-error claim | Reported (Sections 59, 65, 70) |

## Aggregation and uncertainty

- Pooled over the five test folds, so every X-ray is scored once by a
  model that did not train on it. Fold-level values are also reported.
- 95% CIs from 10,000 bootstrap draws of X-rays; paired for differences.
- Two further splits (seeds 1 and 2) as replications (Sections 75, 78). Minimum detectable
  effects given in Section 66 D.

## To do before submission

- Supplement file: `paper/miccai/SUPPLEMENT.md` (AP table, built 2026-10-10; add the DENTEX intervention results when they exist).
- Run the fingerprint through the Metrics Reloaded online tool and compare
  its recommended metrics with the table above (not done; the tool is
  interactive).
- Class-aware aggregation: the framework applies the metrics per class and
  then aggregates with extra care when classes have few objects. Here
  macro and micro top-1 differ by under 0.4 pp and per-FDI top-1 is in
  `benchmark/per_fdi.csv`, so this holds; keep the sentence in the paper.
