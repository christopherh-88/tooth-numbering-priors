# Results log

Every number produced in this project so far, with the script that
produced it, the date, and the split it came from. Each experiment also
has its own README with full method notes and interpretation - this file
is the flat index so nothing has to be reconstructed from memory or from
four different directories later. When a new number is produced, add a row
here rather than only updating the experiment's own README.

All entries below are from **2026-09-07** (a single working session)
unless a later dated entry says otherwise.

## 1. Environment sanity check (CPU, not a real result)

**Script:** `cpu_repro/train_unet_cpu.py` · **Split:** 24 images, 3 epochs,
256x256, random subset (seed 42) · **Date:** 2026-09-07

Proved the TensorFlow/U-Net pipeline runs end-to-end on CPU. Numbers below
are noise from 3 epochs on 24 images - **not comparable to anything, not a
baseline, do not cite these as a result.**

| metric | value |
|---|---|
| dice_coef (final epoch) | 0.0905 |
| val_dice_coef | 0.1016 |
| precision | 0.0786 |
| recall | 0.0029 |

## 2. Coordinate-only baseline (main)

**Script:** `cpu_repro/coord_baseline/build_coord_baseline.py` · **Split:**
image-level grouped 80/20, 5 seeds (0-4), pooled `Dataset/yolo_train_dataset/`
· **Date:** 2026-09-07 · **README:** `cpu_repro/coord_baseline/README.md`

Can (x_center, y_center, width, height, area, aspect_ratio) alone predict
FDI class? Mean ± 95% CI over 5 seeds:

| metric | logistic regression | gradient-boosted tree |
|---|---|---|
| top-1 accuracy (32-way) | 0.6706 ± 0.0095 | 0.6949 ± 0.0077 |
| quadrant accuracy | 0.9677 ± 0.0032 | 0.9653 ± 0.0035 |
| tooth-type accuracy | 0.6919 ± 0.0084 | 0.7200 ± 0.0084 |
| majority-class baseline | 0.0363 ± 0.0019 | 0.0363 ± 0.0019 |
| errors that are mirror-quadrant swaps | 0.0576 ± 0.0031 | 0.0748 ± 0.0048 |
| errors that are neighbor swaps | 0.8079 ± 0.0159 | 0.8287 ± 0.0148 |

**Seed-0 point values** (used as the "C" reference in the go/no-go rule,
Section 9 below - do not substitute the 5-seed mean above for this
purpose):

| metric | logistic regression | gradient-boosted tree |
|---|---|---|
| top-1 accuracy | 0.663814 | **0.692770** |
| quadrant accuracy | 0.967765 | 0.966126 |
| tooth-type accuracy | 0.685121 | 0.718630 |
| majority-class baseline | 0.035331 | 0.035331 |
| n_test (instances) | 5491 | 5491 |

Full data: `cpu_repro/coord_baseline/summary.csv`,
`per_seed_results.csv`, `confusion_matrix_*_seed0.csv/.png`.

## 3. Coordinate baseline - tooth-type and teeth-count-in-crop breakdown

**Script:** `cpu_repro/coord_baseline/error_breakdown.py` · **Split:** same
as Section 2 (5 seeds, pooled) · **Date:** 2026-09-07 · **README:**
`cpu_repro/coord_baseline/README.md`

Accuracy by tooth type (mean ± 95% CI over 5 seeds):

| tooth group | gradient-boosted tree | logistic regression | n instances |
|---|---|---|---|
| Molar | 0.7860 ± 0.0137 | 0.7928 ± 0.0176 | 9252 |
| Premolar | 0.6650 ± 0.0229 | 0.6109 ± 0.0398 | 6973 |
| Canine | 0.6515 ± 0.0310 | 0.6332 ± 0.0546 | 3827 |
| Incisor | 0.6327 ± 0.0153 | 0.5948 ± 0.0131 | 7517 |

Accuracy by teeth annotated in the crop (mean ± 95% CI):

| teeth in crop | gradient-boosted tree | logistic regression | n pooled |
|---|---|---|---|
| 1-15 | 0.4898 ± 0.0889 | 0.4323 ± 0.1159 | 653 |
| 16-23 | 0.5729 ± 0.0675 | 0.5438 ± 0.0496 | 2550 |
| 24-28 | 0.6591 ± 0.0255 | 0.6358 ± 0.0406 | 5151 |
| 29-32 | 0.7283 ± 0.0128 | 0.7033 ± 0.0043 | 18843 |
| 33+ (annotation artifact, n small) | 0.6109 ± 0.0336 | 0.6954 ± 0.0323 | 372 |

Pearson r (teeth-in-crop count vs. per-instance correctness): 0.1211 (GBT),
0.1255 (LogReg).

Full data: `accuracy_by_tooth_type.csv`, `accuracy_by_fdi_class.csv`,
`accuracy_by_teeth_count.csv`, `teeth_count_correlation.csv`,
`pooled_predictions.csv`.

## 4. Coordinate baseline - controls (shuffle / position-only / size-only)

**Script:** `cpu_repro/coord_baseline/controls/run_controls.py` ·
**Split:** same image-level split as Section 2, same 5 seeds, identical
train/test membership across all four conditions · **Date:** 2026-09-07 ·
**README:** `cpu_repro/coord_baseline/controls/README.md`

| condition | classifier | top-1 acc | quadrant acc | tooth-type acc |
|---|---|---|---|---|
| full (real) | GBT | 0.6949 ± 0.0077 | 0.9653 ± 0.0035 | 0.7200 ± 0.0084 |
| full (real) | LogReg | 0.6706 ± 0.0095 | 0.9677 ± 0.0032 | 0.6919 ± 0.0084 |
| full (shuffled per image) | GBT | 0.0324 ± 0.0033 | 0.2461 ± 0.0034 | 0.1273 ± 0.0041 |
| full (shuffled per image) | LogReg | 0.0360 ± 0.0029 | 0.2549 ± 0.0098 | 0.1363 ± 0.0048 |
| position only (x,y) | GBT | 0.6196 ± 0.0058 | 0.9625 ± 0.0025 | 0.6473 ± 0.0078 |
| position only (x,y) | LogReg | 0.6419 ± 0.0134 | 0.9668 ± 0.0036 | 0.6668 ± 0.0133 |
| size only (w,h) | GBT | 0.1473 ± 0.0056 | 0.3476 ± 0.0063 | 0.3652 ± 0.0050 |
| size only (w,h) | LogReg | 0.1568 ± 0.0072 | 0.3496 ± 0.0114 | 0.3811 ± 0.0107 |

Majority-class baseline (all conditions): 0.0363 ± 0.0019.

Full data: `summary.csv`, `per_seed_results.csv`, `controls_comparison.png`.

## 5. UFBA-425 annotation anomaly scan

**Script:** `cpu_repro/anomaly_scan/scan_annotations.py` · **Split:** N/A
(annotation QA, not a train/test split) - all 1021 label files in
`Dataset/yolo_train_dataset/*/labels/` · **Date:** 2026-09-07 ·
**README:** `cpu_repro/anomaly_scan/README.md`

| category | count | of 1021 |
|---|---|---|
| any nonzero anomaly score | 692 | 67.8% |
| primary reason: missing_codes only (expected/low-weight) | 586 | 57.4% |
| primary reason: sparse_annotation (<8 teeth) | 41 | 4.0% |
| primary reason: single_axis_position | 35 | 3.4% |
| primary reason: duplicate_codes | 20 | 2.0% |
| primary reason: wrong_quadrant (both axes) | 10 | 1.0% |

Full data: `full_scan.csv` (all 1021, ranked), `candidates.csv` (top 100),
`reference_quadrant_stats.csv` (convention reference table, see Section 8).

## 6. Children's Dental Panoramic Radiographs dataset (Figshare 6317013) - ruled out

**Investigation, not a script run** · **Date:** 2026-09-07 · **Findings:**
`cpu_repro/anomaly_scan/children_dataset_findings.md`

No numeric table - the result is negative. Downloaded and inspected all
three sub-datasets directly (pediatric disease detection, 100 images;
children's caries segmentation, ~93+ images; adult tooth segmentation,
1417 images). **None annotate per-tooth FDI identity** - disease-category
labels or a single undifferentiated "tooth" class only, no quadrant field
of any kind. Not usable for the anomaly scan, the convention check, or as
an EXTEND dataset for the go/no-go rule (Section 9). Ruled out, not
pending.

## 7. DENTEX (arXiv 2305.19112, MICCAI 2023) - schema and anomaly scan

**Scripts:** `cpu_repro/anomaly_scan/dentex_raw/build_table.py`,
`cpu_repro/anomaly_scan/scan_dentex.py` · **Split:** N/A (annotation QA) -
1358 images across the `quadrant_enumeration` (634) and
`quadrant_enumeration_disease` train+validation (705+50) tiers, the two
DENTEX tiers that carry a full FDI code · **Date:** 2026-09-07 ·
**Findings:** `cpu_repro/anomaly_scan/dentex_findings.md`

| item | count |
|---|---|
| teeth with a full FDI code (public train+validation) | 21,806 |
| images with a full FDI code | 1,358 |
| impacted-tooth annotations (train+validation) | 644 (604 train, 40 val) - all carry a full FDI code |
| images with a duplicate FDI code | 81 |
| images with any position-vs-quadrant violation | 27 |
| ...of which wrong on both axes (opposite-quadrant) | 1 image, 2 boxes |
| images with duplicate code AND a position violation | 7 |
| **images with duplicate and/or position anomaly (the usable candidate pool)** | **101 / 1358 (7.4%)** |

Bottom line: yes, DENTEX contains real position/identity dissociation cases
with FDI labels attached (see `dentex_findings.md` for three worked
examples). Full data: `dentex_results/full_scan_dentex.csv` (all 1358),
`candidates_dentex.csv` (top 100).

## 8. Coordinate-convention verification

**Not a numeric experiment - a documented check**, run before combining
any two datasets' coordinates. Protocol and full case log:
`cpu_repro/CONVENTIONS.md`.

| dataset | quadrant convention | verdict |
|---|---|---|
| UFBA-425 | quadrants 1,4→image-left; 2,3→image-right; 1,2→upper; 3,4→lower | reference (established here) |
| Children's Dental Panoramic Radiographs | N/A | no FDI field to check |
| DENTEX | same as UFBA-425 | **checked, matches** |

## 9. YOLOv8 training - prepared, not yet run

**Script:** `cpu_repro/yolo_training/train_yolo.py` · **Status:** ready to
run on GPU quota reset, not executed · **Date prepared:** 2026-09-07 ·
**README:** `cpu_repro/yolo_training/README.md` · **Pre-registered decision
rule:** `cpu_repro/yolo_training/GO_NO_GO.md`

No results yet. When this run completes, add its row here in the same
format as Section 2 (it's evaluated with the identical
`build_coord_baseline.evaluate()` function specifically so the two are
directly comparable) - do not just update `eval_results/summary.csv` and
consider it logged.

Pre-registered thresholds (from `GO_NO_GO.md`, frozen before any YOLO
output is seen): C = 0.6928 (GBT, seed-0 split, Section 2 table above).
COMMIT if (Y-C) ≤ 5 points, PIVOT if ≥ 15 points, EXTEND if in between;
minimum detection recall 90% or the comparison must be repeated penalizing
missed detections.

## 10. Coordinate-only baseline - DENTEX replication + stratified analysis

**Scripts:** `cpu_repro/coord_baseline/dentex/build_coord_baseline_dentex.py`,
`controls/run_controls_dentex.py`, `stratified_analysis.py` · **Split:**
own image-level grouped 80/20 split over DENTEX's 1358 FDI-coded images
(seed 0: 1087 train / 271 test images), same method/functions as Section 2
but a separate split object (DENTEX has no augmentation-crop duplication,
so the leakage guard that mattered for UFBA-425 doesn't apply here) ·
**Date:** 2026-09-07 · **README:** `cpu_repro/coord_baseline/dentex/README.md`

Main baseline (mean +/- 95% CI, 5 seeds):

| metric | logistic regression | gradient-boosted tree |
|---|---|---|
| top-1 accuracy (32-way) | 0.6129 +/- 0.0180 | 0.6847 +/- 0.0124 |
| quadrant accuracy | 0.9812 +/- 0.0016 | 0.9796 +/- 0.0018 |
| tooth-type accuracy | 0.6271 +/- 0.0194 | 0.7010 +/- 0.0127 |
| majority-class baseline | 0.0375 +/- 0.0009 | 0.0375 +/- 0.0009 |

Within ~1 point of UFBA-425's own numbers (Section 2: 0.6949 GBT top-1,
0.9653 quadrant) - the coordinate-identity relationship replicates closely
on an independent dataset.

Controls (same four conditions as Section 2's controls, GBT top-1 shown):
full (real) 0.6847, full (shuffled per image) 0.0365, position only (x,y)
0.6174, size only (w,h) 0.1336 - same pattern as UFBA-425: shuffling
collapses to the majority baseline, position alone recovers almost all of
the signal.

Stratified comparison (same trained models, sliced by stratum; both strata
cleared the pre-set size thresholds, `too_small=False` for every row):

| stratum | GBT top-1 | drop vs. canonical | pooled n |
|---|---|---|---|
| canonical | 0.6827 +/- 0.0146 | - | 20,229 |
| dissociation (101 flagged images, 1291 test instances) | 0.6195 +/- 0.0531 | **-6.3 points** | 1,291 |
| impacted (644 Impacted-diagnosis instances) | 0.9144 +/- 0.0415 | **+23.2 points** | 614 |

**Correction (2026-09-08):** the "+23.2 points, impacted teeth are easier"
framing originally recorded here overstated the effect and did not lead
with the composition fact that determines how to read it. Corrected below;
see `cpu_repro/coord_baseline/dentex/composition_check.py` and the
"Composition check" section of `cpu_repro/coord_baseline/dentex/README.md`
for the full analysis. The dissociation row above is unaffected by this
correction (its drop survives composition control - see below).

**Composition fact, stated first because it determines how every
impacted-stratum number should be read:** DENTEX's "Impacted" diagnosis
label applies exclusively to third molars in this dataset - 644/644
impacted-diagnosis instances (100%) carry FDI code 18, 28, 38, or 48. The
impacted stratum is not a general anomaly sample; it is a third-molar
sample. Third molars were already the easiest tooth-type group to place
from geometry alone (Section 2's tooth-type breakdown), so the raw
comparison above (impacted, 100% third molars, vs. canonical, 8.7% third
molars) is partly comparing different tooth-type mixes, not just
"impacted vs. not."

Composition-controlled comparison (GBT, pooled over 5 seeds, same trained
models and test-fold predictions as above):

| stratum | n | third-molar share | top-1 acc (third molars only, matched subset) | raw gap vs. canonical | composition-controlled residual |
|---|---|---|---|---|---|
| canonical | 19,885 (1,769 third molars) | 8.7% | 0.7354 | - | - |
| dissociation | 1,354 (192 third molars) | 14.2% | 0.8214 | -6.1pp | **-6.4pp** |
| impacted | 644 (644 third molars) | **100.0%** | **0.9121** | +22.9pp | **+17.1pp** |

("Composition-controlled residual" = actual stratum accuracy minus what
canonical's accuracy would be if reweighted to the stratum's own FDI-code
mix - the part of the gap tooth-type mix does not explain.)

**Corrected reading:**

- **Dissociation drop is not a composition artifact.** Controlling for
  tooth-type mix does not shrink it (-6.1pp raw vs. -6.4pp controlled);
  excluding third molars entirely widens it further to -8.5pp. The
  original dissociation finding stands as reported.
- **Impacted "23pp easier" claim is corrected.** Restricting to a matched
  third-molars-only subset (the cleanest apples-to-apples control):
  impacted third molars score 0.9121 vs. canonical third molars at 0.7354
  - a **17.7pp gap that survives composition control**. Of the original
  raw 22.9pp gap, about 5pp is attributable to the stratum being entirely
  third molars (already the easiest FDI codes geometrically); the
  remaining ~17.1pp is a residual specific to impacted third molars vs.
  erupted third molars. **Corrected statement: impacted third molars are
  about 17.7 points more geometrically predictable than erupted third
  molars (0.912 vs. 0.735), with a further ~5 points of the original raw
  gap attributable to the stratum being entirely third molars to begin
  with.** Two hypotheses for the 17.7pp residual - (1) impacted/unerupted
  teeth may get more consistent bounding boxes from annotators, (2)
  impacted third molars may sit in more extreme or less-crowded positions
  than erupted ones - are stated as open, untested hypotheses, not
  conclusions; this dataset does not distinguish between them.

## Adding a new entry

Append a new numbered section, not an edit to an existing one. Include the
script path, the exact split/seed, the date, and the actual numbers (read
from the CSV, not retyped from memory or a chat transcript) - the same
discipline as `CONVENTIONS.md` and `GO_NO_GO.md`.
