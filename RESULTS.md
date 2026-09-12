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

**Falsification threshold:** we treat top-1 accuracy within 2x of the
majority-class baseline (~7.3%, given the ~3.63% majority baseline on
UFBA-425) as the region that would falsify the claim that geometry
meaningfully predicts tooth identity; this threshold is stated here
explicitly for clarity but was not pre-registered before the initial
experiment was run.

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

**Update (2026-09-11): run completed - see Section 21 for the result and
Section 22 for the Task 2 error-pattern comparison.** This section is
left as-is above (historical record of the pre-run plan) rather than
rewritten, per this file's own append-only convention.

Pre-registered thresholds (from `GO_NO_GO.md`, frozen before any YOLO
output is seen): C = 0.6928 (GBT, seed-0 split, Section 2 table above).
COMMIT if (Y-C) ≤ 5 points, PIVOT if ≥ 15 points, EXTEND if in between;
minimum detection recall 90% or the comparison must be repeated penalizing
missed detections.

## 10. Coordinate-only baseline - DENTEX replication + stratified analysis

**Scripts:** `cpu_repro/coord_baseline/dentex/build_coord_baseline_dentex.py`,
`controls/run_controls_dentex.py`, `stratified_analysis.py`,
`composition_check.py`, `dissociation_mixed_effects.py`,
`dissociation_loio.py` · **Split:**
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
for the full analysis. The dissociation row above is addressed separately:
composition does not explain its drop away, but the size of the effect is
sensitive to estimation method and the raw **-6.3 points** figure in the
row above overstates it - see the full sensitivity analysis in "Corrected
reading" below. The best-supported estimate is roughly **-3.6 to -3.9
points**, not significant at conventional thresholds.

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

- **Dissociation drop is not a composition artifact, but its size is
  estimation-method-sensitive, and the more careful estimate is well under
  half of what was first reported.** Composition does not explain it away
  (composition-controlled table above), but that comparison, the naive
  instance-pooled comparison, and the 5-seed paired mean are all simpler
  estimators that do not account for image-level clustering - and
  accounting for it changes the number substantially.

  **Primary estimate (image + seed crossed random-effects model, full
  1,291 dissociation + 20,229 canonical instances, GBT, `correct ~
  is_dissociation + (1 | image_id) + (1 | seed)`):** stratum coefficient
  **-3.58pp**, 95% CI **[-7.87pp, +0.71pp]**, p = 0.102. Adding
  log(teeth-in-image) as a covariate barely moves this - **-3.91pp**, 95%
  CI [-8.22pp, +0.40pp], p = 0.076 - and teeth-count itself is not a
  significant predictor (coefficient -0.96pp per unit log-teeth, p =
  0.164, wrong direction to explain the gap, closes only ~12% of it), so
  teeth-count/crowding is ruled out as the explanation. **Neither version
  reaches significance at conventional thresholds.**

  **Why this differs from the previously reported -6.32pp:** that number
  came from the paired 5-seed mean (mean diff -6.32pp, 95% CI [-12.67pp,
  +0.03pp], p = 0.0506) and closely matches the naive instance-pooled
  comparison (-6.07pp) - neither accounts for image-level clustering. A
  leave-one-image-out check across all 70 dissociation images shows the
  naive estimate is **not driven by any single image** (excluding any one
  image individually shifts it by at most 1.2pp; 0/70 images shift it by
  more than 2pp). But it **is concentrated in a small cluster of large,
  low-accuracy images**: the 5 lowest-per-image-accuracy dissociation
  images - `validation_triple_36` (2 instances, 0% acc, 2 teeth in the
  whole image), `train_quadrant_enumeration_disease_512` (2 instances, 0%
  acc, only **1** tooth in the whole image), `train_quadrant_enumeration_
  disease_182` (8 instances, 12.5% acc), `train_quadrant_enumeration_
  disease_63` (12 instances, 25% acc), `train_quadrant_enumeration_279`
  (31 instances, 25.8% acc) - just 7% of the 70 images (4.3% of
  instances) - account for about **78% of the gap** between the naive
  pooled estimate and the crossed-model estimate: removing just those 5
  moves the naive number from -6.07pp to -4.27pp. Removing the worst 10
  images (14% of images, 22% of instances, since several are large crops)
  overshoots past zero to +1.39pp. **This is the identified mechanism**:
  instance-pooling implicitly gives images with more annotated teeth
  proportionally more influence than image-level weighting does, and a
  handful of large dissociation images happen to be both low-accuracy and
  instance-heavy; two of the five worst images have only 1-2 teeth in the
  entire image, too little to trust as an individual accuracy point at
  all.

  **Treat the crossed-model estimate (roughly -3.6pp to -3.9pp, not
  significant) as the number to cite, not -6.32pp/-6.07pp.** If this
  result is referenced anywhere, report it with this full chain - a
  single point estimate without this context is not an accurate summary
  of what was measured.
  **Still-untested hypothesis, not a fact:** the dissociation flag is
  image-level, not tooth-level, so even a well-estimated stratum-average
  understates the effect on the specific anomalous teeth within a flagged
  image, diluted by the mostly well-behaved teeth in the same image. This
  dilution direction is plausible but unquantified and does not offset the
  estimation-sensitivity finding above.
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

## 11. Framing decision (2026-09-08)

This section records a framing decision reached in discussion but never
previously written into any file - it did not exist in committed form
before this entry. It supersedes any earlier informal framing ("engineered
priors must fail on anomalies," "implicit shortcut learning") to the
extent that framing appears elsewhere in this repo or in chat.

### 11.1 Primary framing: diagnostic tool, not a claim about detector internals

The paper's primary framing is: **the coordinate-only baseline is a
diagnostic/audit tool** - a cheap sanity-check ablation that should be run
before crediting any tooth-numbering model's reported accuracy to
appearance understanding. If a coordinate-only model trained on nothing
but box geometry gets within a few points of a full image-based detector,
that is a signal the dataset's label space may be substantially
recoverable from position alone, and any accuracy claims for the
image-based model should be read with that in mind.

This is a **methods contribution** (a check anyone can run on their own
tooth-numbering dataset/model), not a claim about what any specific
trained detector's weights are actually doing internally.

### 11.2 What is and is not supported: claim A vs. claim B

Two distinct claims must be kept separate, because the evidence in this
repo supports only one of them:

- **Claim A: FDI tooth identity correlates strongly with bounding-box
  geometry alone**, independent of image appearance. **Well-supported.**
  Cross-dataset replicated (UFBA-425 and DENTEX, Sections 2 and 10, within
  ~1 point of each other), with clean controls (Section 4 and the DENTEX
  controls in Section 10 - shuffling collapses to majority-baseline,
  position-only recovers nearly all of the signal, size-only recovers
  little).
- **Claim B: trained image-based detectors (OralBBNet, HierarchicalDet-style
  architectures) exploit this correlation as an implicit shortcut instead
  of, or in addition to, learning genuine appearance features.** **Not
  directly measured anywhere in this repo.** No real trained detector's
  predictions, confusion matrix, or internal representations have been
  examined here (see Section 11.5, item 1, for the check that would begin
  to close this gap).

**This repo provides strong evidence for A and no direct evidence for B.**
Claiming B on the basis of the results recorded here would overstate what
was actually measured. Any writeup drawing on this repo should state A as
a finding and B as a motivated, falsifiable hypothesis this data does not
test - not as a conclusion.

### 11.3 Foreground quadrant vs. tooth-type - don't bury this distinction

Quadrant accuracy (96.5% UFBA-425, 98.0% DENTEX GBT) is **close to
definitional**: FDI's first digit literally encodes quadrant, which is
itself a spatial concept (which side of which jaw), so a geometry-only
model recovering it at near-ceiling accuracy is close to restating what
the labeling scheme already means, not a surprising empirical finding.

The actual finding is narrower and less trivial: **tooth-type accuracy -
the second digit, which specific tooth within a quadrant - is also ~70-72%
recoverable from geometry alone** (Section 2: 72.0% GBT; Section 10: 70.1%
GBT), with **no definitional reason this should be true**. Nothing about
FDI numbering requires that the 3rd vs. 4th vs. 5th tooth in a quadrant
occupy sufficiently distinct, consistent pixel-space slots for box
geometry alone to tell them apart most of the time - that they do is an
empirical property of panoramic radiograph acquisition and jaw anatomy,
not a restatement of the label scheme. This distinction should be
foregrounded in any writeup, not left as a footnote to the quadrant
number.

### 11.4 Generalization boundary, stated as a falsifiable hypothesis

Rather than a vague disclaimer ("this may not generalize"), the boundary
condition is stated here as a **testable, falsifiable hypothesis**: the
coordinate-only shortcut should appear wherever both of the following
hold, and should vanish where either breaks -

1. a small, structured label space with a **near-deterministic spatial
   slot per class** (FDI numbering: 32 codes, each with an anatomically
   consistent quadrant-and-position meaning), and
2. a **canonicalized acquisition protocol** that pins that slot to a
   consistent pixel-space location across the dataset (panoramic
   radiographs: fixed patient positioning, fixed sensor geometry, minimal
   rotation/zoom variation).

**Tested so far: FDI numbering, panoramic radiographs only** (UFBA-425,
DENTEX). No claim is made here about non-panoramic imaging (bitewing,
periapical, CBCT), or about landmark detection tasks outside dentistry -
those remain untested next steps for the hypothesis, not yet-covered
special cases of an established general result.

**Universal/Palmer relabeling check (2026-09-08):** run and resolved - see
below. It does **not** test either precondition above, and should not be
read as generalization evidence; it confirms an expected invariance, not a
new data point on the hypothesis.

FDI, Universal, and Palmer numbering are three notations for the
**identical** 32-way partition of physical tooth positions - not three
different label spaces. Universal and Palmer were built as relabelings of
`FDI_CODES` and verified as clean bijections before use (32 unique targets
each, no merged or split classes; e.g. FDI 11 -> Universal 8 -> Palmer
UR1, FDI 48 -> Universal 32 -> Palmer LR8). Because the relabeling changes
neither the acquisition protocol (precondition 2) nor which physical
positions get grouped into which class (precondition 1 is about the label
space's structure, unchanged by renaming it), this check could only ever
demonstrate label-invariance, not test the hypothesis's boundary.

5-seed mean +/- 95% CI, same splits/seeds as the existing FDI results:

| dataset | convention | GBT top-1 | LogReg top-1 |
|---|---|---|---|
| UFBA-425 | FDI (existing, Section 2) | 0.6949 +/- 0.0077 | 0.6706 +/- 0.0095 |
| UFBA-425 | Universal | 0.6954 +/- 0.0139 | 0.6707 +/- 0.0097 |
| UFBA-425 | Palmer | 0.6954 +/- 0.0140 | 0.6707 +/- 0.0097 |
| DENTEX | FDI (existing, Section 10) | 0.6847 +/- 0.0124 | 0.6129 +/- 0.0180 |
| DENTEX | Universal | 0.6818 +/- 0.0131 | 0.6130 +/- 0.0180 |
| DENTEX | Palmer | 0.6811 +/- 0.0123 | 0.6130 +/- 0.0180 |

**LogReg is exactly label-invariant** (as expected for a linear
multinomial classifier that treats class labels as nominal): means agree
to within 0.01pp across all three conventions on both datasets. **GBT
shows small differences (0.05-0.36pp)** that are not exactly invariant -
traced to `HistGradientBoostingClassifier`'s internal implementation
(per-class boosting order, early-stopping validation split) being
sensitive to label identity, not to the tooth-numbering problem itself -
but these differences are 3-20x smaller than the seed-to-seed noise (CI
half-widths of 0.77-1.8pp) and are not distinguishable from it.
Verification script: `cpu_repro/coord_baseline/numbering_convention_check.py`.

### 11.5 Open items

- ~~**Error-pattern correlation check.**~~ **Resolved 2026-09-11/12** -
  see Sections 22 (error-type taxonomy comparison), 23 (case study), 26
  (2x2 agreement table and phi coefficient), and 33 (5-seed
  replication of the phi result). Answer: the real detector does not
  show shortcut-reliant error patterns (gap concentrated in tooth-type
  accuracy, not explainable by a position-only signal), though a weak
  but reliably nonzero error correlation with the coordinate-only model
  exists (phi = 0.16-0.24 depending on seed).
- ~~**Universal/Palmer relabeling check.**~~ **Resolved 2026-09-08** - see
  Section 11.4. Confirmed label-invariant (not a generalization data
  point; see that section for why).
- ~~**Dissociation-stratum precision.**~~ **Resolved 2026-09-08** - see
  Section 10's "Corrected reading." Image + seed crossed random-effects
  model is now the reported estimate (-3.6pp to -3.9pp, not significant),
  replacing the paired-seed -6.32pp figure. A follow-on leave-one-image-out
  check explained why the simpler estimators overstated the effect (a
  small cluster of large, low-accuracy images, not a single outlier and
  not teeth-count/crowding).
- ~~**Boundary-condition dataset (not yet chosen/run).**~~ **Resolved
  2026-09-12** - see Section 25. Candidate 1 below (periapical
  radiographs) was the one actually run (DenPAR); candidates 2/3 remain
  untried and are legitimate future extensions, not required to close
  this item:
  1. **Bitewing/periapical radiographs** (breaks precondition 2 -
     acquisition is per-tooth/angled, not a fixed whole-jaw layout;
     same clinical domain and FDI-adjacent labeling as the existing work,
     no volumetric complexity). **Run - Section 25.**
  2. **CBCT slices** (breaks precondition 2 more severely, but adds
     volumetric/3D framing that complicates the box-geometry setup and
     may distract from the core comparison). Not run.
  3. **A structurally different label space on the same panoramic
     modality**, e.g. a landmark-detection task (breaks precondition 1
     instead of 2, isolating that variable rather than conflating both).
     Not run.
- ~~**Mitigation experiment (not yet run).**~~ **Resolved 2026-09-12** -
  see Sections 30 (pre-commitments) and 31 (result: a clean null,
  removing the position/scale jitter produced no detectable accuracy
  change). `cpu_repro/coord_baseline/mitigation/README.md`'s status
  header and falsification-threshold section were updated to match.

## 12. Coordinate-only baseline - paired significance tests

**Script:** `cpu_repro/coord_baseline/significance_tests.py` · **Split:**
same 5 seeds (0-4) as Section 2 · **Date:** 2026-09-08

Supplements the 5-seed mean ± 95% CI already reported in Sections 2 and 4
with paired (by seed) permutation tests, rather than relying on CI
overlap alone. Two comparisons, both using exact sign-flip enumeration
over the 5 seeds (2^5 = 32 arrangements):

| comparison | classifier | mean diff | paired permutation p |
|---|---|---|---|
| real vs. shuffled-per-image control (top1_acc) | logistic regression | 0.6347 | 0.0625 |
| real vs. shuffled-per-image control (top1_acc) | gradient-boosted tree | 0.6625 | 0.0625 |
| top1_acc vs. majority-class baseline | logistic regression | 0.6344 | 0.0625 |
| top1_acc vs. majority-class baseline | gradient-boosted tree | 0.6586 | 0.0625 |

**p = 0.0625 is the floor of this test's resolution at n=5 seeds, not a
borderline result.** With exact sign-flip enumeration over 5 seeds, the
smallest achievable two-sided p-value is 2/32 = 0.0625, and all four
comparisons hit exactly that floor because every one of the 5 seeds
agreed in direction (real > shuffled, real > majority baseline, with no
exceptions). This is the strongest significance this specific test can
express at this sample size - it should not be read as "marginal" or
"just below the conventional 0.05 threshold" in any writeup.

## 13. Natural positional variance by FDI class

**Script:** `cpu_repro/coord_baseline/measure_positional_variance.py` ·
**Data:** all 27,563 pooled instances (`Dataset/yolo_train_dataset`,
same loader as Section 2) · **Date:** 2026-09-08

Measures the per-class spread of (x_center, y_center) already naturally
present across images, to size the geometry-jitter magnitude for the
mitigation experiment (`cpu_repro/coord_baseline/mitigation/README.md`)
rather than picking an arbitrary value. Descriptive statistics only, no
model fitting.

| | x (normalized) | y (normalized) |
|---|---|---|
| mean per-class std | 0.0267 | 0.0490 |
| median per-class std | 0.0248 | 0.0481 |
| mean per-class IQR | 0.0306 | 0.0628 |

Full per-class breakdown: `cpu_repro/coord_baseline/positional_variance_by_class.csv`.

**Resulting jitter magnitude for the mitigation experiment:** at least
0.0535 in x and 0.0979 in y (normalized units, 2x mean per-class std) -
a starting point to be revisited once the mitigation experiment actually
runs, not a tuned final value.

## 14. Task 2 pre-registered detectability check (before GPU training)

**Script:** `cpu_repro/coord_baseline/task2_power_check.py` · **Date:**
2026-09-08 · **Status:** a decision check run before training any
detector, not a measured result - no real detector exists yet.

Question: given how few "wrong" predictions a real, reasonably-accurate
detector will actually produce on this test set (n=5491), would Task 2
(comparing the coordinate-only model's neighbor-error-fraction, 0.8287
GBT, against a real detector's) even be statistically well-powered, or
would it risk comparing noise to noise? One-sample proportion test,
normal approximation, alpha=0.05 two-sided, power=0.80, swept across a
plausible detector-accuracy range (not a prediction of the real
detector's actual accuracy, which is unknown until it's trained):

| detector accuracy | n_wrong (of 5491) | minimum detectable difference |
|---|---|---|
| 75% | 1373 | 2.8 pp |
| 80% | 1098 | 3.2 pp |
| 85% | 824 | 3.7 pp |
| 90% | 549 | 4.5 pp |
| 95% | 275 | 6.4 pp |

**Conclusion: Task 2 is well-powered across the full plausible accuracy
range.** Even at a conservative 95% detector accuracy (only ~275 wrong
predictions to work with), a difference as small as ~6.4 percentage
points from the coordinate-only model's neighbor-error-fraction would be
detectable - and a detector relying on genuinely different cues than
geometry would plausibly produce a far larger gap than that. This
de-risks proceeding with Task 2 as planned; it does not need to be
redesigned or abandoned for lack of statistical power.

**Caveat, stated here rather than glossed over:** this uses a normal
approximation and does not model clustering by image (multiple teeth per
image are not independent draws) - the same issue Section 10's
image+seed random-effects model was built to correct for. Once real
detector output exists, the actual Task 2 analysis should use a
clustered/robust variance estimate, not this simplified power check -
this section only establishes that the comparison is worth running, not
how to run it.

## 15. Geometric ceiling check - is GBT leaving headroom unexploited?

**Script:** `cpu_repro/coord_baseline/geometric_ceiling_check.py` ·
**Split:** identical seed-0 grouped split as Section 2, same 6 features ·
**Date:** 2026-09-08

Question: is the reported 69.3% GBT top-1 accuracy a property of the
underlying geometric structure, or an artifact of that specific
classifier under-fitting real signal a less parametric method would
recover? A k-NN classifier with a large neighborhood approximates the
nonparametric (Bayes-error-adjacent) ceiling without assuming a linear
(logistic regression) or fixed-tree-depth (GBT) decision boundary.

| k | top1_acc | quadrant_acc | tooth_type_acc |
|---|---|---|---|
| 5 | 0.6361 | 0.9556 | 0.6673 |
| 15 | 0.6560 | 0.9608 | 0.6820 |
| 25 | 0.6551 | 0.9616 | 0.6809 |
| 50 | 0.6507 | 0.9594 | 0.6786 |
| 100 | 0.6358 | 0.9590 | 0.6625 |

Best k-NN top1_acc (k=15): 0.6560, vs. GBT 0.6928 (seed-0 point value,
Section 2) - **k-NN does not exceed GBT at any tested k; GBT is 3.7pp
above the best k-NN result.**

**Reading this correctly:** this does NOT mean 69.3% is a rigorously
proven Bayes-error ceiling - k-NN with Euclidean distance on standardized
features doesn't automatically adapt to the differing importance of the
6 features the way GBT's axis-aligned tree splits do, so k-NN
underperforming isn't formal proof nothing is left on the table. What it
does support: **a standard nonparametric method finds no evidence of
headroom GBT is leaving unexploited** - the accuracy number looks like a
property of genuine class overlap in geometric space, not an artifact of
an under-fit classifier that a fancier model would meaningfully beat.
This is worth stating in any writeup as a checked, moderate claim, not
overstated as "we found the exact ceiling."

## 16. Dual-Labeled Dataset (Zhou et al., BMC Oral Health 2024) - supernumerary and overlap verification

**Scripts:** `cpu_repro/dual_labeled_dataset/inspect_labels.py`,
`cpu_repro/dual_labeled_dataset/phash_overlap.py` · **Source:** Kaggle
`zwbzwb12341234/a-dual-labeled-dataset`, license "unknown" per its own
metadata (re-verified live) · **Date:** 2026-09-08 · **Full findings and
raw run output:** `cpu_repro/dual_labeled_dataset/findings.md`

**Partial-release status, exact quote from the dataset's own metadata**
(re-fetched live, not from memory): "Our dataset comprises a total of
2,000 panoramic radiographs... Currently, 500 panoramic images and their
label files have been uploaded. For the remaining data, please contact
the author..." Only this 500-image tranche was downloaded/analyzed.

| item | count |
|---|---|
| total label files in download | 2,066 |
| label files with a paired image (500-image tranche) | 500 |
| label "91" (supernumerary) total instances | 53 |
| distinct label files containing label 91 | 49 |
| of those, with a paired image (usable) | 23 files, 24 instances |
| filename-stem overlap with UFBA-425 | 0 |
| perceptual-hash near-duplicate matches with UFBA-425 (Hamming <=8, hash_size=16) | 0 |

**Usable supernumerary evidence in this download: 24 instances across 23
images** - the real number this tranche supports, not a larger figure
extrapolated from the dataset's stated total. No overlap detected with
UFBA-425 in the downloaded tranche (supports treating them as
independent datasets, does not rule out overlap in the undownloaded
remainder).

**Scope of what this supports:** existence-proof only, not a measured
result. It confirms real, annotated supernumerary-tooth cases exist and
are usable, which is relevant substrate for the clinical-stakes argument
(`paper/DRAFT.md`) - it does **not** show the coordinate-only model (or
any detector) is actually worse on these instances than on typical ones.
See Section 17 for that follow-up.

## 17. Does the coordinate-only model perform worse near supernumerary teeth?

**Script:** `cpu_repro/dual_labeled_dataset/supernumerary_error_check.py`
· **Date:** 2026-09-08 · **Design:** GBT trained on 100% of pooled
UFBA-425 (no held-out split - evaluation is entirely on the separate
Dual-Labeled Dataset, so no leakage concern), evaluated on that dataset's
own standard-FDI-labeled teeth (13,168 instances, 500 paired-image
label files), split into the 23 images containing a supernumerary tooth
(Section 16) vs. the other 477.

**Coordinate-convention check (run first, per `cpu_repro/CONVENTIONS.md`
protocol - required before any cross-dataset comparison):** both
datasets place quadrant-1 classes at a lower mean x_center than
quadrant-2 classes, matching UFBA-425's established convention (UFBA-425
reference: 0.3559 vs. 0.6646; Dual-Labeled Dataset: 0.4003 vs. 0.6053).
**Passed - safe to proceed.**

**Bonus finding: cross-dataset transfer of Claim A.** Overall accuracy on
this third, independent dataset: **32.9%** (13,168 instances) vs. a
majority baseline of **3.76%** - about 8.8x baseline. Real signal
transfers to an unseen dataset, but at roughly half the in-domain
accuracy (69.3% GBT on UFBA-425, Section 2) - consistent with genuine
domain shift (different clinic/scanner/population), not evidence the
in-domain number was inflated.

**Supernumerary-present vs. control comparison:**

| | accuracy | n |
|---|---|---|
| supernumerary-present instances | 0.2953 | 596 |
| control instances | 0.3302 | 12,572 |
| supernumerary-present images (mean) | 0.2880 | 23 images |
| control images (mean) | 0.3230 | 477 images |

Two-proportion z-test (per-instance): z = -1.771, **p = 0.0765**.
Mann-Whitney U test (per-image, accounts for image-level clustering):
**p = 0.3615**.

**Honest reading: the direction matches the clinical-stakes hypothesis
(lower accuracy near supernumerary teeth) but neither test reaches
conventional significance.** The per-image test - the more appropriate
one, since teeth within an image aren't independent draws - is not close
(p=0.36). With only 23 supernumerary-present images, this comparison is
underpowered to detect anything but a large effect.

**Follow-up 1: localized adjacency test (attempted strengthening -
result withdrawn as confounded, not reported as evidence).** Whole-image
averaging dilutes the signal with teeth far from the actual
supernumerary tooth. Attempted a sharper test: for each supernumerary
instance, its K=3 nearest standard teeth (by centroid distance, same
image) as a "near" set (n=72), vs. the remaining same-image teeth ("far,
same-image," n=524) and the full control set (n=12,572):

| group | accuracy | n |
|---|---|---|
| near (K=3 nearest to supernumerary) | 0.5556 | 72 |
| far, same image | 0.2595 | 524 |
| control (all 477 images) | 0.3302 | 12,572 |

Raw result looked dramatic (near vs. far-same-image: p<0.0001; near vs.
control: p=0.0001) - but a class-composition audit
(`supernumerary_error_check.py`, run after this result, same date)
found it's confounded: **92% of the "near" set (66/72) is drawn from
just 6 anterior classes (11/12/13/21/22/23)**, because supernumerary
teeth (mesiodens) are anatomically concentrated in the anterior maxilla,
not randomly positioned. Those anterior classes are already
differentially easy/hard for the coordinate-only model regardless of any
supernumerary connection (e.g. control-only accuracy: FDI 12 = 0.871,
FDI 13 = 0.313 - a 56pp spread among classes that make up the "near" set
by anatomical necessity, not because of proximity). The matched
per-class comparison is noisy and inconsistent in direction (FDI 11:
0.857 near vs. 0.626 control, n=14; FDI 13: 0.111 near vs. 0.313
control, n=9) - too few instances per class to mean anything, and no
consistent sign. **Conclusion: the localized test's apparent
significance is a class-composition artifact, not evidence about
proximity to a supernumerary tooth in either direction. This result is
withdrawn, not used to support or contradict the clinical-stakes claim.**
Recorded here specifically so this confound isn't rediscovered the hard
way in a future session.

**Follow-up 2: control for teeth-count/crowding.** Fit per-image
accuracy ~ supernumerary_present + n_teeth (OLS, all 500 images):

| term | coef | p |
|---|---|---|
| const | 0.0512 | 0.275 |
| supernumerary_present | -0.0304 | 0.399 |
| n_teeth | +0.0103 | <0.001 |

The supernumerary-present effect persists in the same direction after
controlling for teeth-count, still not significant (p=0.40, consistent
with the whole-image test above). **Bonus finding:** n_teeth itself is
positively and strongly associated with accuracy (r=0.2556, p<0.0001) -
images with more visible teeth are easier for the coordinate-only model,
plausibly because a fuller arch looks more like a canonical/complete
layout. This is a real, separate finding worth citing on its own,
unrelated to the supernumerary question.

**Overall conclusion after both follow-ups: this should be reported as
suggestive-but-inconclusive, not as confirmed evidence for the
clinical-stakes claim** - an honest downgrade from what
`paper/DRAFT.md`'s `[PENDING]` note hoped this follow-up might show, but
the accurate one. The attempt to sharpen the test (localized adjacency)
backfired into a confound rather than a stronger result - a real finding
about this analysis, even though not the one being looked for. More
supernumerary-present images (from the undownloaded remainder of this
dataset, or elsewhere) would be needed to settle the direction.

## 18. Literature check: does a public per-FDI-class confusion matrix from a real trained detector already exist?

**Date:** 2026-09-08 · **Method:** targeted web search + primary-source
verification (arXiv HTML), not a script.

**Motivation:** if a published DENTEX/HierarchicalDet-adjacent paper
already reported a per-class confusion matrix or per-FDI-class accuracy
for a real trained detector, it could be correlated against this
project's coordinate-only error taxonomy (Section 2's mirror-quadrant/
neighbor error fractions) as independent corroborating evidence for
Claim B - without waiting on the Friday GPU run. Checked: the DENTEX
benchmark paper itself (Hamamci et al., "DENTEX: An Abnormal Tooth
Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic
X-rays," arXiv:2305.19112), the DENTEX GitHub repo/challenge page, and
the follow-up participant papers (He L. et al., DentexSegAndDet; Mei S.
et al., YOLOrtho; Choi K. et al., DETDet).

**Result: no such data exists publicly - and the benchmark authors say
so explicitly.** Verified directly against arXiv:2305.19112v2, Section
V-C ("Limitation of the Study"), subsection "Evaluation Metric
Constraints":

> "...it doesn't explicitly distinguish the type of error. Failure to
> detect a tooth altogether is fundamentally different from correctly
> locating a tooth but assigning the wrong number."

> "...understanding the prevalence of specific error types, such as
> swapping the enumeration of adjacent teeth, is crucial."

> "Future work could supplement AP with detailed error analysis, such as
> confusion matrix for the enumeration classes."

The follow-up participant papers checked report only aggregate AP/AR per
task, not per-class confusion matrices.

**What this does and does not support:**
- Does NOT substitute for Task 2 - there is no public per-class error
  data to correlate against, so this cannot become an alternative
  evidence source if the Friday GPU run is delayed or fails. That
  possibility (item 1 of the CPU-in-the-meantime list) is closed off,
  not open.
- DOES strengthen the paper's motivation for Task 2: the benchmark's own
  authors identify exactly this gap - a confusion-matrix-level error
  analysis distinguishing error types (naming "swapping the enumeration
  of adjacent teeth" specifically) - as missing from the literature and
  valuable future work. This is a stronger, more specific motivating
  citation than an assertion that this analysis "would be interesting."
  Cite Hamamci et al. (arXiv:2305.19112) directly in the paper's
  Introduction/Related Work when introducing Task 2, not just DENTEX as
  a dataset/challenge source.

## 19. Robustness of the coordinate-only signal to test-time coordinate noise

**Script:** `cpu_repro/coord_baseline/noise_robustness.py` · **Split:**
same image-level grouped 80/20, 5 seeds (0-4), as the main baseline ·
**Date:** 2026-09-08 · **Data:** `noise_robustness_summary.csv`,
`noise_robustness_raw.csv`, `noise_robustness.png`.

**Motivation:** a real trained detector's predicted boxes are never
pixel-perfect copies of ground truth. This asks how much coordinate
imprecision the shortcut signal can tolerate before it disappears -
context for interpreting Task 2 once a real detector's boxes are
available, and a standalone sensitivity characterization of Claim A.
Method: train on clean boxes, add independent Gaussian noise (std in
normalized 0-1 units) to x_center/y_center/width/height at test time
only, recompute area/aspect_ratio from the jittered width/height,
re-evaluate. For reference, Section 13 measured the natural per-class
positional std at 0.0267 (x) / 0.0490 (y).

| noise std | GBT top-1 acc | logreg top-1 acc |
|---|---|---|
| 0.00 (clean) | 0.6949 ± 0.0077 | 0.6706 ± 0.0095 |
| 0.01 | 0.6109 ± 0.0086 | 0.5812 ± 0.0076 |
| 0.02 | 0.4948 ± 0.0129 | 0.4544 ± 0.0096 |
| 0.04 | 0.3484 ± 0.0094 | 0.2910 ± 0.0041 |
| 0.08 | 0.2067 ± 0.0105 | 0.1624 ± 0.0054 |
| 0.16 | 0.1078 ± 0.0045 | 0.0908 ± 0.0034 |
| 0.32 | 0.0615 ± 0.0019 | 0.0559 ± 0.0048 |

Majority baseline: 0.0363 ± 0.0019 (unchanged across noise levels - it
doesn't depend on coordinates at all).

**Reading:** the signal degrades smoothly and monotonically, not as a
sharp cliff - there's no single noise level where accuracy suddenly
collapses. At std=0.04 (comparable in scale to the natural per-class y
spread measured in Section 13), accuracy is roughly half the clean
value but still ~9.6x the majority baseline. Even at std=0.08 (larger
than any measured natural per-class spread), accuracy remains ~5.7x
baseline. Accuracy only approaches majority-baseline territory at
std=0.32, an implausibly large amount of box noise for any reasonable
detector. **Practical reading for Task 2:** ordinary detector
localization error is very unlikely to erase this signal outright, so
if a real trained detector shows no shortcut-consistent error pattern,
that would be a genuine negative result rather than an artifact of
coordinate imprecision washing out the effect.

## 20. Feature ablation: which geometry features actually carry the signal?

**Script:** `cpu_repro/coord_baseline/feature_ablation.py` · **Split:**
same image-level grouped 80/20, 5 seeds (0-4), as the main baseline ·
**Classifier:** gradient-boosted tree only (matches the main baseline's
stronger classifier) · **Date:** 2026-09-08 · **Data:**
`feature_ablation.csv`, `feature_ablation.png`.

**Motivation:** the main baseline's 69.5% uses all six features
(x_center, y_center, width, height, area, aspect_ratio) as one black
box. This asks which of them actually does the work - position
("where on the jaw"), size/shape ("how big/what proportions"), or only
their combination - turning the accuracy number into a mechanistic
claim instead of an opaque one.

**Single-feature accuracy** (mean ± 95% CI, 5 seeds):

| feature | top-1 acc |
|---|---|
| x_center | 0.3608 ± 0.0094 |
| y_center | 0.1054 ± 0.0051 |
| width | 0.1167 ± 0.0069 |
| height | 0.0716 ± 0.0043 |
| area | 0.1010 ± 0.0051 |
| aspect_ratio | 0.1057 ± 0.0085 |

**Grouped feature sets:**

| feature set | top-1 acc |
|---|---|
| position only (x_center, y_center) | 0.6196 ± 0.0058 |
| shape only (width, height, area, aspect_ratio) | 0.1434 ± 0.0032 |
| all six (main baseline) | 0.6949 ± 0.0077 |

**Leave-one-out (all six minus one):**

| dropped feature | top-1 acc |
|---|---|
| x_center | 0.2198 ± 0.0066 |
| y_center | 0.5266 ± 0.0112 |
| width | 0.6942 ± 0.0070 |
| height | 0.6953 ± 0.0101 |
| area | 0.6930 ± 0.0104 |
| aspect_ratio | 0.6972 ± 0.0077 |

(majority baseline for reference: 0.0363 ± 0.0019)

**Reading: the signal is asymmetrically position-dependent - `x_center`
dominates, `y_center` is real but secondary, and shape is inert.**
Dropping `x_center` costs 47pp (0.6949 -> 0.2198); kept alone it's worth
36pp (0.3608) - the single strongest feature by a wide margin. Dropping
`y_center` costs 17pp (0.6949 -> 0.5266), well outside `x_center`'s CI
and far larger than any shape feature's cost, so it is a real,
non-trivial contributor and not noise - but kept alone it is worth only
11pp (0.1054), barely above shape-feature-alone territory (0.07-0.12).
Shape features (width, height, area, aspect_ratio) are inert either
way: dropping any single one changes accuracy by less than 0.3pp - well
within each other's 95% CIs, i.e. indistinguishable from no effect.
`x_center` alone is a plausible proxy for quadrant/left-right side,
consistent with the main baseline's separately-measured 96-97% quadrant
accuracy (Section 2). The correct summary is **"asymmetrically
position-dependent, dominated by `x_center` but not `x_center`-only"** -
not "position generally, x and y equally," and not "x_center only."

**Follow-on observation (not fully decomposed here):** kept together,
x_center+y_center reach 0.6196 - well above the sum of their individual
marginals (0.3608 + 0.1054 = 0.4662), a superadditive interaction of
about 15pp, consistent with the pair jointly encoding something (e.g.
quadrant) that neither axis alone captures as cleanly. That combined
figure also closes most (89%, 0.6196/0.6949) of the gap to the full
six-feature accuracy, leaving only a 7.5pp residual attributable to
shape features and/or higher-order interactions - suggesting most of
what looked like "unexplained" signal beyond `x_center` alone is
joint x/y interaction, not a hidden contribution from shape. This is
flagged as an observation for a future, more rigorous decomposition
(e.g. explicit quadrant/tooth-type features), not a new claim.

**Why this matters for the paper:** it sharpens Claim A from "geometry
predicts identity" to the more specific and more interpretable
"asymmetrically position-dependent identity prediction, independent of
box size or shape" - a cleaner, more falsifiable mechanistic story, and
useful framing for the mitigation experiment (which perturbs position,
the features that actually matter, not the shape features that don't).

## 21. YOLOv8 training run (Task 2 setup) - GO/NO-GO result: PIVOT

**Script:** `cpu_repro/yolo_training/train_yolo.py` (run via a Kaggle GPU
kernel, Tesla T4) · **Split:** same persisted image-level seed-0 split as
the coordinate baseline (`cpu_repro/coord_baseline/image_split_seed0.csv`,
820 train / 201 val images pre-exclusion) · **Model:** yolov8x.pt, 30
epochs, batch 10, imgsz 640, `fliplr=0.0` (disabled - see note below) ·
**Date:** 2026-09-11 · **Pre-registered rule:** `cpu_repro/yolo_training/GO_NO_GO.md`
· **Data:** `cpu_repro/yolo_training/eval_results/summary.csv`,
`confusion_matrix_yolov8_seed0.csv/.png`.

This completes Section 9's placeholder and answers Claim B (RESULTS.md
Section 11.2): does a real trained detector exploit visual signal beyond
what bounding-box position alone predicts?

**Note on `fliplr`:** Ultralytics' default `fliplr=0.5` mirrors box
coordinates during training augmentation but does not remap the FDI class
label, which encodes left/right quadrant - this would have reintroduced
the same label-mismatch confound as the `MIRROR_MAP` bug fixed in
`notebooks/yolov8+unet/yolov8+unet_training.ipynb` (see the flip/label-mismatch
section of `HANDOFF.md`), except silently, inside Ultralytics' own
augmentation pipeline rather than this repo's code. Disabled rather than
remapped, for this run.

**Result** (matched-detections-only accuracy, per the coordinate baseline's
`evaluate()` function - see `cpu_repro/yolo_training/README.md` for why
detection recall is reported separately rather than folded into top-1):

| metric | coordinate-only baseline (C, GBT, seed 0) | YOLOv8 (Y, seed 0) |
|---|---|---|
| top-1 accuracy | 0.6928 | 0.9558 |
| quadrant accuracy | 0.9661 | 0.9959 |
| tooth-type accuracy | 0.7186 | 0.9587 |
| majority-class baseline | 0.0353 | 0.0359 |
| n_test (instances) | 5491 | 5406 (matched detections only) |
| detection recall | n/a (given ground-truth boxes) | 0.9845 |

**Bootstrap 95% CIs (2026-09-12 addition, image-level resampling, n=2000,
`cpu_repro/yolo_training/robustness_analysis.py`, `bootstrap_ci_section21_26.csv`)
- appended per external-review feedback that headline numbers were bare
point estimates:**

| metric | point | 95% CI |
|---|---|---|
| coord top-1 | 0.6928 | [0.6598, 0.7253] |
| coord quadrant | 0.9661 | [0.9572, 0.9732] |
| coord tooth-type | 0.7186 | [0.6881, 0.7483] |
| YOLO top-1 | 0.9558 | [0.9333, 0.9758] |
| YOLO quadrant | 0.9959 | [0.9932, 0.9984] |
| YOLO tooth-type | 0.9587 | [0.9381, 0.9769] |

Resampled at the image level (not per-instance), so within-image
correlation between multiple teeth on the same radiograph is respected.
None of the CIs overlap between coord and YOLO on any of the three
metrics - the PIVOT call is not an artifact of point-estimate noise.

`n_test` differs because YOLO's number is matched detections only (85
ground-truth boxes went undetected, 98 spurious predictions - both
excluded from the accuracy figures, per the README's stated methodology).

**GO/NO-GO call: Y - C = 0.9558 - 0.6928 = 0.2630 (26.3 percentage
points).** Per the pre-registered rule (frozen before this run, Section 1
of `GO_NO_GO.md`): COMMIT if ≤5pp, PIVOT if ≥15pp, EXTEND if in between.
**26.3pp is a PIVOT**, called against the seed-0-specific baseline value
(0.6928), not the 5-seed aggregate (0.6949 ± 0.0077), per the rule's own
note on which C to use.

**Detection-recall guard (Section 2 of `GO_NO_GO.md`): passed.** 98.45%
recall is well above the 90% minimum, so Y is not an artifact of the
detector only finding easy teeth - the penalized/recall-adjusted Y'
variant specified for a recall failure is not needed here.

**Where the gap comes from (Section 5 of `GO_NO_GO.md`, read
diagnostically alongside the top-1 call, not as an independent
threshold):** quadrant accuracy is flat and near-ceiling for both models
(0.9661 -> 0.9959, +3pp, already close to the 100% ceiling on either
side). The gap is concentrated in tooth-type accuracy (0.7186 -> 0.9587,
+24pp) - precisely the sub-task the coordinate-only model was already
weakest at. Per the pre-written interpretation, this pattern "is
consistent with vision adding value specifically where the coordinate
baseline was already weakest," supporting the PIVOT reading rather than
some other artifact (e.g. a detector that's simply better at the
near-solved quadrant task, which would look different).

**What PIVOT means (Section 3 of `GO_NO_GO.md`), stated but not acted on
here:** the "numbering is primarily a geometric problem" framing is not
well supported by this dataset's own strongest available baseline. The
pre-registered next steps for this branch (error analysis on
detector-right/coordinate-wrong cases; reframing the contribution around
positional priors improving a vision model rather than showing priors
are sufficient alone; or reconsidering whether UFBA-425 is the right
testbed) are a separate decision, not made in this entry.

## 22. Task 2 - YOLOv8 error-pattern comparison against the coordinate-only baseline

**Scripts:** `cpu_repro/coord_baseline/build_coord_baseline.py`
(`is_mirror_quadrant_error`, `is_neighbor_error` - already statically
audited, see `cpu_repro/coord_baseline/mitigation/README.md`) ·
**Data:** `cpu_repro/coord_baseline/per_seed_results.csv` (seed 0),
`cpu_repro/yolo_training/eval_results/summary.csv` · **Date:** 2026-09-11

This is Task 2 proper: given Section 21's real trained-detector
predictions, do YOLO's wrong predictions fall into the same error
categories (mirror-quadrant swaps, same-quadrant neighbor swaps) as the
coordinate-only baseline's, computed the same way, on the same seed-0
split?

| metric | coordinate-only baseline (GBT, seed 0) | YOLOv8 (seed 0) |
|---|---|---|
| n wrong | 1687 (of 5491) | 239 (of 5406 matched) |
| mirror-quadrant error fraction | 0.0753 | 0.0669 |
| mirror-quadrant error count | 127 | 16 |
| neighbor error fraction | 0.8417 | 0.8996 |
| neighbor error count | 1420 | 215 |
| other (neither category) | 140 (8.3%) | 8 (3.3%) |

**Reading:** the two models' wrong predictions land in the *same two
error categories at similar relative rates* - both are overwhelmingly
same-quadrant neighbor swaps (84.2% vs. 90.0% of errors) with a smaller
mirror-quadrant component (7.5% vs. 6.7%), and few errors fall outside
both categories (8.3% vs. 3.3%). YOLO's error-type *distribution* does
not diverge from the coordinate-only baseline's in a way that would
suggest a qualitatively different failure mode - despite YOLO making far
fewer absolute errors (239 vs. 1687, consistent with Section 21's 26.3pp
accuracy gap), the errors it does make are concentrated in the same
anatomically-adjacent-class confusions as the geometry-only model's.
Read this as a similarity-in-error-structure result, not a magnitude
comparison - the mirror/neighbor fractions being close does not offset
or qualify the large top-1 gap in Section 21, which remains the
controlling number for the GO/NO-GO call.

**Not done here (separate, deferred step - see Section 21's closing
note and `HANDOFF.md`):** characterizing *which specific cases* YOLO
gets right that the coordinate-only model gets wrong, i.e. what visual
signal is doing the work on the 26.3pp of accuracy that position alone
does not explain. That is the PIVOT-branch error analysis named in
`GO_NO_GO.md` Section 3, and is not part of this entry.

**Update (2026-09-11): done - see Section 23.**

## 23. PIVOT-branch case study: what is YOLO getting right that the coordinate-only model gets wrong?

**Script:** `cpu_repro/yolo_training/case_study_yolo_vs_coord.py` · **Split:**
same seed-0 split as Sections 21/22 · **Date:** 2026-09-11 · **Data:**
`cpu_repro/yolo_training/eval_results/case_study_yolo_correct_coord_wrong.csv`
(full case set), `cpu_repro/yolo_training/eval_results/case_study_crops/`
(18 sampled image crops + `manifest.csv`).

**Exploratory/diagnostic - this is the PIVOT-branch next step named in
`GO_NO_GO.md` Section 3 and flagged as not-yet-done at the close of
Section 22. No pre-registered pass/fail rule applies here; nothing below
is a GO/NO-GO call.** The framing decision this analysis feeds into
(positional priors as a complement to vision vs. reconsidering whether
UFBA-425 is the right testbed) is explicitly not made here.

**Method:** the coordinate-only GBT model was refit locally (seed 0,
identical to Section 2/21's own training) with per-instance identity
retained, and YOLO's `best.pt` checkpoint (downloaded from the Kaggle
run behind Section 21) was re-run locally on CPU over the same val
images. Both pipelines parse the same label files in the same line
order, so `(label_file, line_idx)` is an exact join key identifying the
same physical tooth annotation in both - verified by an assertion in the
script rather than assumed. This reproduces Section 21/22's numbers
exactly at the aggregate level (5491 joined instances) while keeping the
per-instance predictions both models discard once evaluated.

**Case set:** instances where YOLO's prediction equals ground truth and
the coordinate-only model's does not - **1490 of 5491 instances
(27.1%).**

**Error-type breakdown of the coordinate model's own mistakes, within
this case set** (using the same `is_mirror_quadrant_error`/
`is_neighbor_error` taxonomy as Section 22):

| error type | count | % of case set | for comparison, Section 22's overall coord-baseline rate |
|---|---|---|---|
| neighbor | 1272 | 85.4% | 84.2% |
| mirror-quadrant | 112 | 7.5% | 7.5% |
| other | 106 | 7.1% | 8.3% |

These percentages closely track the coordinate model's overall
error-type mix from Section 22 (84.2% / 7.5% / 8.3%). **Reading: YOLO is
not preferentially fixing one error category over another - it corrects
the coordinate model's mistakes in roughly the same proportions those
mistakes already occur in.** There is no evidence here that YOLO is
specifically better at, say, mirror-quadrant confusions while leaving
neighbor confusions untouched, or vice versa.

**Ground-truth quadrant distribution in the case set:** 1=374, 2=384,
3=404, 4=328 - roughly even across all four quadrants, no strong
clustering.

**Ground-truth tooth-type distribution in the case set** (digit 2 of the
FDI code: 1/2=incisors, 3=canine, 4/5=premolars, 6/7/8=molars):
4 (first premolar) and 1 (central incisor) are the two largest groups
(256, 254), 8 (third molar/wisdom tooth) the smallest (68) - but the
case set spans all eight tooth-type groups without an extreme
concentration in any one, consistent with Section 21's finding that the
top-1 gap is broadly concentrated in tooth-type accuracy rather than one
specific tooth class.

**Qualitative review of sampled crops (8 of the 18 sampled crops
directly inspected, stratified across the three error-type categories
above; full manifest of all 18 in `case_study_crops/manifest.csv`).**
The initial crop (tight, 15% padding around the box) showed only the
single tooth with no surrounding context and was not usable for judging
crowding or adjacent-tooth patterns - crops were regenerated with 150%
padding to show local arch context before this review. Two patterns
were visually consistent across the inspected sample, stated as
observations from a small manually-reviewed set, not a quantified claim:

1. **Mirror-quadrant cases sit visually near the anatomical midline**,
   as expected given `x_center` is the dominant coordinate feature
   (Section 20) - the sampled 31/41 and 11/21 crops show the box
   genuinely close to the jaw's central axis, where left/right position
   alone is inherently close to ambiguous, rather than showing any
   obvious image-quality problem.
2. **Several neighbor/other-category cases show visible irregular tooth
   spacing** - a missing-tooth gap or a visibly tilted/atypically-placed
   tooth in the local arch (most clearly in `case_17_true36_coordpred26_other.jpg`,
   which shows a gap in the lower arch and what looks like a displaced
   upper molar, and `case_13_true33_coordpred35_other.jpg` and
   `case_14_true15_coordpred17_other.jpg`, both showing a visible gap
   near the case tooth). This is consistent with a mechanistic story:
   irregular spacing shifts a tooth's pixel position away from where a
   geometry-only model expects that FDI class to sit, while shape/local
   anatomy still identifies it correctly regardless of position -
   exactly the kind of case a coordinate-only model cannot get right by
   construction, and a real detector can.

**Not a claim:** that missing-tooth/irregular-spacing images make up a
majority of the 1490-case set - only 8 of 1490 cases were actually
looked at. A systematic pass (e.g. an automated crowding/spacing proxy
computed per image, correlated against case-set membership) would be
needed to turn this from a qualitative observation into a quantified
one, and is not done here.

## 24. Framing resolution after the PIVOT result (2026-09-11)

This section records the framing decision made after Section 21's PIVOT
result, resolving the open item Sections 21-23 each explicitly declined
to resolve on their own.

**A timing fact worth being explicit about:** `cpu_repro/yolo_training/GO_NO_GO.md`
was written 2026-09-08T07:55:30-07:00. Section 11's framing decision
("the coordinate-only baseline is a diagnostic/audit tool, not a claim
about detector internals") was written 2026-09-08T08:24:52-07:00 - 29
minutes later, same session - and Section 11 explicitly states it
"supersedes any earlier informal framing ... to the extent that framing
appears elsewhere in this repo." `GO_NO_GO.md` Section 3's prose
interpretation of what a PIVOT result would mean for the paper ("the
'numbering is primarily geometric' framing is not well supported...
vulnerable to the obvious rebuttal") was therefore written for a framing
Section 11 replaced less than half an hour later. This does not change
the pre-registered *numeric* thresholds (Section 1 of `GO_NO_GO.md`,
which correctly governed the COMMIT/PIVOT/EXTEND call in Section 21) -
only the prose interpretation of what a PIVOT outcome means for the
paper's contribution, which this section supersedes in turn.

**Decision: keep the diagnostic-tool framing (Section 11.1) as the
paper's primary contribution.** Claim A (Section 11.2: FDI tooth
identity correlates strongly with bounding-box geometry alone,
well-supported, cross-dataset replicated) is untouched by the YOLO
result - nothing in Sections 21-23 weakens it. What Section 21 actually
measured is whether a specific real trained detector relies on that
geometric correlation as a shortcut, and the answer is a clear **no**:
a detector "gaming" the shortcut would plateau near the coordinate-only
ceiling (69.28%), not exceed it by 26.3 points concentrated almost
entirely in tooth-type accuracy - the exact sub-task Section 20's
feature-ablation already showed geometry cannot push further (shape
features inert, `x_center`/`y_center` together topping out around 62%
even with their superadditive interaction). A detector that were merely
exploiting position would have no way to reach 95.9% tooth-type accuracy
when position alone tops out well below that.

**What this means going forward:** Sections 21-23 are not a pivot away
from the project's contribution - they are the diagnostic-tool framing's
own audit, carried out and quantified, with a clean and interpretable
answer. Future work (the mitigation experiment, boundary-condition
dataset search - both done as of 2026-09-12, see Sections 25 and 31)
should be written up under this framing: not "does the shortcut break under
pressure" as the central question, but continuing to build out the
diagnostic tool and its worked examples (UFBA-425 seed-0 YOLO comparison
being the first full worked example, DENTEX/boundary-condition datasets
as further ones) - a methods contribution, per Section 11.1, not a claim
about what any specific detector's weights are doing internally beyond
what Section 21-23's direct measurement already supports.

**Mitigation-experiment framing, revisited (2026-09-12) now that Section
31's result exists.** `cpu_repro/coord_baseline/mitigation/README.md`
described itself partly in shortcut-reliance terms ("if accuracy
collapses, that itself is further... evidence for Claim B"). Section
31's result was a clean null - removing the position/scale jitter Section
21's run happened to train with did not move accuracy (paired top-1 CI
[-0.74pp, +0.34pp], contains zero) - so there is no collapse to interpret
either as evidence for or against Claim B under that document's own
framing. Read together with this section's reasoning above (a detector
gaming a coordinate shortcut would plateau near the coordinate-only
ceiling, not exceed it by 26.3 points concentrated in tooth-type
accuracy) and Section 20's feature-ablation ceiling (position alone tops
out around 62%, far below either run's ~94-96%), Section 31 is read as
**further support for the diagnostic-tool framing already adopted above,
not a competing data point requiring its own resolution**: jitter
magnitude does not move YOLO's tooth-identification accuracy because
that accuracy was never substantially resting on position to begin with,
consistent with "no" to Claim B rather than a result this section's
framing needs to accommodate. `mitigation/README.md`'s own prose should
still be updated to reflect this outcome rather than left describing a
hypothetical collapse that did not occur - not done here, flagged for
whoever next edits that file.

## 25. Boundary-condition dataset test: DenPAR periapical radiographs

**Script:** `cpu_repro/boundary_condition/denpar_periapical/build_coord_baseline_denpar.py`
· **Dataset:** DenPAR (Rasnayaka et al., *Scientific Data* 12:1615, 2025),
1000 intra-oral periapical (IOPA) radiographs, CC BY 4.0,
[10.5281/zenodo.16645076](https://doi.org/10.5281/zenodo.16645076) -
downloaded directly from Zenodo (md5 `4fcecde6bfa4dc47daec83b1f7856c7f`,
matches the record's published checksum) · **Split:** own image-level
grouped 80/20, 5 seeds (0-4) · **Date:** 2026-09-12.

This is the first real test of the falsifiable boundary-condition
hypothesis stated in Section 11.4: the coordinate-only shortcut should
weaken where the **canonicalized acquisition protocol** precondition
breaks, even with the **structured FDI label space** precondition intact.
UFBA-425 and DENTEX (Sections 2, 10) are both panoramic radiographs - a
full jaw in one consistently-framed image. DenPAR is periapical: each
image shows a small, variable subset of teeth (this dataset: 1-8 teeth,
mean 3.96 after filtering - see below), framed however that exposure
happened to be positioned, with no consistent "where quadrant 3 sits in
the frame" the way a panoramic radiograph has.

**Data construction - the box-to-FDI-code correspondence is not given
directly by the dataset and was independently verified, not assumed.**
DenPAR provides pixel-space bounding boxes per image (`Key Points
Annotations/<id>.json`) and, separately, a spreadsheet giving each
image's visible teeth as an FDI code list in ascending numeric order
(`RawData/Characteristics.xlsx`) - but no documented rule links a
specific box to a specific code in that list. Two images were checked by
hand before writing the loader: image 1 (lower-right, codes
44,45,46,47,48) and image 2 (lower-left, codes 36,37,38). For both,
sorting that image's boxes by x-center ascending and zipping with the
FDI list in the order given produced anatomically correct results -
premolar-shaped crowns at the 44/36 end progressing to molar/wisdom-tooth
shapes toward 48/38, confirmed by directly viewing both radiographs
crop-by-crop. This rule is what the script uses throughout, with an
additional automatic per-image check (the FDI list must itself be
strictly ascending) that drops rather than guesses at any row where the
convention doesn't hold.

**Exclusions, all counted rather than silently dropped:** of 1000 images,
221 excluded for primary/deciduous-tooth FDI codes (51-85 range - pediatric
radiographs, outside this repo's 32-class permanent-tooth `FDI_CODES`) or
malformed spreadsheet entries; a further 146 excluded where the box count
didn't match the FDI-list count for that image (ambiguous correspondence -
dropped rather than guessed at). **Final: 2505 tooth instances from 633
images**, substantially smaller than UFBA-425's 27,563 or DENTEX's 4,872,
which widens this result's confidence intervals accordingly.

**Result** (mean ± 95% CI over 5 seeds):

| metric | UFBA-425 (Section 2, GBT) | DENTEX (Section 10, GBT) | DenPAR periapical (GBT) |
|---|---|---|---|
| top-1 accuracy | 0.6949 ± 0.0077 | 0.6847 ± 0.0124 | **0.2658 ± 0.0277** |
| quadrant accuracy | 0.9653 ± 0.0035 | ~0.98 | **0.4553 ± 0.0236** |
| tooth-type accuracy | 0.7200 ± 0.0084 | - | **0.4501 ± 0.0228** |
| majority-class baseline | 0.0363 ± 0.0019 | - | 0.0759 ± 0.0103 |

Full data: `summary.csv`, `per_seed_results.csv`,
`confusion_matrix_gradient_boosted_tree_seed0.csv`.

**Shuffled-label negative control (seed 0, GBT):** top-1 collapses to
4.2% (below the 8.0% majority baseline, as expected for a genuinely
shuffled/noise classifier) - confirming the 26.6% real result reflects a
real geometry-label correlation, not a training-pipeline artifact on
this much smaller dataset.

**Reading, against the pre-stated falsification threshold (Section 2:
top-1 within 2x of majority baseline falsifies the geometry-predicts-identity
claim):** 2x majority baseline here is 15.2%; the real result (26.6%) is
above that line, so the coordinate-only shortcut has **not** vanished
entirely - but it has dropped from ~69-70% on panoramic radiographs to
~27% here, a collapse to roughly a third of its panoramic-dataset
strength. **Quadrant accuracy is the more striking number**: it was
"close to definitional" on panoramic radiographs (96.5-98%, Section
11.3), but here sits at 45.5% - barely better than tooth-type accuracy
(45.0%) on the same data, and nowhere near the near-ceiling regularity
seen when a full jaw is canonically framed. This is consistent with the
two-precondition hypothesis specifically, not just "harder dataset,
lower accuracy generally": quadrant identity is exactly the coordinate
information that should become unrecoverable once the frame no longer
pins a consistent left/right position to a consistent pixel location,
and that is the sharpest drop of the three accuracy numbers.

**Not a full falsification, and not a clean confirmation either - a
partial, quantified answer:** the hypothesis predicted the shortcut
should *weaken* where the canonicalization precondition breaks while the
label-structure precondition holds, and that is what happened (69-70% ->
27%, above 2x-majority but far below panoramic levels). It does not
predict the shortcut should vanish to noise, and it did not. Read this as
one data point supporting the two-precondition framing, on one
periapical dataset with a much smaller sample than the panoramic
datasets - not as a definitive test of the hypothesis in general.

**Not decided here:** whether/how to commit the raw DenPAR image data
(`cpu_repro/boundary_condition/denpar_periapical/RawData/`, ~217MB) to
this repository. UFBA-425's images are committed directly (existing
project convention), but DenPAR is roughly 3.5x larger and a separate
external license (CC BY 4.0, attribution required) - left as an open
question for whoever commits this section's outputs, not decided
unilaterally here.

**Bootstrap 95% CIs (2026-09-12 addition, image-level resampling,
n=2000, `cpu_repro/boundary_condition/denpar_periapical/bootstrap_ci.py`,
`bootstrap_ci_section25.csv`):**

| metric | point | 95% CI |
|---|---|---|
| top-1 accuracy | 0.2535 | [0.2066, 0.3036] |
| quadrant accuracy | 0.4467 | [0.4004, 0.4940] |
| shuffled-control top-1 | 0.0423 | [0.0242, 0.0600] |

The real-result and shuffled-control CIs do not overlap (real result's
lower bound 0.2066 is well above the control's upper bound 0.0600),
confirming the "not a training-pipeline artifact" claim above holds
under resampling, not just at the point estimate.

**Verification expansion (2026-09-12 addition,
`cpu_repro/boundary_condition/denpar_periapical/verify_expanded.py`,
`verification_review/`) - appended per external-review feedback that a
2-image hand-check should not stand in for real reconciliation.**

*(a) Programmatic check, full coverage (not a sample):* the
spreadsheet's own Arch (Upper/Lower) and Site (Right/Left/Anterior)
metadata was checked against the FDI quadrant digit implied by each
row's own code list, across all 999 parseable spreadsheet rows (FDI
quadrants 1/2/5/6 = upper, 3/4/7/8 = lower; 1/4/5/8 = right side,
2/3/6/7 = left side, correctly including primary-tooth quadrants 5-8,
not just the 32 permanent-tooth classes this repo otherwise uses - a
first version of this check omitted them and reported an inflated
failure count, caught before reporting). Result: **Arch consistent in
979/999 rows (98.0%)**, **Site consistent in 941/999 rows (94.2%,
excluding "Anterior" rows which legitimately span both sides)**. Of the
inconsistent rows, **12 (Arch) and 35 (Site)** are part of Section 25's
actual 633-image used set - e.g. image 81 labeled "Upper" with codes
`42,43,44,45` (unambiguously quadrant 4, lower). This check verifies the
spreadsheet's own internal consistency, not the box-to-code
correspondence itself (no independent second source gives box identity
directly) - but it surfaces real residual label-quality noise in the
instance table beyond what the original exclusion filters (primary-tooth
codes, non-ascending order, box-count mismatch) caught. Not corrected or
re-run here - flagged for whoever next touches this dataset's pipeline.

*(b) Visual spot-check, random sample (not hand-picked):* 20 images
drawn via `numpy.random.RandomState(1)` (a seed distinct from the
project's usual seed 0, and excluding the original 2 hand-picked images)
from the 633-image used set. 2 were unrenderable (box/label count
mismatch, same exclusion rule as the main pipeline - consistent with
that filter's own logic). **Of the 18 renderable images, 10 were
directly reviewed** (rendered with boxes and assigned FDI labels drawn
on the actual radiograph) due to time budget, not all 18: **8/10
anatomically plausible (tooth shape matches assigned label, consistent
progression from premolar- to molar-shaped crowns along the assigned
sequence), 2/10 flagged uncertain** - one (`review_574.jpg`) shows the
radiograph at an atypical near-horizontal orientation, where the
"sort by x-center" convention verified on normally-oriented images may
not hold (the anatomical sequence in that image likely runs along a
different axis); one (`review_1275.jpg`) has an ambiguous tooth-size
judgment call not confidently resolved either way. **Zero clear
failures** (no case where a label was confidently wrong). This is a
larger and non-cherry-picked sample than the original 2-image check, but
still a partial, self-graded visual read, not a ground-truth
reconciliation against an independent label source - stated plainly as
a limitation, not resolved further here.

**Materiality check (2026-09-12 addition,
`cpu_repro/boundary_condition/denpar_periapical/sensitivity_check.py`) -
does the flagged label noise actually move the headline numbers?** The
question raised by the 12/35 flagged rows above is whether they're
concentrated enough in the seed-0 test split to be inflating or
deflating the reported 25.35%/44.67% top-1/quadrant accuracy. Re-scored
the same seed-0 model/test-set from Section 25 with the 15 flagged
(Arch- or Site-inconsistent) images in that test split dropped entirely
(497 instances/126 images -> 435 instances/111 images):

| metric | full test set | excl. flagged images | delta |
|---|---|---|---|
| top-1 accuracy | 0.2535 | 0.2736 | +2.00pp |
| quadrant accuracy | 0.4467 | 0.4782 | +3.15pp |

Both deltas are well inside the bootstrap 95% CI half-widths reported
above (top-1: +/-4.85pp; quadrant: +/-4.68pp) - the flagged rows nudge
the point estimate slightly upward (consistent with them being noise
that the model can't fit, not a source of inflated accuracy) but do not
materially change either headline number or its interpretation. No
correction to Section 25's reported figures is warranted; the flagged
rows are left in for consistency with the rest of the pipeline (which
does not hand-filter on this axis) and the finding is recorded here for
anyone auditing the label-quality caveat above.

## 26. Do YOLO and the coordinate-only baseline fail on the same samples?

**Script:** `cpu_repro/yolo_training/error_correlation_analysis.py` ·
**Split:** same seed-0 split as Sections 21/22/23 · **Date:** 2026-09-12
· **Data:** `cpu_repro/yolo_training/eval_results/error_correlation_joined.csv`
(full per-instance joined table), `error_correlation_summary.csv`.

**Exploratory/diagnostic - no GO/NO-GO rule applies here.** Sections
21/22 compare error *rates* and error *taxonomy* between the two models
but never joined their predictions to check whether they fail on the
same samples. This fills that gap using data that already exists - the
same deterministic coordinate-model refit and CPU YOLO inference as
Section 23, unchanged, just without filtering down to one quadrant of
the 2x2 table. No new training or GPU run.

**2x2 breakdown** (5491 joined instances - undetected ground-truth boxes,
85 of them, count as YOLO-wrong rather than being dropped, since the
coordinate-only model is always given a box to classify and dropping
YOLO's misses would understate its real miss rate on this joined set):

| | coord correct | coord wrong |
|---|---|---|
| **YOLO correct** | 3677 (67.0%) | 1490 (27.1%) |
| **YOLO wrong** | 127 (2.3%) | 197 (3.6%) |

(The 1490 "YOLO correct, coord wrong" cell is exactly Section 23's case
set - same number, same join, cross-checks cleanly.)

**Overlap vs. independence:** P(YOLO wrong) = 0.0590, P(coord wrong) =
0.3072. Under independence, P(both wrong) would be 0.0590 x 0.3072 =
0.0181 (99.5 of 5491 instances). Observed both-wrong is **197 instances
(3.59%) - 1.98x the independence expectation.** Phi coefficient (Pearson
correlation between the two binary wrong/right indicators - chosen over
Cohen's kappa because kappa corrects for chance *agreement*, counting
"both right" and "both wrong" symmetrically, where phi is the direct
correlation between the two error indicators and is numerically
identical to the Matthews correlation coefficient for a 2x2 table) =
**0.163** - a real but modest positive correlation, not independence and
not a strong shared-blind-spot signal either.

**Error-type breakdown on the 197 shared-failure instances**, using the
same `is_mirror_quadrant_error`/`is_neighbor_error` taxonomy as Section
22, computed separately against each model's own wrong prediction:

| error type | coordinate model's own errors | YOLO's own errors |
|---|---|---|
| neighbor | 148 (75.1%) | 133 (88.1% of the 151 with a prediction) |
| other | 34 (17.3%) | 8 (5.3%) |
| mirror-quadrant | 15 (7.6%) | 10 (6.6%) |

YOLO's column covers only 151 of the 197 shared-failure instances -
the remaining 46 are cases where YOLO never detected the tooth at all
(no prediction to categorize), not a wrong-class prediction. Both
models' error-type mix on this shared subset tracks their own overall
error-type mix from Section 22 (dominated by neighbor errors in both
cases) - the shared-failure subset does not concentrate in a distinct
error category relative to either model's general error pattern.

**What this implies for the "complementary signal" framing (Section
24), reported factually - not resolved here:** the dominant relationship
in the 2x2 table is strongly asymmetric (27.1% YOLO-fixes-coord vs. 2.3%
coord-fixes-YOLO, a ~12x imbalance), consistent with Section 21's
26.3-point accuracy gap. The shared-failure cell is real and
statistically above independence (1.98x expected, phi = 0.163) but
small in absolute terms (3.6% of all instances) and not concentrated in
a distinguishable error category from either model's general pattern.
Both facts - the strong asymmetry and the modest-but-nonzero
correlation - are in the data; which one matters more for how the paper
frames "complementary" is not decided in this entry.

**Leverage-point check (2026-09-12 addition,
`cpu_repro/yolo_training/robustness_analysis.py`,
`leverage_check_section26.csv`) - appended per external-review feedback
that a phi of 0.163 risks being oversold and that no check had been made
for whether one FDI class was driving it.** The most frequent single
class among the 197 shared-failure instances is FDI 41 (20 instances,
10.2% of the 197). Excluding every instance with true class 41 (5300 of
5491 instances remain) and recomputing:

| | original (n=5491) | FDI 41 excluded (n=5300) |
|---|---|---|
| phi | 0.1633 | 0.1546 |
| observed/expected ratio | 1.979 | 1.963 |

Both numbers move by less than 0.02 - the effect is not a leverage-point
artifact of one class. **Bootstrap 95% CI on phi (image-level
resampling, n=2000): [0.0985, 0.2202]** - reliably different from 0 (no
correlation) but the interval sits entirely in weak-to-moderate
territory by conventional phi/Cohen's-d-style benchmarks (roughly
<0.1 negligible, 0.1-0.3 small) - consistent with this section's own
"real but modest" characterization above, not a basis for a stronger
claim than that.

## 27. Per-FDI-class breakdown of the YOLO vs. coordinate-baseline gap

**Script:** `cpu_repro/yolo_training/robustness_analysis.py` · **Split:**
same seed-0 split as Sections 21/22/23/26 · **Date:** 2026-09-12 ·
**Data:** `cpu_repro/yolo_training/eval_results/per_class_breakdown.csv`.

**Why:** Sections 21/22 report aggregate top-1/quadrant/tooth-type
accuracy only. A reviewer's first move on a result like this is to check
whether the 26.3pp gap is roughly uniform across all 32 FDI classes or
concentrated in a handful, with some classes possibly flat or reversed
(YOLO losing to the coordinate baseline). Not previously checked.

**Per-class accuracy, both models, sorted by delta (YOLO - coord):**

| FDI | n | coord acc | YOLO acc | delta |
|---|---|---|---|---|
| 46 | 141 | 0.865 | 0.894 | +0.028 |
| 48 | 147 | 0.898 | 0.973 | +0.075 |
| 38 | 148 | 0.912 | 0.993 | +0.081 |
| 18 | 141 | 0.879 | 0.965 | +0.085 |
| 47 | 167 | 0.802 | 0.910 | +0.108 |
| 45 | 169 | 0.769 | 0.929 | +0.160 |
| 28 | 144 | 0.813 | 0.993 | +0.181 |
| 26 | 170 | 0.735 | 0.918 | +0.182 |
| 37 | 166 | 0.765 | 0.952 | +0.187 |
| 36 | 132 | 0.735 | 0.924 | +0.189 |
| 15 | 164 | 0.756 | 0.970 | +0.213 |
| 21 | 179 | 0.760 | 0.983 | +0.223 |
| 27 | 173 | 0.746 | 0.983 | +0.237 |
| 13 | 184 | 0.707 | 0.946 | +0.239 |
| 42 | 194 | 0.670 | 0.918 | +0.247 |
| 16 | 170 | 0.688 | 0.941 | +0.253 |
| 11 | 181 | 0.724 | 0.978 | +0.254 |
| 35 | 163 | 0.663 | 0.926 | +0.264 |
| 17 | 178 | 0.691 | 0.955 | +0.264 |
| 25 | 174 | 0.672 | 0.948 | +0.276 |
| 32 | 192 | 0.641 | 0.917 | +0.276 |
| 12 | 179 | 0.693 | 0.972 | +0.279 |
| 43 | 194 | 0.624 | 0.918 | +0.294 |
| 44 | 184 | 0.630 | 0.935 | +0.304 |
| 22 | 176 | 0.670 | 0.977 | +0.307 |
| 23 | 179 | 0.609 | 0.939 | +0.330 |
| 24 | 165 | 0.576 | 0.909 | +0.333 |
| 34 | 192 | 0.547 | 0.885 | +0.339 |
| 33 | 194 | 0.593 | 0.933 | +0.340 |
| 41 | 191 | 0.508 | 0.874 | +0.366 |
| 14 | 172 | 0.605 | 0.988 | +0.384 |
| 31 | 188 | 0.473 | 0.899 | +0.426 |

**Reversals: zero.** Every one of the 32 FDI classes shows YOLO accuracy
≥ coordinate-only accuracy - not a single class where the coordinate
baseline wins. Stated plainly, per the task's instruction not to soften
whichever way the data goes: **the gap is not concentrated with
reversals; it is uniformly positive and broadly distributed**, though
its *magnitude* varies substantially (delta ranges from +0.028 for FDI
46 to +0.426 for FDI 31).

**Concentration check:** total net correct-count gain (sum of
YOLO-correct minus coord-correct across all 32 classes) = 1363 instances.
The top 5 contributing classes (31, 41, 14, 33, 34 - by share of that
total, not by delta) account for **25.5%** of it. With 32 classes, a
perfectly uniform contribution would put every class at 3.1%; the top 5
(15.6% of classes) contributing 25.5% of the gain is a mild, not
dramatic, over-representation - consistent with "broadly distributed"
rather than "concentrated in a handful." Full per-class table with
`n_instances` and `pct_of_total_gap` columns in `per_class_breakdown.csv`.

**Scope note:** "zero reversals" above is the seed-0 result only, as
explicitly stated. Section 33's multi-seed addition replicates this
breakdown at seeds 1-4 and finds one reversal (seed 1, FDI 38, -1.6pp,
2 instances) - see Section 33 for the full 5-seed picture before citing
"zero reversals" as holding at every seed.

## 28. Baseline hyperparameter effort audit

**Date:** 2026-09-12. Code-reading audit, not a new experiment - checks
whether the coordinate-only baseline and YOLO received comparably
serious hyperparameter search, per external-review feedback that unequal
tuning effort between two arms of a central comparison is a specific,
common failure mode.

**Coordinate-only baseline** (`cpu_repro/coord_baseline/build_coord_baseline.py`,
line 198-199):
```python
"logistic_regression": lambda: LogisticRegression(max_iter=2000),
"gradient_boosted_tree": lambda: HistGradientBoostingClassifier(random_state=0),
```
`max_iter=2000` on the logistic regression is a convergence-safety bump,
not a tuned hyperparameter (`C`, the actual regularization strength,
is left at sklearn's default of 1.0). `HistGradientBoostingClassifier`
has no hyperparameters set at all beyond `random_state` - `learning_rate`,
`max_iter`, `max_leaf_nodes`, `max_depth`, `min_samples_leaf`,
`l2_regularization` are all left at sklearn's library defaults. No
`GridSearchCV`, `RandomizedSearchCV`, or any cross-validation-based
search appears anywhere in this file or elsewhere in the repo (checked
by grep across `.py`/`.md` files for "hyperparameter search", "grid
search", "tuned", "tuning" - no hits describing an actual search
process for either model).

**YOLO** (`cpu_repro/yolo_training/train_yolo.py`): `epochs=30, batch=10,
imgsz=640, dropout=0.6, close_mosaic=0, cos_lr=True, warmup_epochs=10,
lrf=0.005`. These are **not defaults** - Ultralytics' own defaults differ
(e.g. default `lrf=0.01`, `warmup_epochs=3.0`, `dropout=0.0`). But they
were not searched within this project either: they were copied verbatim
from `notebooks/yolov8/yolov8_train.ipynb`'s single, pre-existing CLI
invocation (`epochs=30 batch=10 imgsz=640 ... dropout=0.6 close_mosaic=0
cos_lr=True ... warmup_epochs=10 lrf=0.005`) - one hardcoded command,
written before this round of work, with no visible comparison across
configurations in that notebook or anywhere else in the repo.

**Finding, stated plainly: neither model's hyperparameters were tuned or
searched within this project.** This is not the "deep model gets a real
search, simple baseline gets a token effort" pattern the external
reviews warn about - both arms used un-searched configurations. But
there is a real, if hard-to-quantify, asymmetry worth naming: the
coordinate baseline uses untouched library defaults, while YOLO's
config reflects deliberate, non-default choices whose own tuning history
(if any) predates and is external to this project and is not
documented anywhere accessible here. Whether that undocumented prior
effort matters for interpreting the 26.3pp gap is a judgment call this
audit does not make. **Not acted on here** - per the task's instruction,
this reports the current state; whether to run a real search on the
coordinate baseline (the cheaper of the two to search, since it trains
in seconds) before treating the full 26.3pp gap as attributable to
signal type rather than partly to tuning effort is the user's call.

**Decision (2026-09-12):** disclose, not fix. GPU/CPU budget for the rest
of this project is committed to the mitigation experiment (the
geometry-jitter follow-up to Section 21's PIVOT result, in progress on
Kaggle GPU) and, after that, multi-seed replication of Sections
21/22/25/26 - independently flagged as the
single highest-value remaining investment across the external reviews.
Running a real hyperparameter search now, on either model, would spend
budget on a question this paper's central claim does not depend on: the
claim is about the *relative* exploitability of geometric shortcuts
under a structured label space and canonical framing (Section 11.2's
claim A/B distinction), not about either model being optimally tuned in
an absolute sense. Both models being under-tuned in an undirected way -
neither favored by a search the other didn't get - does not itself bias
that relative comparison; it only means the exact magnitude of the
26.3pp gap should not be over-read as a precisely calibrated number.
Committing to state this plainly in the paper's limitations section,
rather than quietly closing it with a late search whose result could
otherwise be mistaken for having settled the question.

## 29. RESULTS.md internal consistency audit (Sections 21-26)

**Date:** 2026-09-12. Read Sections 21 through 26 in full and
cross-checked every number cited in a later section against the section
where it was originally reported, per external-review feedback that a
contradicted headline number (table vs. figure vs. prose) is a real,
observed failure mode after heavy manual editing - which this file has
had a lot of this session.

**Checked, explicitly:**
- Section 21's table values (0.6928/0.9558 top-1, 0.9661/0.9959
  quadrant, 0.7186/0.9587 tooth-type, 0.0353/0.0359 majority, 26.3pp
  gap) against `eval_results/summary.csv` and
  `cpu_repro/coord_baseline/per_seed_results.csv` seed-0 rows directly -
  match to the reported precision.
- Section 22's n-wrong and error-type counts (1687/239 wrong,
  127/16 mirror, 1420/215 neighbor, 140/8 other) - internally consistent
  (each pair of counts sums to its stated n-wrong; fractions match counts
  divided by n-wrong to the reported precision).
- Section 23's case-set size (1490 of 5491, 27.1%) against Section 26's
  2x2 table - **the 1490 "YOLO-correct/coord-wrong" cell in Section 26
  matches Section 23's case-set count exactly**, and both were verified
  today to also match this session's independently-rerun
  `robustness_analysis.py` output (3677+1490+127+197=5491).
- Section 23's error-type breakdown within the case set (1272/112/106,
  summing to 1490) against Section 22's overall coordinate-model rates
  (84.2%/7.5%/8.3%) - the percentages Section 23 quotes as "for
  comparison" (84.2%/7.5%/8.3%) match Section 22's own reported
  fractions exactly.
- Section 24's citations of Section 21 (26.3pp, 69.28% ceiling, 95.9%
  tooth-type) and Section 20 (position-only ceiling "around 62%",
  actual value 0.6196) - all match their source sections.
- Section 25's table values against `summary.csv`/`per_seed_results.csv`
  in `cpu_repro/boundary_condition/denpar_periapical/` - match to the
  reported precision, including the derived falsification-threshold
  arithmetic (2 x 0.0759 = 0.1518, reported as "15.2%").
- Section 26's P(YOLO wrong)/P(coord wrong) values (0.0590, 0.3072)
  against independently recomputed wrong-counts (324 = 127+197 for
  YOLO, 1687 = 127+... wait, 1490+197 for coord) - **P(coord
  wrong)=0.3072 correctly reproduces Section 22's n-wrong=1687 exactly**
  (1687/5491=0.3072), a cross-section match that would have surfaced a
  stale number had one existed.
- Section 26's shared-failure error-type breakdown (148/34/15 coord,
  133/10/8 YOLO of 151) - independently reproduced by this session's own
  `robustness_analysis.py` run today, exact match to the digit.

**Result: no mismatches found.** Every cross-referenced number checked
out against its source, including several that were independently
recomputed from the raw prediction files today rather than just
re-read from the prose - stated explicitly, per the task's instruction,
rather than silently passing this check.

**Update (2026-09-12): re-audit after today's Section 25/28 appendices.**
The original pass above covers Sections 21-26 as they stood on
2026-09-11 and predates today's three additions (Section 25's bootstrap
CIs, verification expansion, and materiality check; Section 28's
disclosure decision). Re-checked those specifically:
- Section 25's bootstrap CI point estimates (0.2535 top-1, 0.4467
  quadrant, 0.0423 shuffled-control) against `bootstrap_ci_section25.csv`
  directly - match. The stated CI half-widths caught one real error in
  this pass: quadrant accuracy's half-width was originally written as
  "+/-4.66pp"; recomputed from the CSV ([0.40040, 0.49398] around point
  0.44668) it is +/-4.68pp (averaging the asymmetric 4.63pp/4.73pp
  bounds) - fixed in place, not left for a future pass.
- Section 25's materiality-check table (497/126 -> 435/111
  instances/images, 0.2535->0.2736 top-1, 0.4467->0.4782 quadrant)
  against `sensitivity_check_section25.csv` directly - match.
- Found and fixed a real stale cross-reference, independent of the new
  numeric content: Section 24's closing paragraph pointed to "Section
  25+" as where the mitigation experiment would eventually be written
  up, written before Section 25 was assigned to the DenPAR
  boundary-condition test instead. Corrected to point at "its own later
  section, once the GPU run completes" rather than a stale section
  number. Section 28's new decision paragraph had the same category of
  error (attributed the mitigation experiment to being "Section 25/26's
  own follow-up", which is wrong - it follows from Section 21's PIVOT
  result, not the DenPAR/error-correlation sections) - corrected
  alongside it.
- No other mismatches found in Section 27 (per-class breakdown) or the
  rest of Section 28 against their own cited source CSVs.

## 30. Mitigation experiment - pre-commitments made before reading the zero-jitter run's results

**Date:** 2026-09-12, written after `christopherhuang88/tooth-numbering-yolo-train-zerojitter`
reached `KernelWorkerStatus.COMPLETE` on Kaggle but before downloading or
reading its `summary.csv`/predictions - both items below were decided
with no result numbers in hand, per external-review checklist items 2
and 6.

**Confound verification (checklist item 6):** `diff cpu_repro/yolo_training/train_yolo.py
cpu_repro/yolo_training/train_yolo_zerojitter.py` reviewed line by line.
`train_yolo_zerojitter.py` imports `EPOCHS, BATCH, IMGSZ, DEVICE, DROPOUT,
CLOSE_MOSAIC, COS_LR, WARMUP_EPOCHS, LRF, SINGLE_CLS, SAVE_PERIOD,
BASE_WEIGHTS` and the `fliplr=0.0` label-confound fix directly from
`train_yolo` (module-level, not copy-pasted), uses the same persisted
`image_split_seed0.csv` split, and calls `train_yolo`'s own `evaluate()`
unchanged (`base_evaluate`). Its local `train()` override sets only
`TRANSLATE=0.0, SCALE=0.0` plus `RUN_NAME`/`EVAL_RESULTS_DIR` (needed so
this run cannot overwrite Section 21's checkpoint or `summary.csv`).
**Confirmed: TRANSLATE/SCALE is the only intended difference between the
two runs' training configuration.** Hardware: both runs were manually
set to GPU T4 via the Kaggle web UI (per this session's earlier
troubleshooting of the Kaggle API's unreliable `--accelerator` flag,
documented in `HANDOFF.md`) - same accelerator type for both, not left
to chance.

**Analysis method, decided in advance (checklist item 2 - paired vs.
marginal variance, already applied correctly to Section 26's phi
coefficient via image-level bootstrap resampling):** the jittered
(Section 21, `train_yolo.py`) vs. near-zero-jitter (this run,
`train_yolo_zerojitter.py`) comparison will use **paired differences on
the same held-out val images/instances**, not a marginal comparison of
two independent summary numbers. Both runs share the identical seed-0
image-level split (same `image_split_seed0.csv`), so every val instance
has a prediction from both models and can be joined on
`(label_file, line_idx)` exactly as Section 23/26 already do for the
YOLO-vs-coordinate comparison. The write-up will report: (a) the
marginal top-1/quadrant/tooth-type accuracy gap between the two runs,
for direct comparability with Section 21's table format, and (b) a
paired per-instance table (both-correct / jittered-only-correct /
zerojitter-only-correct / both-wrong), a McNemar-style or bootstrap
paired-difference CI on the accuracy gap (image-level resampling,
consistent with every other CI in this document), and detection-recall
for both runs side by side as a guard against jitter changing what gets
detected at all, not just what gets classified once detected. This
commitment is written down now specifically so the choice of paired vs.
marginal analysis is not made after the numbers are already visible.

## 31. Mitigation experiment result: near-zero-jitter vs. jittered YOLOv8

**Scripts:** `cpu_repro/yolo_training/train_yolo_zerojitter.py` (training,
run on Kaggle GPU T4), `cpu_repro/yolo_training/mitigation_analysis.py`
(paired analysis, CPU) · **Split:** identical seed-0 split as Section 21
· **Date:** 2026-09-12 · **Kernel:**
`christopherhuang88/tooth-numbering-yolo-train-zerojitter`, status
`COMPLETE`. Confound verification and the paired-analysis-method
pre-commitment for this write-up are in Section 30, written before these
numbers were read.

**Marginal accuracy** (each run's own `summary.csv`, matched-detections
only - `n_test` differs slightly between runs since detection recall
differs slightly):

| metric | jittered (Section 21) | near-zero-jitter | delta |
|---|---|---|---|
| top-1 accuracy | 0.9558 | 0.9525 | -0.33pp |
| quadrant accuracy | 0.9959 | 0.9963 | +0.04pp |
| tooth-type accuracy | 0.9587 | 0.9551 | -0.36pp |
| detection recall | 0.9845 | 0.9860 | +0.15pp |
| n_test (matched) | 5406 | 5414 | - |

**Paired analysis** (per Section 30's pre-commitment - both checkpoints
re-run locally on CPU over the identical 5491 ground-truth instances,
joined on `(label_file, line_idx)`; here an undetected box counts as
that run's own miss rather than being dropped, same convention as
Section 26, which is why these top-1 numbers read slightly lower than
the matched-only `summary.csv` figures above - they are over a larger,
harder-by-construction denominator, not a different result):

| | zerojitter correct | zerojitter wrong |
|---|---|---|
| **jittered correct** | 5084 (92.59%) | 83 (1.51%) |
| **jittered wrong** | 73 (1.33%) | 251 (4.57%) |

Marginal top-1 (this denominator): jittered=0.9410, zerojitter=0.9392,
delta=-0.18pp. **Bootstrap 95% CI on the paired top-1 delta
(image-level resampling, n=2000): [-0.74pp, +0.34pp] - contains zero.**
Quadrant accuracy CIs also overlap heavily (jittered 99.59% [99.43,
99.75], zerojitter 99.63% [99.47, 99.78]).

**Reading: a clean null result, not an ambiguous one.** The
pre-registered mitigation question (`cpu_repro/coord_baseline/mitigation/README.md`)
was whether removing the position/scale jitter that Section 21's run
happened to train with would collapse YOLO's accuracy - the signature
predicted if that jitter had been suppressing reliance on a
coordinate-based shortcut. It did not: the two runs are statistically
indistinguishable on both top-1 (CI spans zero, both directions) and
quadrant accuracy, with the jittered/zerojitter-only-correct cells
(1.51%/1.33%) nearly balanced rather than lopsided in either direction.
This is evidence *against* the shortcut-suppression framing specifically
(there is no jitter-dependent robustness effect to explain, because
jitter magnitude does not move accuracy at all here) - not merely an
inconclusive result awaiting more power, since 5491 paired instances
with a ~1pp-wide CI is a reasonably tight comparison for this effect
size. Read together with Section 20's feature-ablation ceiling (position
alone tops out around 62%) and Section 24's reasoning (a detector
exploiting position as a shortcut could not reach 95.9% tooth-type
accuracy), this closes the mitigation experiment's original
shortcut-reliance framing with a direct answer: **no**, not "collapsed
under pressure" and not "improved by forcing appearance-invariance to
position" either - YOLO's tooth-type accuracy here does not depend on
the jitter augmentation one way or the other, consistent with it not
being position-reliant to begin with.

**Update (2026-09-12):** Section 24's framing paragraph has since been
revisited with this result in hand - see Section 24's "Mitigation-
experiment framing, revisited" addition. Read as further support for the
diagnostic-tool framing, not a competing data point.

## 32. Multi-seed replication scoping (planning only - no new training run in this entry)

**Date:** 2026-09-12. Scopes the single highest-value remaining
investment flagged across all 12 external reviews (single-seed fragility
of the YOLO-side headline numbers) before committing GPU time to it - a
sizing exercise, not the replication itself.

**1. Per-seed wall-clock time, measured directly (not estimated):** two
real end-to-end Kaggle kernel runs exist at this exact configuration
(30 epochs, batch 10, `yolov8x.pt`, 820 train/201 val images, GPU T4) -
Section 21's original run (2133.0s) and Section 31's zero-jitter run
(2144.9s), both read directly from each kernel's own downloaded log
(`tooth-numbering-yolo-train.log`, `tooth-numbering-yolo-train-zerojitter.log`,
last stdout/stderr event timestamp). **Mean: 2139s ≈ 35.6 minutes ≈ 0.594
GPU-hours per seed**, essentially unaffected by the jitter setting (as
expected - TRANSLATE/SCALE change what gets computed per augmented image,
not how many images or epochs run). The derived CPU-only analyses
(Section 22/26-style joins) add a few more minutes per seed on top - not
GPU-bound, not a meaningful addition to this budget.

**2. Feasibility against Kaggle's quota:** Kaggle's publicly documented
default quota is 30 GPU-hours/week (account-wide, shared across all GPU
kernels) and a 12-hour cap per individual session - **not verified
against this specific account's actual remaining balance**, which this
session has no API access to query (`kaggle kernels status` reports
per-kernel run status, not account-level quota remaining); confirm the
current remaining balance on the Kaggle account page before treating the
plan below as final. At the measured ~0.6 GPU-hr/seed, **GPU-hours are
not the binding constraint** even for a full run: 4 additional seeds
(1-4, to reach 5 total and match Section 2/10/25's existing 5-seed
convention) costs ~2.4 GPU-hours, under 10% of a full weekly budget, and
each individual run (36 min) is nowhere near the 12-hour session cap.
**The actual bottleneck observed this session was not compute-time but a
manual step**: Kaggle's API `--accelerator` flag was unreliable at
actually attaching a T4 (repeatedly defaulted to P100 regardless of the
flag), and the fix both prior runs required was a human manually setting
the accelerator in the Kaggle web UI before each "Save & Run All" -
i.e., a per-kernel manual step, not a per-GPU-hour cost. Four more seeds
means four more such manual kernel launches, not four more hours of
waiting on quota.

**3. Which sections to replicate:** narrower than "all four" once each
section's actual dependency is checked:
- **Section 25 (DenPAR) is already 5-seed** (seeds 0-4, `SEEDS` in
  `build_coord_baseline_denpar.py`) - no new work needed, already
  reported as mean ± 95% CI over 5 seeds.
- **Section 21 (YOLO training)** is the one section that actually needs
  new GPU runs - currently seed-0 only.
- **Section 22 (error-pattern comparison) and Section 26 (error
  correlation / phi)** are CPU-only analyses *derived* from a trained
  YOLO checkpoint plus the coordinate baseline (already 5-seed) - once
  Section 21 has checkpoints at seeds 1-4, these can be recomputed at
  each seed for free (no additional GPU time), using the same join
  pattern already built in `case_study_yolo_vs_coord.py`/
  `error_correlation_analysis.py`/`mitigation_analysis.py`, generalized
  to accept a seed parameter (currently hardcoded to seed 0's split
  file and `runs/yolov8_seed0split/` path - a code change, not a new
  method).
  **Recommendation: replicate all four sections, not a subset** - since
  the GPU cost of the one section that actually needs new training
  (Section 21) is small enough (~2.4 GPU-hr for 4 seeds) that there is
  no real budget pressure to prioritize among them.

**4. What a material shift would mean for the paper:** Section 21's
26.3pp gap is far larger than any seed-to-seed CI half-width measured
anywhere else in this project - UFBA-425's coordinate baseline: ±0.77-
1.8pp (Section 2/11.4); DenPAR: ±2.3-2.8pp (Section 25/materiality
check above). A shift of that magnitude across 5 seeds reversing the
qualitative PIVOT finding (Y-C gap collapsing under the pre-registered
5pp GO threshold, `GO_NO_GO.md`) would be a genuine surprise, not a
plausible outcome given the margin involved. **Pre-committing now, before
any new seed is run:** the write-up will report the actual 5-seed mean ±
95% CI honestly regardless of outcome. If the mean moves but the
qualitative pattern (YOLO >> coordinate-only, gap far exceeding the
5pp threshold) holds, that only **narrows or widens reported
uncertainty** - Section 21's current single-seed CIs (Section 21's
bootstrap addition) capture resampling variance within one training run,
not across-training-run variance, so a real 5-seed CI is a strictly more
honest number, not a correction implying the current one was wrong. Only
a qualitative reversal (gap shrinking under 5pp, or a sign flip) would
require reopening the framing decision (Section 24) - in which case that
reopening happens the same way Section 31's result was handled: written
up factually before deciding what it implies, not reframed after the
fact to fit the existing conclusion.

**Not started here:** no new Kaggle kernel has been launched for this -
this entry is the sizing/scoping decision only, per the task's own
framing ("scope," not "run"). Launching the 4 additional training runs
is a small, well-bounded next step once this scope is confirmed.

## 33. Multi-seed replication result (seeds 1-4, Section 32 executed)

**Scripts:** `cpu_repro/yolo_training/train_yolo_seed{1,2,3,4}.py` (training,
Kaggle GPU T4 each), `cpu_repro/yolo_training/multiseed_analysis.py`
(CPU analysis, seed-parameterized generalization of
`case_study_yolo_vs_coord.py`/`error_correlation_analysis.py`, verified
against Section 21/26/31's published seed-0 numbers before being trusted
for seeds 1-4 - see the script's own assertion, which passed) · **Splits:**
`cpu_repro/coord_baseline/image_split_seed{1,2,3,4}.csv`, generated by the
now-generalized `export_split.py` · **Date:** 2026-09-12 · **Kernels:**
`christopherhuang88/tooth-numbering-yolo-train-seed{1,2,3,4}`, all
`COMPLETE`. Every seed's first push landed on a P100 and fail-fasted in
~10 seconds (same known issue as Sections 21/31 - `ensure_gpu()`'s real
CUDA-matmul check correctly aborted before any GPU-hours were spent);
all four succeeded on the second attempt after manually setting
Accelerator to GPU T4 x2 in the Kaggle web UI before Save & Run All.

**Per-seed results** (matched-detections-only convention, each seed's own
`summary.csv` - directly comparable to Section 21's original table):

| seed | top-1 | quadrant | tooth-type | detection recall | n_test |
|---|---|---|---|---|---|
| 0 (Section 21) | 0.9558 | 0.9959 | 0.9588 | 0.9845 | 5406 |
| 1 | 0.9499 | 0.9970 | 0.9530 | 0.9832 | 5252 |
| 2 | 0.9706 | 0.9983 | 0.9718 | 0.9788 | 5715 |
| 3 | 0.9624 | 0.9981 | 0.9643 | 0.9900 | 5857 |
| 4 | 0.9539 | 0.9980 | 0.9552 | 0.9803 | 4883 |

**5-seed mean +/- 95% CI** (t-distribution, `mean_ci95` - same helper
used everywhere else in this document):

| metric | mean | 95% CI |
|---|---|---|
| top-1 accuracy | 0.9585 | [0.9485, 0.9686] |
| quadrant accuracy | 0.9974 | [0.9962, 0.9987] |
| tooth-type accuracy | 0.9606 | [0.9512, 0.9700] |
| detection recall | 0.9834 | [0.9779, 0.9888] |

**Section 26-style paired analysis, all 5 seeds** (undetected boxes count
as wrong, per Section 26's convention - this is why these top-1 numbers
read slightly lower than the matched-only table above; same denominator
difference already explained in Section 31, not a new discrepancy):

| seed | n | yolo top-1 | coord top-1 | gap | yolo quadrant | phi |
|---|---|---|---|---|---|---|
| 0 | 5491 | 0.9410 | 0.6928 | +24.8pp | 0.9959 | 0.163 |
| 1 | 5342 | 0.9339 | 0.6937 | +24.0pp | 0.9970 | 0.235 |
| 2 | 5839 | 0.9500 | 0.7025 | +24.7pp | 0.9983 | 0.133 |
| 3 | 5916 | 0.9528 | 0.6990 | +25.4pp | 0.9981 | 0.179 |
| 4 | 4981 | 0.9352 | 0.6864 | +24.9pp | 0.9980 | 0.209 |

5-seed mean +/- 95% CI: **gap = 24.77pp +/- 0.61pp [24.16, 25.38]**,
**phi = 0.184 +/- 0.049 [0.134, 0.233]**.

**Reading, per Section 32's pre-commitment (report the honest 5-seed
result regardless of outcome):** the qualitative PIVOT finding replicates
cleanly and is not seed-fragile. The gap's 95% CI [24.16pp, 25.38pp] sits
nowhere near the pre-registered 5pp GO threshold (`GO_NO_GO.md`) - the
narrowest single-seed gap observed (seed 1, 24.0pp) is still nearly 5x
that threshold. Seed-to-seed spread on the gap itself is small (+/-0.61pp
half-width, tighter than Section 25/DenPAR's own seed spread), consistent
with Section 32's prediction that this was very unlikely to reverse.
**Phi (the error-correlation coefficient from Section 26) shows more
seed-to-seed movement than the gap does** - ranging 0.133-0.235 across
the 5 seeds, mean 0.184 with a 95% CI of [0.134, 0.233] - but every
single seed lands in the same qualitative "real but modest positive
correlation" territory Section 26 already reported for seed 0 alone; no
seed shows independence (phi~0) or a strong correlation (phi>0.4). Per
Section 32's Item 4: this is a **narrowing/widening of reported
uncertainty, not a reversal** - the framing decision (Section 24) is not
reopened. No qualitative surprise occurred; the single-seed fragility
concern raised across the external reviews is answered directly: the
headline numbers hold up under replication.

**Update (2026-09-12): the deferred per-seed breakdowns were run after
all**, using the cached per-seed joined tables `build_merged()` now
saves (`cpu_repro/yolo_training/eval_results/multiseed/joined_seed{0-4}.csv`)
- `cpu_repro/yolo_training/multiseed_analysis.py`'s new
`per_class_breakdown()`/`error_taxonomy_breakdown()`/`run_breakdowns()`
functions, generalizing `robustness_analysis.py`'s Section-27 logic and
Section 22's error-type comparison to an arbitrary seed.

**Per-seed error-taxonomy comparison** (Section 22-style, each model's
own wrong predictions - `error_taxonomy_multiseed.csv`), 5-seed mean +/-
95% CI:

| | coordinate-only | YOLOv8 |
|---|---|---|
| mirror-quadrant fraction | 0.0748 +/- 0.0048 | 0.0492 +/- 0.0198 |
| neighbor fraction | 0.8287 +/- 0.0148 | 0.9343 +/- 0.0244 |
| other fraction | 0.0965 +/- 0.0147 | 0.0165 +/- 0.0177 |

Consistent with Section 22's seed-0-only reading across all 5 seeds:
both models' errors are dominated by same-quadrant neighbor confusions
at every seed, with YOLO's error mix, if anything, *more* concentrated
in the neighbor category (93.4% vs. 82.9%) and less in mirror-quadrant
or other errors than the coordinate-only model - not a qualitatively
different failure mode at any seed.

**Per-seed per-class breakdown** (Section 27-style -
`per_class_breakdown_multiseed.csv`): top-5-share-of-gain stays in a
tight 25.0-28.2% band across all 5 seeds (seed 0: 25.5%, matching
Section 27 exactly). **One reversal found, seed 1 only**: FDI 38, coord
89.8% vs. YOLO 88.3% (delta -1.6pp, net -2 correct instances of 128) -
the first reversal observed across 5 seeds x 32 classes = 160
seed-class combinations. Read honestly rather than folded quietly into
"zero reversals": this is a single class, single seed, -1.6pp magnitude,
in a region where both models are already near-ceiling (~88-90%) and a
2-instance swing is well within noise for n=128 - not evidence of a
systematic YOLO weakness on FDI 38, but also not pretended away. Section
27's "zero reversals across all 32 classes" claim was accurate as
stated (it was explicitly scoped to seed 0) and remains true at seed 0;
it does not generalize to "zero reversals at every seed," which this
addition corrects with the actual 5-seed picture.

## Adding a new entry

Append a new numbered section, not an edit to an existing one. Include the
script path, the exact split/seed, the date, and the actual numbers (read
from the CSV, not retyped from memory or a chat transcript) - the same
discipline as `CONVENTIONS.md` and `GO_NO_GO.md`.
