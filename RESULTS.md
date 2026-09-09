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

- **Error-pattern correlation check.** Compare a real trained detector's
  confusion matrix (mirror-quadrant swaps, neighbor-tooth confusions)
  against the coordinate-only model's, using the existing
  `is_mirror_quadrant_error` / `is_neighbor_error` taxonomy code in
  `cpu_repro/coord_baseline/build_coord_baseline.py`. No retraining
  needed - run existing trained-detector predictions through the existing
  error-taxonomy code. This is the cheapest available step toward
  evidence on claim B (Section 11.2).
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
- **Boundary-condition dataset (not yet chosen/run).** Section 11.4 states
  the shortcut should vanish where either precondition breaks. Candidates,
  not yet searched for or downloaded:
  1. **Bitewing/periapical radiographs** (breaks precondition 2 -
     acquisition is per-tooth/angled, not a fixed whole-jaw layout;
     same clinical domain and FDI-adjacent labeling as the existing work,
     no volumetric complexity). Current leading candidate.
  2. **CBCT slices** (breaks precondition 2 more severely, but adds
     volumetric/3D framing that complicates the box-geometry setup and
     may distract from the core comparison).
  3. **A structurally different label space on the same panoramic
     modality**, e.g. a landmark-detection task (breaks precondition 1
     instead of 2, isolating that variable rather than conflating both).
  None of these have been located, downloaded, or run yet - this is a
  target list, not a result.
- **Mitigation experiment (not yet run).** See
  `cpu_repro/coord_baseline/mitigation/README.md` - design only, blocked
  on GPU access to train a real detector with/without the intervention.

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

## Adding a new entry

Append a new numbered section, not an edit to an existing one. Include the
script path, the exact split/seed, the date, and the actual numbers (read
from the CSV, not retyped from memory or a chat transcript) - the same
discipline as `CONVENTIONS.md` and `GO_NO_GO.md`.
