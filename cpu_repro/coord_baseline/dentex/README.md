# Coordinate-only baseline: DENTEX

Same question as `../README.md` (`../build_coord_baseline.py`), asked of a
second, independent dataset: **does the UFBA-425 coordinate-only result
replicate on DENTEX?** Same feature set
(`x_center, y_center, width, height, area, aspect_ratio`), same 5 seeds
(0-4), same image-level grouped 80/20 split, same `evaluate()` function -
all imported directly from `build_coord_baseline.py`, not reimplemented -
so every number below lands in the same column as the UFBA-425 table and is
directly comparable without re-deriving anything.

Data: `cpu_repro/anomaly_scan/dentex_raw/dentex_full_fdi_table.csv` (built
by `dentex_raw/build_table.py`) - 21,806 tooth instances across 1358
images, pooling DENTEX's `quadrant_enumeration` (634 images) and
`quadrant_enumeration_disease` (705 train + 50 validation) tiers, the two
tiers that carry a full FDI code. Convention verified to match UFBA-425
before combining anything (`../../CONVENTIONS.md`); the two tiers were
hash-compared on a 30-image sample and found effectively disjoint (0/30
content matches despite overlapping filename indices), so splitting the
pooled set by `image_id` carries no cross-tier leakage.

## Answer: yes, it replicates, and closely

| metric | UFBA-425 (GBT) | DENTEX (GBT) |
|---|---|---|
| top-1 accuracy (32-way) | 0.6949 +/- 0.0077 | **0.6847 +/- 0.0124** |
| quadrant accuracy | 0.9653 +/- 0.0035 | **0.9796 +/- 0.0018** |
| tooth-type accuracy | 0.7200 +/- 0.0084 | 0.7010 +/- 0.0127 |
| majority-class baseline | 0.0363 +/- 0.0019 | 0.0375 +/- 0.0009 |

Two datasets, different institutions, different acquisition protocols,
different label pipelines - within ~1 point of top-1 accuracy and DENTEX's
quadrant accuracy is actually *higher*. This is strong evidence the
coordinate-identity relationship is a structural property of panoramic
dental geometry, not an artifact of one dataset's collection or annotation
process.

## Results (mean +/- 95% CI, 5 seeds, image-level grouped split)

| metric | logistic regression | gradient-boosted tree |
|---|---|---|
| top-1 accuracy (32-way) | 0.6129 +/- 0.0180 | **0.6847 +/- 0.0124** |
| quadrant accuracy (1st FDI digit) | 0.9812 +/- 0.0016 | 0.9796 +/- 0.0018 |
| tooth-type accuracy (2nd FDI digit) | 0.6271 +/- 0.0194 | **0.7010 +/- 0.0127** |
| majority-class baseline | 0.0375 +/- 0.0009 | 0.0375 +/- 0.0009 |
| errors that are mirror-quadrant swaps | 0.0334 +/- 0.0056 | 0.0438 +/- 0.0047 |
| errors that are immediate-neighbor swaps | 0.8187 +/- 0.0126 | 0.8423 +/- 0.0135 |

Per-seed split sizes (seed 0): 1087 train images / 271 test images, 17,455
train instances / 4,351 test instances - all 5 seeds land within a few
hundred instances of that.

Full per-seed numbers: `per_seed_results.csv`. Machine-readable summary:
`summary.csv`. Confusion matrices (seed 0):
`confusion_matrix_<classifier>_seed0.csv` / `.png`.

## Controls

Same four conditions as `../controls/README.md`, same method, run on
DENTEX (`controls/run_controls_dentex.py`):

| condition | classifier | top-1 acc | quadrant acc | tooth-type acc |
|---|---|---|---|---|
| full (real) | gradient-boosted tree | **0.6847 +/- 0.0124** | 0.9796 +/- 0.0018 | 0.7010 +/- 0.0127 |
| full (real) | logistic regression | 0.6129 +/- 0.0180 | 0.9812 +/- 0.0016 | 0.6271 +/- 0.0194 |
| full (shuffled per image) | gradient-boosted tree | 0.0365 +/- 0.0027 | 0.2521 +/- 0.0085 | 0.1375 +/- 0.0066 |
| full (shuffled per image) | logistic regression | 0.0402 +/- 0.0041 | 0.2645 +/- 0.0091 | 0.1506 +/- 0.0078 |
| position only (x,y) | gradient-boosted tree | 0.6174 +/- 0.0088 | 0.9722 +/- 0.0044 | 0.6368 +/- 0.0083 |
| position only (x,y) | logistic regression | 0.6342 +/- 0.0203 | 0.9755 +/- 0.0006 | 0.6520 +/- 0.0204 |
| size only (w,h) | gradient-boosted tree | 0.1336 +/- 0.0066 | 0.3386 +/- 0.0093 | 0.3346 +/- 0.0084 |
| size only (w,h) | logistic regression | 0.1425 +/- 0.0092 | 0.3255 +/- 0.0088 | 0.3567 +/- 0.0118 |

Full numbers: `controls/summary.csv`, `controls/per_seed_results.csv` (4
conditions x 5 seeds x 2 classifiers = 40 runs). Plot:
`controls/controls_comparison.png`.

Same pattern as UFBA-425: shuffling collapses accuracy to essentially the
majority-class baseline (3.65-4.02% vs. 3.75% baseline), ruling out a
data-layout artifact as the source of the real result. Position alone
(x,y) recovers almost all of the full signal (61.7-63.4% vs. 61.3-68.5%
top-1) - even more of it than on UFBA-425, where position-only trailed the
full feature set by a slightly wider margin. Size alone (w,h) gets
13.4-14.3%, well above the 3.75% majority baseline (~3.6-3.8x chance) but a
small fraction of the full result - the same secondary, location-blind size
signal seen on UFBA-425.

## Stratified analysis: does accuracy drop on the anomalous strata?

Uses the exact same 5 trained models as the main baseline above (not
retrained) - per seed, both classifiers are trained once on that seed's
train fold, and the resulting test-set predictions are then sliced by
stratum membership before scoring, so this is a slice of the same result,
not a new experiment (`stratified_analysis.py`).

Two strata, each compared against a **canonical** remainder (every test
instance in neither stratum):

- **dissociation** (image-level flag): every test instance belonging to
  one of the 101 images flagged by `scan_dentex.py` for a duplicate FDI
  code and/or a position-vs-quadrant violation (`../../anomaly_scan/
  dentex_findings.md`). 1354 instances total across the full dataset, not
  just the flagged box itself - flagging is per-image, not per-tooth.
- **impacted** (instance-level flag): every test instance whose diagnosis
  is "Impacted" (644 total in the full dataset).
- The two strata overlap a little (77 instances are both flagged-image and
  impacted-diagnosis) - reported, not hidden, and each instance still only
  counts once per stratum in its own row.

| stratum | classifier | top-1 acc | quadrant acc | tooth-type acc | pooled n | min seed n |
|---|---|---|---|---|---|---|
| canonical | gradient-boosted tree | 0.6827 +/- 0.0146 | 0.9815 +/- 0.0025 | 0.6981 +/- 0.0141 | 20,229 | 3,968 |
| canonical | logistic regression | 0.6080 +/- 0.0202 | 0.9828 +/- 0.0013 | 0.6214 +/- 0.0213 | 20,229 | 3,968 |
| dissociation | gradient-boosted tree | 0.6195 +/- 0.0531 | 0.9484 +/- 0.0138 | 0.6509 +/- 0.0512 | 1,291 | 210 |
| dissociation | logistic regression | 0.5506 +/- 0.0442 | 0.9532 +/- 0.0129 | 0.5791 +/- 0.0366 | 1,291 | 210 |
| impacted | gradient-boosted tree | 0.9144 +/- 0.0415 | 0.9790 +/- 0.0117 | 0.9339 +/- 0.0415 | 614 | 109 |
| impacted | logistic regression | 0.9440 +/- 0.0287 | 0.9889 +/- 0.0105 | 0.9551 +/- 0.0277 | 614 | 109 |

Both strata clear the size thresholds set before running this
(`MIN_POOLED_N=100`, `MIN_PER_SEED_N=20` per seed) - `too_small` is `False`
for every row (`stratified_summary.csv`). Neither stratum is too small to
support the comparison below.

**Plain answer:**

- **dissociation: yes, accuracy drops, by about 6 points.** GBT top-1 goes
  from 68.3% (canonical) to 62.0% (dissociation), a drop of **6.3 points**;
  logistic regression drops 5.7 points (60.8% to 55.1%). This is the
  expected direction: images flagged for duplicate codes or a
  position-vs-quadrant violation are, almost by definition, images where a
  box's position doesn't cleanly match one true FDI identity, so a
  coordinate-only model - which has no other information to fall back on -
  does worse on them. The drop is real but modest (not catastrophic),
  because a flagged *image* still contains mostly well-behaved boxes; only
  a minority of the ~13 teeth in a typical flagged image are the actual
  anomalous ones. See "Composition check" below - tooth-type composition
  does not explain this drop away, but the drop itself is directionally
  consistent, not statistically significant at conventional thresholds
  (see the paired-seed statistics there).
- **impacted: no - accuracy goes up, but the original "23 points"
  framing overstated why.** See "Composition check" immediately below
  before reading any impacted-stratum number on its own - the raw
  comparison mixes a compositional effect with a real one, and needs to be
  read as a third-molar-only result, not a cross-tooth-type one.

Full numbers: `stratified_summary.csv` (aggregated),
`stratified_per_seed_results.csv` (every seed x classifier x stratum row,
including `n_test` per cell).

## Composition check: is the impacted-stratum result a tooth-type artifact?

**Read this first: DENTEX's "Impacted" diagnosis label applies exclusively
to third molars in this dataset - 644/644 impacted-diagnosis instances
(100%) carry FDI code 18, 28, 38, or 48.** Anyone treating DENTEX's
impacted label as a general anomaly category is measuring something
narrower than they think: it is not "any impacted tooth," it is
"impacted third molar," because no other tooth position in this dataset is
ever annotated as Impacted. Every accuracy number for the impacted stratum
below should be read as a third-molar-only result, not a cross-tooth-type
one.

Third molars were already shown to be the easiest tooth-type group to
place from geometry alone in the UFBA-425 breakdown (`../README.md`), so
any raw comparison between impacted (100% third molars) and canonical
(8.7% third molars) is partly comparing different tooth-type mixes, not
just "impacted vs. not." `composition_check.py` controls for this two
ways, reusing the exact same trained models and test-fold predictions as
`stratified_analysis.py` (not new models, same 5 seeds):

| stratum | n | third-molar share (full dataset) |
|---|---|---|
| canonical | 19,885 | 8.7% |
| dissociation | 1,354 | 14.2% |
| impacted | 644 | **100.0%** |

**Matched subset - third molars only (FDI 18/28/38/48), GBT, pooled over 5 seeds:**

| stratum | n | top-1 acc (third molars only) |
|---|---|---|
| canonical | 1,769 | 0.7354 |
| dissociation | 168 | 0.8214 |
| **impacted** | 614 | **0.9121** |

**Raw vs. composition-controlled gap vs. canonical (GBT, pooled over 5 seeds):**

| stratum | raw gap | composition-controlled residual | non-third-molar-only gap |
|---|---|---|---|
| dissociation | -6.1pp (0.6220 vs 0.6827) | **-6.4pp** (0.6220 vs canonical reweighted to 0.6855) | -8.5pp (0.5922 vs 0.6776, third molars excluded entirely) |
| impacted | +22.9pp (0.9121 vs 0.6827) | **+17.1pp** (0.9121 vs canonical reweighted to 0.7411) | n/a - impacted is 100% third molars, no non-third-molar subset exists |

("Composition-controlled residual" = actual stratum accuracy minus what
canonical's accuracy *would be* if canonical's per-FDI-code accuracy were
reweighted to match the stratum's own FDI-code mix - i.e. the part of the
gap tooth-type mix does not explain. Full numbers: `composition_summary.csv`,
`composition_matched_subset.csv`, `composition_standardized.csv`.)

**Corrected reading:**

- **Dissociation: not a composition artifact, but not an established
  finding either.** Dissociation's third-molar share (14.2%) is only
  mildly higher than canonical's (8.7%), and controlling for it does not
  shrink the drop - it is slightly larger controlled (-6.4pp) than raw
  (-6.1pp), and excluding third molars entirely widens it further
  (-8.5pp). Composition is not the explanation. But those are pooled point
  estimates, not a significance test. Computed as a paired comparison
  across the same 5 seeds (dissociation vs. canonical, GBT): mean paired
  diff **-6.32pp**, 95% CI **[-12.67pp, +0.03pp]**, paired t(4) = -2.76,
  **p = 0.0506**, direction consistent in 5/5 seeds -
  **directionally consistent but not significant at conventional
  thresholds given n=5 independent splits**. Do not describe this as
  "surviving" or "holding up."
  **Untested hypothesis, not a fact:** the dissociation flag is
  image-level, not tooth-level, so a flagged image still contains mostly
  well-behaved teeth; this likely dilutes the stratum-average estimate
  toward zero relative to the true per-tooth effect on genuinely anomalous
  instances, but that dilution has not been quantified here.
- **Impacted: the raw "23pp easier" framing conflated two effects.**
  Restricting the comparison to third-molars-only (the matched-subset
  table above) removes the tooth-type-mix confound directly: impacted
  third molars score 0.9121 vs. canonical third molars at 0.7354, a
  **17.7pp gap that survives the matched, apples-to-apples comparison**.
  The composition-standardized calculation agrees: of the raw 22.9pp gap,
  about 5pp (22.9 - 17.1) is attributable to the stratum being entirely
  third molars (the easiest FDI codes to place from geometry alone), and
  the remaining **~17.1pp is a residual difference specific to impacted
  third molars vs. other third molars**, not explained by tooth-type
  composition. The accurate statement is: **impacted third molars are
  about 17.7 points more geometrically predictable than erupted third
  molars (0.912 vs. 0.735)**, with a further ~5 points of the original raw
  22.9pp gap attributable to the stratum being entirely third molars to
  begin with. The original "impacted teeth are 23pp easier" claim
  overstated the effect by treating a within-third-molar difference as if
  it applied across all tooth types.

**Two hypotheses for the residual (not tested by this data, not
conclusions):**

1. *Annotation-consistency hypothesis:* unerupted/impacted third molars may
   get bounding boxes drawn more consistently by annotators than erupted
   teeth (less occlusal overlap, less ambiguity about where the crown
   starts or ends), which could make their geometry more learnable
   regardless of anatomical predictability.
2. *Position-extremity hypothesis:* impacted third molars may sit in more
   extreme or less-crowded positions within the jaw than erupted third
   molars (e.g. more consistently at the far posterior/inferior extreme,
   less neighbor overlap), making them easier to localize by coordinates
   alone.

Neither hypothesis is checked here - both remain open questions this
dataset alone cannot resolve, and nothing in this analysis distinguishes
between them.

## Method notes

- One DENTEX image is one atomic annotation unit (no Roboflow-style
  augmented-crop duplication like UFBA-425 has) - `label_file` is set to
  `image_id` directly in `load_instances()`, and the grouped split
  operates on that.
- DENTEX only has permanent-dentition annotations (quadrants 1-4), so it
  uses the identical 32-code `FDI_CODES` list from `build_coord_baseline.py`
  - a `class_id` means the same FDI code in both baselines, with no
  remapping.
- Same classifiers, same scaling, same majority-baseline-per-split
  convention, same t-interval (df=4) 95% CI as `../build_coord_baseline.py`
  - see its README for the full method notes; nothing about the modeling
  approach was changed for DENTEX.

## Rerun

```bash
cd /Users/christopherhuang/Documents/GitHub/tooth-numbering-priors
source .venv312/bin/activate
python cpu_repro/coord_baseline/dentex/build_coord_baseline_dentex.py            # main baseline (~3 min)
python cpu_repro/coord_baseline/dentex/controls/run_controls_dentex.py           # controls (~10 min)
python cpu_repro/coord_baseline/dentex/stratified_analysis.py                    # stratified comparison (~3 min)
python cpu_repro/coord_baseline/dentex/composition_check.py                      # composition check (~3 min)
```

## Files

- `build_coord_baseline_dentex.py` - main baseline script.
- `summary.csv` / `per_seed_results.csv` - main baseline results.
- `confusion_matrix_<classifier>_seed0.csv` / `.png` - 32x32 confusion
  matrices, seed-0 split.
- `controls/run_controls_dentex.py` - the four-condition controls script.
- `controls/summary.csv` / `controls/per_seed_results.csv` /
  `controls/controls_comparison.png` - controls results.
- `stratified_analysis.py` - the dissociation/impacted vs. canonical
  comparison.
- `stratified_summary.csv` / `stratified_per_seed_results.csv` -
  stratified results.
- `composition_check.py` - controls the stratified comparison for
  tooth-type (third-molar) composition.
- `composition_summary.csv` / `composition_matched_subset.csv` /
  `composition_standardized.csv` - composition-check results.
