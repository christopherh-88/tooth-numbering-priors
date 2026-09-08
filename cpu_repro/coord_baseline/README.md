# Coordinate-only baseline

**Question:** can FDI tooth class be predicted from bounding-box geometry
alone (x, y, width, height - no pixels)? If yes, position is carrying most
of the classification decision, and a coordinate-only model is a strong
sanity-check baseline for any "vision" tooth-numbering model to beat.

**Answer: yes, overwhelmingly.** A gradient-boosted tree using only
`(x_center, y_center, width, height, area, aspect_ratio)` gets **69.5% top-1
accuracy** across 32 FDI classes, against a **3.6% majority-class baseline**
(i.e. ~19x chance). It gets the **quadrant right 96.5%** of the time and the
**tooth type right 72.0%** of the time. Coordinates alone go most of the way
to solving tooth numbering on this dataset.

## Results (mean +/- 95% CI, 5 seeds, image-level grouped split)

| metric | logistic regression | gradient-boosted tree |
|---|---|---|
| top-1 accuracy (32-way) | 0.6706 +/- 0.0095 | **0.6949 +/- 0.0077** |
| quadrant accuracy (1st FDI digit) | **0.9677 +/- 0.0032** | 0.9653 +/- 0.0035 |
| tooth-type accuracy (2nd FDI digit) | 0.6919 +/- 0.0084 | **0.7200 +/- 0.0084** |
| majority-class baseline | 0.0363 +/- 0.0019 | 0.0363 +/- 0.0019 |
| errors that are mirror-quadrant swaps (e.g. 11 vs 21) | 0.0576 +/- 0.0031 | 0.0748 +/- 0.0048 |
| errors that are immediate-neighbor swaps (e.g. 11 vs 12) | 0.8079 +/- 0.0159 | 0.8287 +/- 0.0148 |

Full per-seed numbers: `per_seed_results.csv`. Machine-readable summary:
`summary.csv`.

**Falsification threshold:** we treat top-1 accuracy within 2x of the
majority-class baseline (~7.3%, given the ~3.63% majority baseline on
UFBA-425) as the region that would falsify the claim that geometry
meaningfully predicts tooth identity; this threshold is stated here
explicitly for clarity but was not pre-registered before the initial
experiment was run.

**Controls** (shuffled-geometry negative control, position-only,
size-only): see `controls/README.md`. Short version: shuffling box
coordinates within each image collapses accuracy to the majority-class
baseline (~3.2-3.6%), confirming the result above is a real
geometry-identity relationship and not a data artifact; position (x,y)
alone recovers ~90% of the full result (62-64% vs 67-69% top-1); size (w,h)
alone gets only 14.7-15.7% top-1 but is still ~4x chance, showing a real,
secondary, location-blind size signal.

## Reading the errors

- **Quadrant is almost never the problem** (~96.5% correct) - geometry alone
  tells you upper/lower and left/right jaw with near-detector-level
  reliability. Only ~6-7% of mistakes are mirror-quadrant swaps (11 vs 21,
  31 vs 41, etc.), even though those are the pairs closest together near the
  midline.
- **Tooth type is where it gets hard** (~70-72% correct) - distinguishing
  "which of the 8 teeth in this quadrant" from box geometry alone is the
  actual difficulty, not which side of the mouth.
- **~81-83% of all errors are immediate-neighbor swaps** (e.g. 14 vs 15, 33
  vs 34) - when the model is wrong, it is almost always off by one tooth
  position along the arch, not confusing unrelated teeth. The five biggest
  single confused pairs in the saved confusion matrix (gradient-boosted
  tree, seed 0) are 31->41, 31->32, 25->24, 34->35, 41->42 - four of five are
  same-quadrant neighbors, and 31<->41 is the midline mirror pair for the
  lower first premolars, which sit close together and similar in size on
  either side of center.
- This is exactly the pattern you'd expect if position/size is a genuinely
  strong but imperfect signal: quadrant (coarse position) is nearly free,
  exact tooth type (fine position + relative size within the arch) is where
  a vision model's texture/shape cues would need to do the remaining work.

## Do errors concentrate in particular tooth types?

Yes, but **not the way "premolars/molars are more self-similar" predicts** -
that hypothesis is about image texture, and this model never sees pixels.
For a coordinate-only model, what matters is geometric separability, and the
result is close to the opposite: **molars are the easiest class to place,
incisors are the hardest.**

| tooth group | gradient-boosted tree | logistic regression |
|---|---|---|
| Molar | **0.786 +/- 0.014** | **0.793 +/- 0.018** |
| Premolar | 0.665 +/- 0.023 | 0.611 +/- 0.040 |
| Canine | 0.652 +/- 0.031 | 0.633 +/- 0.055 |
| Incisor | 0.633 +/- 0.015 | 0.595 +/- 0.013 |

(mean +/- 95% CI over 5 seeds; full table in `accuracy_by_tooth_type.csv`,
plot in `accuracy_by_tooth_type.png`)

The individual-class breakdown (`accuracy_by_fdi_class.csv`) makes the reason
plain: the worst nine classes are 31, 41, 32, 34, 24, 14, 33, 44, 22 - lower
and upper incisors/first premolars, all crowded around the midline where
mirror-quadrant boxes sit close together and left/right position is only
weakly distinguishing. The best nine are 38, 48, 46, 18, 28, 37, 47, 36, 45 -
third and second molars, which sit farthest from the midline, are the most
posterior boxes in the arch, and have the least positional overlap with any
other tooth. Molars are geometrically the *most* distinctive class by
position, even though they'd be argued to look most alike as images -
because this baseline only ever sees geometry, that's the axis that decides
its errors.

## Does accuracy vary with how many teeth are in the image?

Yes, clearly, in the expected direction: fewer teeth in a crop means less
positional context (fewer neighbors to triangulate arch position against),
and accuracy drops accordingly.

| teeth annotated in crop | gradient-boosted tree | logistic regression |
|---|---|---|
| 1-15 | 0.490 +/- 0.089 (n=653) | 0.432 +/- 0.116 (n=653) |
| 16-23 | 0.573 +/- 0.068 (n=2550) | 0.544 +/- 0.050 (n=2550) |
| 24-28 | 0.659 +/- 0.026 (n=5151) | 0.636 +/- 0.041 (n=5151) |
| 29-32 | **0.728 +/- 0.013** (n=18843) | 0.703 +/- 0.004 (n=18843) |
| 33+ | 0.611 +/- 0.034 (n=372) | 0.695 +/- 0.032 (n=372) |

(mean +/- 95% CI over 5 seeds; full table in `accuracy_by_teeth_count.csv`,
plot in `accuracy_by_teeth_count.png`; pooled per-instance predictions with
`teeth_in_crop` count in `pooled_predictions.csv`)

Accuracy rises monotonically from 43-49% (crops with 1-15 teeth) to 70-73%
(crops with 29-32 teeth, the bulk of the data at n=18843) - roughly a
0.28-0.30 absolute-accuracy gap between the sparsest and fullest crops.
Pearson correlation between raw teeth-in-crop count and per-instance
correctness is weak-but-positive and consistent across both classifiers
(r=0.126 logistic regression, r=0.121 gradient-boosted tree;
`teeth_count_correlation.csv`), matching the binned trend.

The one bin that breaks the monotonic pattern is 33+ teeth (n=372, the
smallest bin by far): FDI has only 32 valid codes, so "33+ annotated" means
duplicate or overlapping box annotations in the source labels, not more real
teeth - treat that bin as a noisy edge case from annotation artifacts, not
a genuine reversal of the trend.

## Method notes

- **Leakage guard**: Roboflow generated ~2-3 augmented crops per source
  X-ray (e.g. `cate1-00002_jpg.rf.<hash>.txt` appears 3x with different box
  coordinates, all derived from the same original `cate1-00002.jpg`).
  Splitting by label *file* would leak near-duplicate views of the same
  X-ray across train/test. We split by the *original* image id (filename
  before `_jpg.rf.<hash>`), so every augmented copy of a given X-ray stays
  entirely on one side. Verified: 425 unique source X-rays, 1021 label
  files (avg ~2.4 augmented crops/image), zero image-id overlap between our
  train/test halves in any seed.
- We pooled the repo's `train/valid/test` folders and did our own 80/20
  grouped random split per seed (5 seeds: 0-4), rather than using the
  original YOLO train/valid/test boundary, since the question here is about
  the coordinate signal in general, not about matching the detector's
  training split.
- Features: `x_center, y_center, width, height` (as given, normalized 0-1)
  plus derived `area = width*height` and `aspect_ratio = width/height`.
- Logistic regression: `sklearn.linear_model.LogisticRegression`, features
  standardized first (fit on train only). Gradient-boosted tree:
  `sklearn.ensemble.HistGradientBoostingClassifier` (sklearn's fast
  histogram-based GBM; used instead of the classic `GradientBoostingClassifier`
  for CPU speed at ~22k training rows x 5 seeds - same family of model).
- Majority-class baseline is computed per split (most frequent class_id in
  that seed's train fold), matching what "chance" looks like for that fold.
- 95% CI is a t-interval (df=4) over the 5 seed results, not a bootstrap.

## Rerun

```bash
cd /Users/christopherhuang/Documents/GitHub/tooth-numbering-priors
source .venv312/bin/activate   # or your own env with cpu_repro/requirements.txt installed
python cpu_repro/coord_baseline/build_coord_baseline.py   # main baseline (~2.5 min)
python cpu_repro/coord_baseline/error_breakdown.py        # tooth-type / teeth-count breakdown (~11 min - retrains both classifiers across all 5 seeds again)
```

Edit `SEEDS`, `TEST_FRACTION`, or `CONFUSION_MATRIX_SEED` at the top of
`build_coord_baseline.py` to change the sweep (both scripts import these
from there, so they stay in sync). Edit `TEETH_COUNT_BINS` at the top of
`error_breakdown.py` to change the teeth-count bucketing.

## Files

- `build_coord_baseline.py` - the main baseline script.
- `summary.csv` - mean + 95% CI per classifier (top-level results table).
- `per_seed_results.csv` - all 5x2 individual runs.
- `confusion_matrix_<classifier>_seed0.csv` / `.png` - 32x32 confusion
  matrix from the seed-0 split, one per classifier.
- `error_breakdown.py` - the tooth-type / teeth-count-in-crop follow-up
  analysis.
- `accuracy_by_tooth_type.csv` / `.png` - accuracy grouped into
  Incisor/Canine/Premolar/Molar.
- `accuracy_by_fdi_class.csv` - accuracy for each of the 32 individual FDI
  codes.
- `accuracy_by_teeth_count.csv` / `.png` - accuracy binned by how many teeth
  were annotated in the source crop.
- `teeth_count_correlation.csv` - Pearson correlation between teeth-in-crop
  count and per-instance correctness.
- `pooled_predictions.csv` - every test-set prediction from all 5 seeds x 2
  classifiers, with `teeth_in_crop` and `label_file` attached, in case you
  want to slice it differently.
