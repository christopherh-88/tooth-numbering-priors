# Coordinate baseline: controls

Four conditions, same 5 seeds (0-4) and the same image-level grouped
train/test split per seed as the main baseline (`../build_coord_baseline.py`)
- only the feature columns fed to the classifiers differ, so the four
accuracy numbers below are a fair side-by-side comparison.

| condition | what it tests |
|---|---|
| full (real) | the main baseline: `x_center, y_center, width, height, area, aspect_ratio`, unchanged |
| full (shuffled per image) | same 6 features, but within each image the feature rows are randomly permuted across that image's teeth before the split - same boxes, wrong owners. Negative control: if the real result reflects a genuine geometry-identity relationship rather than a data-layout artifact, this should collapse to the majority-class baseline |
| position only (x,y) | drops width/height/area/aspect entirely - how much signal is in *where* the box is, alone |
| size only (w,h) | drops x/y entirely - how much signal is in *how big/what shape* the box is, alone, with zero positional information |

## Results (mean +/- 95% CI over 5 seeds)

| condition | classifier | top-1 acc | quadrant acc | tooth-type acc |
|---|---|---|---|---|
| full (real) | gradient-boosted tree | **0.6949 +/- 0.0077** | 0.9653 +/- 0.0035 | 0.7200 +/- 0.0084 |
| full (real) | logistic regression | 0.6706 +/- 0.0095 | 0.9677 +/- 0.0032 | 0.6919 +/- 0.0084 |
| full (shuffled per image) | gradient-boosted tree | 0.0324 +/- 0.0033 | 0.2461 +/- 0.0034 | 0.1273 +/- 0.0041 |
| full (shuffled per image) | logistic regression | 0.0360 +/- 0.0029 | 0.2549 +/- 0.0098 | 0.1363 +/- 0.0048 |
| position only (x,y) | gradient-boosted tree | 0.6196 +/- 0.0058 | 0.9625 +/- 0.0025 | 0.6473 +/- 0.0078 |
| position only (x,y) | logistic regression | 0.6419 +/- 0.0134 | 0.9668 +/- 0.0036 | 0.6668 +/- 0.0133 |
| size only (w,h) | gradient-boosted tree | 0.1473 +/- 0.0056 | 0.3476 +/- 0.0063 | 0.3652 +/- 0.0050 |
| size only (w,h) | logistic regression | 0.1568 +/- 0.0072 | 0.3496 +/- 0.0114 | 0.3811 +/- 0.0107 |

Majority-class baseline (reference, same for every condition since it only
depends on the label distribution, not the features): **0.0363**.

Full numbers: `summary.csv` (aggregated), `per_seed_results.csv` (all 4
conditions x 5 seeds x 2 classifiers = 40 individual runs). Plot:
`controls_comparison.png`.

## What this confirms

- **The shuffled control collapses almost exactly to the majority-class
  baseline** (~3.2-3.6% vs. 3.63% baseline, and note the *shuffled* condition
  even lands essentially on top of the *real* majority-baseline number,
  which is exactly what should happen when the geometry-identity link is
  destroyed but the label distribution is not). This rules out the concern
  that the main baseline's 67-69% accuracy is some kind of leakage or
  data-layout artifact (e.g. label files always listing teeth in a fixed
  left-to-right order that a classifier could exploit independent of true
  geometry) - once the true correspondence between a box and its tooth is
  broken, all the signal is gone. Interesting secondary detail: shuffling
  does NOT destroy quadrant/type accuracy down to chance-for-8-or-4-classes
  levels (24-25% quadrant, 13-14% type) - that's because shuffling happens
  *within* an image, and most images contain roughly similar numbers of
  teeth per quadrant/type, so even a random within-image box still lands in
  a plausible region reasonably often. This is expected residual structure
  from the shuffling scheme, not leftover signal.
- **Position alone (x, y) recovers almost all of the real result**: 62-64%
  top-1 vs. 67-69% for the full feature set, and quadrant accuracy is
  essentially unchanged (96.3-96.7% vs 96.5-96.8%). Width/height/area/aspect
  ratio add only ~3-5 points of top-1 accuracy on top of raw position. Most
  of what the "full" model is doing is triangulating from (x, y) alone.
- **Size alone (w, h) is far weaker but not useless**: 14.7-15.7% top-1,
  well above the 3.6% majority baseline (roughly 4x chance) but nowhere near
  position's 62-64%. Quadrant accuracy craters to ~35% under size-only,
  which makes sense - box width/height carries no left/right information at
  all, since tooth shape is roughly mirror-symmetric across the midline.
  Tooth-type accuracy under size-only (36.5-38.1%) is actually within
  shouting distance of what position-only achieves for type (64.7-66.7% is
  still much better, but size-only clearly captures a real, independent
  signal): tooth size genuinely correlates with tooth type (molars are
  bigger than incisors), so size alone gets meaningfully more than chance
  at distinguishing "what kind of tooth" even with zero location
  information.
- Put together: position is doing the vast majority of the classification
  work; size is a real but secondary and location-blind signal; and the
  shuffled control demonstrates the whole effect is genuine geometric
  correlation, not a data artifact.

## Method notes

- Shuffling is done once per seed, over the **whole** dataset (train and
  test together) before the grouped split is applied by row index, so both
  the shuffled-train and shuffled-test folds have the geometry-identity
  link broken - this is a clean ablation of the signal itself, not a
  robustness-to-noise check.
- "Per image" = per YOLO label file (each `.txt` file is one image's/crop's
  full set of annotated teeth) - the natural atomic unit in this dataset.
  Images with fewer than 2 annotated teeth are left unshuffled (nothing to
  permute).
- Train/test membership (which images fall in the held-out fold) is
  identical across all four conditions for a given seed - the split only
  depends on `image_id`, never on the feature values - so differences
  between conditions are purely about which features the classifier saw.
- Same two classifiers as the main baseline: `LogisticRegression` (features
  standardized, fit on train only) and `HistGradientBoostingClassifier`.

## Rerun

```bash
cd /Users/christopherhuang/Documents/GitHub/tooth-numbering-priors
source .venv312/bin/activate   # or your own env with cpu_repro/requirements.txt installed
python cpu_repro/coord_baseline/controls/run_controls.py
```

Took about 10 minutes on an M-series Mac CPU (40 model fits: 4 conditions x
5 seeds x 2 classifiers). Edit `SEEDS`/`TEST_FRACTION` in
`../build_coord_baseline.py` to change the sweep (this script imports them
from there to stay in sync with the main baseline).

## Files

- `run_controls.py` - the script.
- `summary.csv` - mean + 95% CI per condition x classifier (the table above).
- `per_seed_results.csv` - all 40 individual runs.
- `controls_comparison.png` - bar chart of top-1 accuracy across the four
  conditions, both classifiers, with the majority-class baseline marked.
