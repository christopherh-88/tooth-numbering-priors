# Detector robustness across training backends: Kaggle CUDA vs local Apple Silicon (MPS)

Three detectors (YOLOv8x, RT-DETR-l, Faster R-CNN `fasterrcnn_resnet50_fpn_v2`), 30 epochs each, one fixed configuration per detector, no retuning. Every seed has its own grouped image-level split (340 train / 85 test images). All numbers come from `detector_backend_comparison.py` (`eval_results/detector_backend_comparison.csv`); a seed with no result is printed as MISSING.

Run layout:

| Group | Seeds | Backend |
|---|---|---|
| CUDA seeds 0-4 | 0-4 | Kaggle T4 |
| CUDA seeds 5-9 | 5-9 | Kaggle T4 |
| MPS seeds 5-9 | 5-9 | MacBook Air M3 |
| CUDA seeds 10-14 | 10-14 | Kaggle T4 (only seeds 10-11 done so far) |

Seeds 5-9 were trained on both backends with identical splits, so that pair is compared seed by seed. The other groups are separate samples on different splits.

## Results

Headline metric is all-GT top-1: every labeled tooth is in the denominator, and a tooth with no matching predicted box counts as wrong.

| Detector | Group | n | All-GT top-1 (mean ± sd) | Range | Missed-tooth rate | Matched-only top-1 |
|---|---|---|---|---|---|---|
| YOLOv8x | CUDA seeds 0-4 | 5 | 0.9426 ± 0.0086 | 0.9339-0.9528 | 1.66% | 0.9585 |
| YOLOv8x | CUDA seeds 5-9 | 5 | 0.9380 ± 0.0205 | 0.9132-0.9547 | 2.31% | 0.9601 |
| YOLOv8x | MPS seeds 5-9 | 5 | 0.9423 ± 0.0062 | 0.9365-0.9509 | 1.99% | 0.9614 |
| YOLOv8x | CUDA seeds 10-14 | 2 | 0.9359 ± 0.0010 | 0.9352-0.9366 | 2.23% | 0.9573 |
| RT-DETR-l | CUDA seeds 0-4 | 5 | 0.9448 ± 0.0072 | 0.9345-0.9512 | 1.11% | 0.9554 |
| RT-DETR-l | CUDA seeds 5-9 | 5 | 0.9532 ± 0.0062 | 0.9461-0.9612 | 0.82% | 0.9611 |
| RT-DETR-l | MPS seeds 5-9 | 5 | 0.9527 ± 0.0075 | 0.9446-0.9624 | 0.91% | 0.9615 |
| RT-DETR-l | CUDA seeds 10-14 | 1 | 0.9484 | 0.9484 | 0.98% | 0.9578 |
| Faster R-CNN | CUDA seeds 0-4 | 5 | 0.9332 ± 0.0095 | 0.9212-0.9425 | 1.36% | 0.9461 |
| Faster R-CNN | CUDA seeds 5-9 | 5 | 0.9400 ± 0.0088 | 0.9279-0.9519 | 1.11% | 0.9505 |
| Faster R-CNN | MPS seeds 5-9 | 5 | 0.9377 ± 0.0106 | 0.9213-0.9470 | 1.21% | 0.9492 |
| Faster R-CNN | CUDA seeds 10-14 | 2 | 0.9315 ± 0.0129 | 0.9224-0.9406 | 1.76% | 0.9481 |

Across all CUDA seeds run so far: YOLOv8x 0.9396 ± 0.0137 (n=12), RT-DETR-l 0.9490 ± 0.0073 (n=11), Faster R-CNN 0.9358 ± 0.0095 (n=12).

## Paired comparison on the shared splits (seeds 5-9)

MPS minus CUDA, all-GT top-1, same split per seed:

| Detector | Mean difference | Sd of differences | Per seed (5, 6, 7, 8, 9) |
|---|---|---|---|
| YOLOv8x | +0.0042 | 0.0180 | -0.0038, -0.0030, +0.0209, -0.0177, +0.0246 |
| RT-DETR-l | -0.0005 | 0.0015 | +0.0009, -0.0025, -0.0006, +0.0011, -0.0015 |
| Faster R-CNN | -0.0023 | 0.0043 | -0.0050, -0.0046, +0.0027, +0.0017, -0.0066 |

- **RT-DETR-l** agrees across backends to within 0.25 points on every seed.
- **Faster R-CNN** differs by at most 0.7 points, with mixed signs.
- **YOLOv8x** is the loosest: single-seed gaps reach 2.5 points, in both directions. The swings come from the missed-tooth rate (below), not from classification of found teeth: matched-only top-1 is 0.9506-0.9660 on CUDA and 0.9527-0.9659 on MPS.
- For all three, the mean backend difference is smaller than the seed-to-seed spread within a backend.

## Per-seed results (all-GT / missed / matched-only)

**YOLOv8x**

| Seed | CUDA | MPS | MPS − CUDA |
|---|---|---|---|
| 0 | 0.9410 / 1.55% / 0.9558 | - | - |
| 1 | 0.9339 / 1.68% / 0.9499 | - | - |
| 2 | 0.9500 / 2.12% / 0.9706 | - | - |
| 3 | 0.9528 / 1.00% / 0.9624 | - | - |
| 4 | 0.9352 / 1.97% / 0.9539 | - | - |
| 5 | 0.9547 / 1.17% / 0.9660 | 0.9509 / 1.49% / 0.9653 | -0.0038 |
| 6 | 0.9497 / 1.24% / 0.9616 | 0.9467 / 1.54% / 0.9615 | -0.0030 |
| 7 | 0.9184 / 3.99% / 0.9566 | 0.9393 / 2.29% / 0.9614 | +0.0209 |
| 8 | 0.9542 / 1.20% / 0.9658 | 0.9365 / 3.04% / 0.9659 | -0.0177 |
| 9 | 0.9132 / 3.93% / 0.9506 | 0.9378 / 1.57% / 0.9527 | +0.0246 |
| 10 | 0.9366 / 1.89% / 0.9546 | - | - |
| 11 | 0.9352 / 2.58% / 0.9600 | - | - |
| 12-14 | not run | - | - |

**RT-DETR-l**

| Seed | CUDA | MPS | MPS − CUDA |
|---|---|---|---|
| 0 | 0.9501 / 1.13% / 0.9610 | - | - |
| 1 | 0.9345 / 1.12% / 0.9451 | - | - |
| 2 | 0.9512 / 1.27% / 0.9634 | - | - |
| 3 | 0.9481 / 1.00% / 0.9577 | - | - |
| 4 | 0.9402 / 1.04% / 0.9501 | - | - |
| 5 | 0.9578 / 0.77% / 0.9652 | 0.9586 / 0.84% / 0.9668 | +0.0009 |
| 6 | 0.9520 / 0.67% / 0.9584 | 0.9495 / 0.83% / 0.9575 | -0.0025 |
| 7 | 0.9489 / 1.13% / 0.9598 | 0.9483 / 1.31% / 0.9609 | -0.0006 |
| 8 | 0.9612 / 0.87% / 0.9697 | 0.9624 / 0.70% / 0.9692 | +0.0011 |
| 9 | 0.9461 / 0.66% / 0.9524 | 0.9446 / 0.89% / 0.9531 | -0.0015 |
| 10 | MISSING: Kaggle run canceled at the 12 h timeout, no output; rerun pending | - | - |
| 11 | 0.9484 / 0.98% / 0.9578 | - | - |
| 12-14 | not run | - | - |

**Faster R-CNN**

| Seed | CUDA | MPS | MPS − CUDA |
|---|---|---|---|
| 0 | 0.9383 / 1.24% / 0.9500 | - | - |
| 1 | 0.9212 / 1.65% / 0.9366 | - | - |
| 2 | 0.9392 / 1.61% / 0.9546 | - | - |
| 3 | 0.9425 / 0.90% / 0.9510 | - | - |
| 4 | 0.9249 / 1.43% / 0.9383 | - | - |
| 5 | 0.9519 / 0.86% / 0.9602 | 0.9470 / 1.18% / 0.9583 | -0.0050 |
| 6 | 0.9381 / 0.97% / 0.9473 | 0.9335 / 1.22% / 0.9450 | -0.0046 |
| 7 | 0.9380 / 1.47% / 0.9519 | 0.9407 / 1.37% / 0.9538 | +0.0027 |
| 8 | 0.9441 / 1.22% / 0.9558 | 0.9458 / 1.12% / 0.9566 | +0.0017 |
| 9 | 0.9279 / 1.02% / 0.9375 | 0.9213 / 1.17% / 0.9322 | -0.0066 |
| 10 | 0.9224 / 2.01% / 0.9413 | - | - |
| 11 | 0.9406 / 1.50% / 0.9550 | - | - |
| 12-14 | not run | - | - |

## What this does not show

- **No equivalence claim.** Five seed pairs per detector cannot rule out a small backend effect, and the paired standard deviations (0.15 to 1.8 points) are larger than the mean differences.
- **Backend is confounded with software stack and batch size.** The CUDA and MPS runs used different torch/torchvision/ultralytics builds and kernels. On MPS, YOLOv8x ran at batch 2 and RT-DETR-l at batch 4 (ultralytics accumulates to a nominal batch of 64), so batch-norm statistics differ from the CUDA runs. This is a disclosed confound and was not varied.
- **CUDA seeds 0-4, 5-9 and 10-14 are different splits.** Differences between those groups (for example RT-DETR-l 0.9448 vs 0.9532) are not a backend or time effect; they are different test sets and different seeds.
- **Groups are not pooled across backends.** The "all CUDA seeds" line pools one backend only.
- **Incomplete coverage.** CUDA seeds 12-14 have not been run for any detector, and RT-DETR-l seed 10 timed out on Kaggle (12 h) without output. Seeds are reported as they finish, none dropped.
- **Seed column label bug.** The Kaggle Faster R-CNN CUDA seeds 5-9 outputs were written by an older script that hard-coded `seed=0` in `summary.csv`. Results here are keyed by folder name, and each total of labeled teeth (`n_test` + missed) matches the MPS run on the same split.

## Limitation: missed teeth (Faster R-CNN)

About 1.3% of labeled teeth get no predicted box at the matching IoU threshold, and they count as errors in all-GT top-1 (703 of 54,722 across the ten seeds 0-9 analysed: 374/27,569 = 1.36% on CUDA seeds 0-4, 329/27,153 = 1.21% on MPS seeds 5-9). `missed_tooth_analysis.py` breaks them down; the pattern is the same on both backends:

- **Box size is the main driver.** The smallest quarter of boxes miss at 2.9% and account for 57% of all misses, versus 0.64% for the middle half and 0.94% for the largest quarter (CUDA 3.2% / MPS 2.6% for the smallest quarter).
- **Neighbor overlap does not matter.** Miss rates are 1.1-1.4% whether a tooth's box has no overlap, light overlap or heavy overlap with its nearest labeled neighbor.
- **Image edge:** teeth within 3% of a border are rarely labeled (37 of 54,722) but are missed often (15 of them, 41%). They contribute only 2% of all misses, so this does not move the headline number.
- **No single tooth dominates.** The most-missed FDI classes (15, 23, 25, 31, 32, 12, 26, 45) each hold 5-6% of misses at 2-2.4% per class, with no third-molar concentration. The top classes differ between the CUDA and MPS groups, so the class-level ordering is noise.

This analysis covers Faster R-CNN seeds 0-9 only. It does not yet include CUDA seeds 5-14 for any detector.

The Faster R-CNN detector was not retuned (resolution, anchors, score threshold) so that all seeds share one configuration. Per-tooth rows for the MPS seeds 5-9 come from `per_tooth_predictions_mps.py`, which reruns inference on the saved `best.pt` files and checks its totals against each seed's `summary.csv`.

## Missed teeth: YOLOv8x and RT-DETR-l (MPS seeds 5-9)

`missed_tooth_analysis_yolo_rtdetr.py` runs the same breakdown on YOLOv8x and RT-DETR-l, using per-tooth predictions from `per_tooth_predictions_ultralytics_mps.py` (same rerun-and-verify protocol as the Faster R-CNN script). This only covers the MPS seeds; the CUDA seeds 5-14 outputs are summary-only, so the two CUDA spikes noted below cannot be broken down the same way.

| Detector | Seeds | Missed | Total | Rate |
|---|---|---|---|---|
| YOLOv8x | MPS 5-9 | 534 | 27,153 | 1.97% |
| RT-DETR-l | MPS 5-9 | 247 | 27,153 | 0.91% |

**Per seed, YOLOv8x MPS:** 1.49%, 1.54%, 2.29%, **3.04% (seed 8)**, 1.57%.
**Per seed, RT-DETR-l MPS:** 0.84%, 0.83%, 1.31%, 0.70%, 0.89%.

The general pattern matches Faster R-CNN's: small boxes drive most misses (YOLO's smallest quarter misses at 3.80% vs 1.35-1.36% for the rest; RT-DETR's at 1.80% vs 0.46-0.91%), edge proximity is a small, rare category, and no single tooth dominates (YOLO's worst class, FDI 23, holds 9.0% of misses at 5.21%; RT-DETR's worst, FDI 45, holds 8.9% at 2.48%).

**YOLOv8x MPS seed 8, the one spike this data can explain:** its own box-size quartiles are 4.71% (small), 2.62% (mid), 2.20% (large) - all three roughly double the pooled MPS rate for that quartile, not just the small one. So seed 8 is not simply "more small teeth"; the whole split is harder for detection. Its top missed classes are canines and premolars (FDI 23, 28, 41, 14, 13, 22 at 5-9% each), not third molars. No further cause (image quality, crowding) was checked.

**Still unexplained:** the two YOLOv8x CUDA spikes, seeds 7 and 9 (about 4% each, from `summary.csv` alone - no per-tooth breakdown exists for the CUDA runs). They occur on different seeds than the MPS spike (seed 8), so this is not one bad split replicated across backends; each backend has its own hard seed(s). A CUDA per-tooth rerun would need the saved Kaggle checkpoints, which were not downloaded (only `summary.csv`).
