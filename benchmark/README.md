# Position-only benchmark for tooth numbering (UFBA-425)

A box's position on a panoramic X-ray already predicts its FDI number for
78.5% of teeth. This benchmark scores a tooth-numbering detector against
that baseline on the same teeth and the same cross-validation folds. It
also reports results on the teeth where three different detectors and
the position-only model all fail together.

## Files

- `teeth.csv`: one row per labeled tooth (11,602 teeth, 425 uncropped
  X-rays). Columns:
  - `image_id`, `category`, `fdi`, `class_id` (0 to 31 for FDI 11 to 18,
    21 to 28, 31 to 38, 41 to 48)
  - `x_center`, `y_center`, `width`, `height`: the labeled box, as
    fractions of the image size
  - `fold_seed0`, `fold_seed1`: test fold under the two 5-fold splits
    (grouped by X-ray, stratified by category)
  - `next_to_gap`: an arch neighbor of this tooth has no label in the
    X-ray (third molars count)
  - `position_only_pred_seed0`, `position_only_pred_seed1`: the
    out-of-fold answer of a gradient-boosted model on box position and
    shape only (6 features)
  - `yolov8x_pred`, `rtdetr_l_pred`, `fasterrcnn_pred`: reference
    detectors, split 0, empty if the tooth was missed
  - `joint_failure`, `shared_wrong_class`: all three reference detectors
    and the position-only model are wrong with the same answer (161 teeth)
- `score.py`: scores your detections (format in its docstring).
- `build_benchmark.py`: rebuilds `teeth.csv` from the repo's results.

## Protocol

1. Train one model per fold of a split (`fold_seed0` or `fold_seed1`),
   never on that fold's X-rays. The splits and inner validation sets are
   in `cpu_repro/cv/folds.csv` and `folds_seed1.csv`.
2. Run each model on its test fold and write all detections to one CSV.
3. `python benchmark/score.py --pred your.csv --split 0`

Matching: confidence at least 0.5, IoU at least 0.5, one-to-one, highest
confidence first, class-agnostic. A labeled tooth with no match counts as
wrong. CIs come from 10,000 bootstrap draws of X-rays.

## Reference results (split 0)

| | top-1 | gap over position-only (pp) | top-1 next to a gap | top-1 on joint failures |
|---|---|---|---|---|
| YOLOv8x | 94.98 | 16.49 (15.09 to 17.89) | 79.99 | 0 |
| RT-DETR-l | 95.16 | 16.66 (15.25 to 18.06) | 80.72 | 0 |
| Faster R-CNN | 92.82 | 14.33 (12.97 to 15.69) | 75.51 | 0 |
| Position-only | 78.50 | | | 0 |

Top-1 on the joint failures is 0 for these models by construction. For a
new model it shows whether it escapes the failure the others share, and
`repeats_shared` shows whether it falls into the same wrong answer.

`false_pos_per_xray` and `f1_numbered` need the raw detections, which are
not in this table for the reference detectors (they were scored from
matched answers only).

## Limits

One dataset (UFBA-425, Budagam et al. 2025, doi 10.6084/m9.figshare.29827475;
cite it as the repo README shows), one annotation per tooth, no
dentist re-check yet. The joint-failure set comes from split 0 models and
is not refit for split 1.
