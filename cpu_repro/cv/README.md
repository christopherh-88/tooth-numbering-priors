# Uncropped five-fold cross-validation (the MICCAI paper's experiments)

Everything here runs on Kaggle (T4 GPU for training and image
interventions, CPU for scoring). Each experiment's decision rule was
written, approved and frozen in a `*_RULES.md` file before it ran;
`RESULTS.md` (repo root) has every result with its section number.

## Data and splits

- `prepare_uncropped_cv.py`: boxes from the tooth masks (`boxes.csv`) and
  the grouped, stratified five-fold splits (`folds.csv`, `folds_seed1.csv`,
  `folds_seed2.csv`).
- `train_cv.py`: trains YOLOv8x, RT-DETR-l or Faster R-CNN per fold and
  saves test detections. Options: `--cv-seed` (split), `--train-seed`,
  `--augment-gaps` (Section 77).
- `kaggle/make_kernels.py` (training) and `kaggle/make_phase3_kernels.py`
  (analysis) write the Kaggle kernels.

## Scoring

- `score_cv.py`: confidence-first matching, top-1, gap over the
  position-only model, X-ray bootstrap CIs, joint failures (Sections 60, 65).
- `seed1_grade.py`: compares a split or training seed with split 0
  (Sections 70, 75, 78, 83).
- `calib_sweep.py`, `label_check.py`: cutoffs, calibration and label
  checks (Sections 61, 62, 64).
- `batch3.py`: predicted-box baseline, arch-order fix, review flag, power
  (Section 66).

## Mechanism

| Script | Question | Rule | Section |
|---|---|---|---|
| `shift_test.py` | Does moving the whole image change answers? | PHASE3_RULES.md, PHASE3_BATCH2_RULES.md | 63, 67 |
| `context_test.py` | Do detectors need the surrounding teeth? | PHASE3_BATCH2_RULES.md | 68 |
| `gap_check.py` | Do shared errors sit next to missing teeth? | PHASE3_BATCH3_RULES.md | 69 |
| `gap_intervention.py`, `gap_intervention_overlap.py` | Does erasing a tooth cause them? | GAP_INTERVENTION_RULES.md | 71 |
| `drift_check.py` | Are they at closed gaps? | DRIFT_RULES.md | 72 |
| `closure_intervention.py` | Does closing a gap cause them (slide, tip)? | CLOSURE_INTERVENTION_RULES.md, TIPPING_RULES.md | 76, 79 to 82 |
| `tip_vs_slide.py` | Tip against slide on the same teeth | TIPPING_RULES.md | 80, 82 |
| `gap_augment_grade.py` | Does training on simulated gaps help? | GAP_AUGMENT_RULES.md | 77 |
| `gap_flag.py` | Can the detector's own output flag the shared errors? | GAP_FLAG_RULES.md | 84 |
| `dentex_external.py` | Do the UFBA models' results carry over to DENTEX? | DENTEX_EXTERNAL_RULES.md | 85 |
| `dentex_sensitivity.py` | DENTEX thresholds and framing (reported only) | DENTEX_EXTERNAL_RULES.md | 85 |
| `dentex_prepare.py`, `train_cv.py --dataset` | Train the three detectors within DENTEX | DENTEX_INDOMAIN_RULES.md | 86 |
| `gap_strata.py` | Gap ratio by number of missing teeth (reported only) | GAP_STRATA_RULES.md | 87 |
| `train_cv.py --dataset --train-seed 1` | DENTEX second training seed | DENTEX_TSEED_RULES.md | pending (GPU quota) |
| `dentist_sheet.py` | Blinded expert sheet (built, not used) | LABEL_CHECK_RULES.md | 65 |

## Results

`results/kaggle_*/` holds the CSVs pulled from each Kaggle kernel (only
CSV and example PNG outputs; models stay on Kaggle).
