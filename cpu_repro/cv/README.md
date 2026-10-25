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
| `train_cv.py --dataset --train-seed 1` | DENTEX second training seed | DENTEX_TSEED_RULES.md | 88 |
| `dentex_tseed_compare.py` | Seed 0 against seed 1, pooled, overlap (reported only) | DENTEX_TSEED_RULES.md | 88 |
| `closure_intervention.py --dataset dentex` | Closing a gap and tipping a tooth in DENTEX-trained detectors (slide and tip, training seeds 0 and 1) | DENTEX_INTERVENTION_RULES.md | 90, 91 |
| `closure_intervention.py --dataset dentex --work-width 0`; third training seed | The same at the X-ray's stored size (92); a third training seed, scored and with its slide and tip runs (93) | DENTEX_FOLLOWUP_RULES.md | 92, 93 |
| `ap_eval.py` | COCO-style AP of every detector set (appendix, reported only) | AP_RULES.md | 89 |
| `dentist_sheet.py` | Blinded expert sheet (built, not used) | LABEL_CHECK_RULES.md | 65 |
| `missing_teeth.py` | Missing teeth per X-ray in UFBA and DENTEX (reported only) | none | 94 |
| `post_hoc_two_detector.py` | The stored-size increase with YOLOv8x and RT-DETR-l only (post hoc, reported only) | none | 95 |
| `fold_overlap.py` | How much the three splits' test folds overlap (reported only) | none | 96 |
| `ap_vs_pycocotools.py` | Check of `ap_eval.py` against pycocotools on synthetic detections | none | 97 |
| `gap_rank.py`, `gap_rank_power.py` | Does ordering the Section 84 flag by estimated space help? (rules 1 and 2 met on seed 0, rule 3 not met); the power simulation that set its thresholds | GAP_RANK_RULES.md | 98 |

## Results

`results/kaggle_*/` holds the CSVs pulled from each Kaggle kernel (only
CSV and example PNG outputs; models stay on Kaggle).
