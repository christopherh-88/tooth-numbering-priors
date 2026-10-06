# CLAIM check

Checked against the Checklist for Artificial Intelligence in Medical
Imaging (CLAIM; Mongan et al. 2020, updated by Tejani et al. 2024). Items
are grouped by the checklist's sections and paraphrased from memory, so
the item numbers are left out; match each row to the official 2024 list
before submission. Status: done, partial, gap, or n/a.

## Title, abstract, introduction

| Item | Status | Where / what is missing |
|---|---|---|
| Says it is an AI study and names the task | done | Draft title in `paper/miccai/DRAFT.md` |
| Structured abstract with design, data, results and limits | partial | Draft in `paper/miccai/DRAFT.md` |
| Clinical or scientific background and the question | done | `paper/DRAFT.md` introduction; question: does box position alone predict FDI number, and do detectors share position's errors |
| Study goals and hypotheses stated before testing | done | Every experiment has a rule file frozen before running (`cpu_repro/cv/*_RULES.md`) |

## Methods: design and data

| Item | Status | Where / what is missing |
|---|---|---|
| Retrospective or prospective; study design | done | Retrospective analysis of a public dataset |
| Data source, collection dates, sites | partial | UFBA-425 (Budagam et al. 2025, figshare). Collection dates, site and scanner come from the dataset paper and are not restated yet |
| Inclusion and exclusion | done | All 425 uncropped originals. Roboflow copies randomly cropped 0 to 20% are excluded (Section 59). Masks with no pixels are dropped when building boxes |
| Preprocessing | done | Boxes from tooth masks (`prepare_uncropped_cv.py`); images at 512 x 512 as released |
| Data subsets and why | done | Dataset categories kept as a stratification variable in the folds |
| De-identification | n/a | Public, already de-identified |
| Missing data | done | Missing teeth are part of the question (Sections 69 to 71), not imputed |
| Demographics (age, sex) | gap | Not in the released data; say so in limitations |

## Methods: ground truth

| Item | Status | Where / what is missing |
|---|---|---|
| Definition of the reference standard | done | FDI number from the dataset's per-tooth masks |
| Rationale for it | done | Only reference available; standard for this task |
| Annotators' number and qualifications | partial | From the dataset paper; not restated |
| Annotation tool and instructions | partial | Same |
| Inter- or intra-rater variability | gap | Single annotation. Proxy: confident-learning check flags 10.0% of joint failures (Sections 61, 65), and the position match holds without them. An expert re-read was dropped; state it as a limitation |

## Methods: data partitions

| Item | Status | Where / what is missing |
|---|---|---|
| How data were split and at what level | done | 5-fold CV grouped by X-ray, stratified by category; 15% inner validation inside each training fold for model selection; the test fold is never used for selection |
| Number of images and teeth per partition | done | Fold tables in Sections 60, 65, 70 |
| Independence between partitions | done | Grouping by X-ray; uncropped originals only, so no crop of a test X-ray is in training |
| Second split | done | Seed 1 replication (Section 70) |

## Methods: models and training

| Item | Status | Where / what is missing |
|---|---|---|
| Model description, inputs, outputs | done | YOLOv8x and RT-DETR-l (Ultralytics 8.4.143), Faster R-CNN ResNet-50 FPN v2 (torchvision); position-only HistGradientBoosting on 6 box features |
| Software, versions, hardware | done | `ENVIRONMENT.md`, kernel builders in `cpu_repro/cv/kaggle/`; Kaggle T4 |
| Initialization (pretrained weights) | partial | COCO-pretrained (`yolov8x.pt`, `rtdetr-l.pt`, torchvision default weights for Faster R-CNN); state it in the paper |
| Training details: epochs, optimizer, learning rate, augmentation, stopping | partial | In `train_cv.py`: 30 epochs, horizontal flip off (a flip would swap left and right FDI numbers). Faster R-CNN: SGD, learning rate scaled to batch 2 (0.0025; the earlier non-CV runs used the batch-16 rate, Section 58), steps at epochs 18 and 25 |
| Model selection criterion | done | Epoch with the best inner-validation fitness (0.1 x mAP50 + 0.9 x mAP50-95, Ultralytics' rule, used for all three) |
| Ensembling | n/a | None |

## Methods: evaluation

| Item | Status | Where / what is missing |
|---|---|---|
| Metrics and why | done | `paper/miccai/METRICS_RELOADED.md` |
| Statistical methods, CIs, significance | done | X-ray bootstrap (10,000), paired differences, permutation nulls, minimum detectable effects (Section 66 D) |
| Robustness or sensitivity analysis | done | Matching cutoffs and order (62, 64, 65), image shift (63, 67), context masking (68), second split (70) |
| Explainability or interpretation | done | Interventions instead of saliency maps: shift, masking, tooth erasure (63, 67, 68, 71) |
| External validation | gap | DENTEX not used in this phase. State it as a limit |
| Benchmark or comparison with other work | partial | `benchmark/` releases the per-tooth table and scorer; no comparison with published numbering systems |

## Results

| Item | Status | Where / what is missing |
|---|---|---|
| Flow of images and teeth | partial | Counts exist (425 X-rays, 11,602 teeth); no flow diagram |
| Case characteristics | partial | Category counts, missing teeth (1,364 teeth next to a gap); no demographics |
| Performance with CIs | done | Sections 65, 70 |
| Failure analysis | done | Joint failures, gaps, erasure (65, 69, 71) |
| Per-class results | done | Section 74, `benchmark/per_fdi.csv` |

## Discussion and other information

| Item | Status | Where / what is missing |
|---|---|---|
| Limitations | done | In `paper/miccai/DRAFT.md`: one dataset, single annotation, no external set, no demographics, closure is correlational, one extra seed |
| Implications for practice | partial | The gap check wording in `paper/cjsj/REVISIONS.md` sections 8 to 10 |
| Code availability | done | GitHub repo; Zenodo DOI on hold |
| Data availability | done | Public dataset; per-tooth table in `benchmark/` |
| Funding, conflicts | gap | For the authors to fill |
| Protocol or registration | partial | Frozen rule files with dates in the repo; no external registry |
