# CLAIM check

Checked against the Checklist for Artificial Intelligence in Medical
Imaging, 2024 update (CLAIM; Tejani et al., Radiology: Artificial
Intelligence 2024; 44 items). Item numbers and wording follow the list as
printed in the open-access PMC copy (PMC11304031), read on 2026-10-09.
Status: done, partial, gap, or n/a. Statuses were refreshed on 2026-10-09,
after the DENTEX experiments (RESULTS.md Sections 85 to 87).

## Title, abstract, introduction

| # | Item | Status | Where / what is missing |
|---|---|---|---|
| 1 | Identification as a study of AI methodology, with the technology category | done | Title and abstract name detectors for tooth numbering |
| 2 | Summary of design, methods, results and conclusions | done | Abstract in `paper/miccai/DRAFT.md` and `latex/main.tex` |
| 3 | Scientific and clinical background, intended use and role of the AI approach | done | Introduction. The tooth-numbering detectors are the AI approach; the study asks whether they read position |
| 4 | Study aims, objectives, hypotheses | done | Every experiment has a rule file frozen before running (`cpu_repro/cv/*_RULES.md`) |

## Methods: study design and data

| # | Item | Status | Where / what is missing |
|---|---|---|---|
| 5 | Prospective or retrospective | done | Retrospective analysis of public datasets |
| 6 | Study goal | done | Introduction |
| 7 | Data sources | partial | UFBA-425 (Budagam et al. 2025, figshare) and DENTEX (Hamamci et al. 2023, Kaggle copy). Collection dates and sites come from the dataset papers and are not restated |
| 8 | Inclusion and exclusion criteria | done | UFBA: all 425 uncropped originals; Roboflow copies cropped 0 to 20% excluded (Section 59); masks with no pixels dropped. DENTEX: the 634 quadrant-enumeration X-rays (`dentex_prepare.py`) |
| 9 | Data preprocessing | done | Boxes from tooth masks (`prepare_uncropped_cv.py`); images at 512 x 512 as released |
| 10 | Selection of data subsets | done | Dataset categories kept as a stratification variable in the folds |
| 11 | De-identification | n/a | Public, already de-identified (done by the dataset authors) |
| 12 | Missing data | done | Missing teeth are part of the question (Sections 69 to 71, 87), not imputed |
| 13 | Image acquisition protocol | partial | Panoramic X-rays; scanner and settings are in the dataset papers and are not restated |

## Methods: reference standard

| # | Item | Status | Where / what is missing |
|---|---|---|---|
| 14 | Definition of the method used to obtain the reference standard | done | FDI number from the dataset's per-tooth masks (UFBA) or labels (DENTEX) |
| 15 | Rationale for the reference standard | done | Only reference available; standard for this task |
| 16 | Source of reference standard annotations (who, qualifications) | partial | From the dataset papers; not restated |
| 17 | Annotation of the test set | partial | Same annotators and tool as training data; not restated |
| 18 | Inter- and intrarater variability | gap | Single annotation. Proxy: confident-learning check flags 10.0% of joint failures (Sections 61, 65), and the position match holds without them. An expert re-read was dropped; stated as a limitation |

## Methods: partitions, model, training

| # | Item | Status | Where / what is missing |
|---|---|---|---|
| 19 | How data were assigned to partitions | done | 5-fold CV grouped by X-ray, stratified by category; 15% inner validation inside each training fold; the test fold is never used for selection |
| 20 | Level at which partitions are disjoint | done | By X-ray. Uncropped originals only, so no crop of a test X-ray is in training |
| 21 | Intended sample size | partial | The public datasets set it (425 and 634 X-rays). Minimum detectable effects are in Section 66 D; no sample size was chosen in advance |
| 22 | Detailed description of the model | done | YOLOv8x and RT-DETR-l (Ultralytics), Faster R-CNN ResNet-50 FPN v2 (torchvision); position-only HistGradientBoosting on 6 box features |
| 23 | Software libraries, frameworks, packages | done | `ENVIRONMENT.md` (Ultralytics 8.4.143, torch 2.14.0); kernel builders in `cpu_repro/cv/kaggle/`; Kaggle T4 |
| 24 | Initialization of model parameters | done | COCO-pretrained (`yolov8x.pt`, `rtdetr-l.pt`, torchvision default weights); stated in the Models paragraph of `latex/main.tex` |
| 25 | Details of the training approach | partial | Paper states 30 epochs, 640 pixels, flips off, Faster R-CNN SGD with a rate scaled to batch 2. Not restated: Faster R-CNN rate 0.0025 and steps at epochs 18 and 25, and the Ultralytics defaults (all in `train_cv.py`) |
| 26 | Method of selecting the final model | done | Epoch with the best inner-validation fitness (0.1 x mAP50 + 0.9 x mAP50-95), same rule for all three |
| 27 | Ensembling | n/a | None |

## Methods: evaluation

| # | Item | Status | Where / what is missing |
|---|---|---|---|
| 28 | Metrics of model performance | done | `paper/miccai/METRICS_RELOADED.md`, including the one departure from its recommendation (no AP) |
| 29 | Statistical measures of significance and uncertainty | done | X-ray bootstrap (10,000), paired differences, permutation nulls, minimum detectable effects (Section 66 D) |
| 30 | Robustness or sensitivity analysis | done | Matching cutoffs and order (62, 64, 65), image shift (63, 67), context masking (68), two further splits and a second training seed (75, 78, 83), gap closure and tipping interventions with replications and a paste control (76, 79 to 82) |
| 31 | Explainability or interpretability | done | Interventions instead of saliency maps: shift, masking, tooth erasure, closure, tipping (63, 67, 68, 71, 76, 80) |
| 32 | Evaluation on internal data | done | Pooled five-fold test folds; Sections 65, 70 |
| 33 | Testing on external data | partial | DENTEX: UFBA-trained models run unchanged (Section 85), detectors trained within DENTEX on two training seeds (Sections 86, 88), and the slide and tip interventions on DENTEX-trained detectors (Sections 90, 91). Partial replication: 1 of 3 observational bars met in Section 86; of the interventions, the slide meets its bar on both seeds (11.8 and 12.0% against UFBA's 20.8%) and the tip meets it on one (4.8 and 5.5%, bar 5%) |
| 34 | Clinical trial registration | n/a | Not a trial. Frozen rule files with dates in the repo; no external registry |

## Results

| # | Item | Status | Where / what is missing |
|---|---|---|---|
| 35 | Numbers of patients or examinations included and excluded | partial | Counts exist (425 X-rays, 11,602 teeth; 634 X-rays, 18,095 teeth); no flow diagram |
| 36 | Demographic and clinical characteristics per partition and dataset | gap | Not in the released data. Category counts and missing teeth are reported (Section 87); say so in limitations |
| 37 | Performance metrics and measures of statistical uncertainty | done | Sections 65, 70, 86, with CIs |
| 38 | Estimates of diagnostic performance and their precision | n/a | No diagnostic or patient-level claim; numbering accuracy per tooth is item 37 |
| 39 | Failure analysis of incorrect results | done | Joint failures, gaps, erasure, closure (65, 69, 71, 76); missed against misnumbered |

## Discussion and other information

| # | Item | Status | Where / what is missing |
|---|---|---|---|
| 40 | Study limitations | done | Limitations paragraph in `latex/main.tex`: single annotation, no demographics, DENTEX partial replication, 2D interventions |
| 41 | Implications for practice, intended use or clinical role | partial | Discussion: report a position-only baseline; check the numbering next to missing teeth (a rule from outside the model). No claim about clinical use or a deployed tool |
| 42 | Reference to the full study protocol or technical details | done | `RESULTS.md` and the frozen `*_RULES.md` files in the repo; anonymized release for review |
| 43 | Availability of software, trained model, data | partial | Code and per-tooth table in the repo and `benchmark/`; anonymized link pending; trained weights stay on Kaggle and are not in the release; Zenodo DOI on hold |
| 44 | Sources of funding, role of funders | gap | For the authors to fill (camera-ready) |
