# Paper figures

Built by `make_figures.py` from committed result CSVs only; rerun with
`python paper/figures/make_figures.py` (repo environment, `ENVIRONMENT.md`).
Output is deterministic: the PDFs are byte-identical across runs. Each figure
is a vector PDF with embedded TrueType fonts (no Type 3) plus a 600-dpi PNG,
sized to fit the LNCS/MICCAI text width (122 mm). Colours are from the
Okabe-Ito colour-blind-safe palette; the coordinate-only prior is always grey.

Unless stated otherwise, small dots are individual seeds and large markers
are the mean with a 95% t-interval over seeds (`mean_ci95` in
`cpu_repro/coord_baseline/build_coord_baseline.py`, as in `RESULTS.md`).
"All labeled teeth" top-1 counts a tooth the detector did not find as wrong.
The script prints every plotted summary value; the numbers below are copied
from that output.

## Draft captions

**Fig. 1. Geometry alone predicts FDI tooth identity on panoramic
radiographs.** (a) Top-1 accuracy (32 classes) of a classifier that sees
only each box's centre, width, height, area and aspect ratio, over 5 seeds
of image-level train/test splits. Gradient-boosted tree: UFBA-425 69.5%
(95% CI ±0.8), DENTEX 68.5% (±1.2), DenPAR periapical 26.6% (±2.8).
Dashed lines: majority-class baseline (3.6%, 3.75%, 7.6%). (b) The same
gradient-boosted tree split into quadrant accuracy and tooth-type accuracy
(position 1-8 within the quadrant). On panoramic images, quadrant is nearly
solved (96.5%, 98.0%) and tooth type is the harder part (72.0%, 70.1%); on
periapical images both fall to about 45%. Sources: `RESULTS.md` Sections 2,
10, 25.

**Fig. 2. The geometric signal is position, mostly horizontal, and it
degrades smoothly with box noise.** (a) Gradient-boosted tree top-1 on
UFBA-425 with feature subsets (5 seeds, bars are means with 95% CIs).
Removing x costs 47.5 pp and removing y 16.8 pp; removing any single shape
feature changes accuracy by at most 0.3 pp. Shuffling the box geometry among
the teeth of each image drops accuracy to 3.2%, below the 3.6%
majority-class baseline. (b) Top-1 when Gaussian noise is added to test-box
coordinates (training boxes clean). Grey band: the mean natural per-class
spread of box centres (s.d. 0.027 horizontally to 0.049 vertically, as a
fraction of the image). Sources: Sections 4, 13, 19, 20.

**Fig. 3. Three trained detectors outperform the coordinate-only prior on
the same test teeth.** CUDA seeds 0-4; each seed evaluates the prior and
all three detectors on the identical held-out teeth (4,981-5,916 per seed).
(a) Top-1 over all labeled teeth: prior 69.5%, YOLOv8x 94.3%, RT-DETR-l
94.5%, Faster R-CNN 93.3%. (b) Detector minus prior: 24.8 pp (95% CI ±0.6),
25.0 pp (±0.8), 23.8 pp (±0.9), against the pre-registered decision
thresholds (`GO_NO_GO.md`; COMMIT ≤5 pp would indicate reliance on the
shortcut, PIVOT ≥15 pp evidence against). (c) Correlation (φ) between each
detector's errors and the prior's errors: 0.184, 0.187, 0.180. Sources:
Sections 33, 40, 45.

**Fig. 4. Every tooth class favours the detector.** Per-FDI-class
difference in top-1 (detector minus prior, percentage points) for CUDA seeds
0-4, with 95% paired-bootstrap intervals (2,000 resamples of whole
(seed, image) clusters). All 96 detector-class intervals lie above zero;
differences range from 7.2 to 38.5 pp. The smallest gains are the molars,
where the prior is already strongest (e.g. YOLOv8x, FDI 38: +7.2 pp,
95% CI +3.5 to +10.9). Intervals are descriptive: seeds can share test
images. Source: Section 50.

**Fig. 5. When all three detectors fail on the same tooth, they fail
alike, and toward the position-predicted answer.** Each point is one seed
(CUDA seeds 0-4 and 11-14; MPS seeds 5-9; seed 10 has no three-detector
table). Backends are kept separate. (a) Among teeth all three detectors got
wrong, the share where all three gave the same wrong class. (b) Among wrong
answers shared by a detector pair, the share that are adjacent-tooth
confusions (pooled over the three pairs). (c) Among the teeth in (a) with a
shared wrong class, the share where that class is also the coordinate-only
prior's prediction. Means: (a) CUDA 97.3%, MPS 97.4%; (b) 92.4%, 92.4%;
(c) 86.1%, 87.5%. This analysis was not pre-registered. Sources: Sections
46, 51, 52.

**Fig. 6. Training backend changes how often YOLOv8x misses teeth, not how
well found teeth are numbered.** Seeds 5-9 trained on Kaggle CUDA (T4) and
on Apple Silicon (MPS), same split per seed; dashed line is equality.
(a) Top-1 over all labeled teeth: MPS and CUDA differ by at most 0.25 pp
for RT-DETR-l, 0.66 pp for Faster R-CNN and 2.46 pp for YOLOv8x. (b) The
YOLOv8x differences come from the missed-tooth rate (up to 2.36 pp apart).
(c) Top-1 on matched teeth only stays within 0.53 pp for every detector.
Sources: Sections 49, 53; `cpu_repro/yolo_training/BACKEND_COMPARISON.md`.

**Fig. S1. Per-tooth comparison in all three seed groups.** As Fig. 4, for
(a) CUDA seeds 0-4, (b) MPS seeds 5-9 and (c) CUDA seeds 11-14, analysed
separately. All 96 detector-class intervals lie above zero in every group;
the lowest interval bound is +0.7 pp (RT-DETR-l, FDI 38, CUDA seeds 11-14).
Source: Section 54.

## Data behind each figure

| Figure | Files |
|---|---|
| 1 | `cpu_repro/coord_baseline/per_seed_results.csv`, `cpu_repro/coord_baseline/dentex/per_seed_results.csv`, `cpu_repro/boundary_condition/denpar_periapical/per_seed_results.csv` |
| 2 | `cpu_repro/coord_baseline/feature_ablation.csv`, `controls/summary.csv`, `noise_robustness_summary.csv`, `positional_variance_by_class.csv` |
| 3 | `cpu_repro/yolo_training/eval_results/{multiseed,rtdetr_multiseed,fasterrcnn_multiseed}/*_summary.csv` |
| 4, S1 | `cpu_repro/yolo_training/eval_results/per_tooth_head_to_head.csv` |
| 5 | `cpu_repro/yolo_training/eval_results/cross_architecture_agreement_summary.csv` |
| 6 | `cpu_repro/yolo_training/eval_results/detector_backend_comparison.csv` |
