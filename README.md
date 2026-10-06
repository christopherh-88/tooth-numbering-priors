# Do tooth-numbering models read anatomy or position?

AI models that number teeth on panoramic dental X-rays score well, but a
panoramic X-ray is taken with the head held in a fixed position, so every
tooth tends to land in the same part of the image. A model could learn to
guess a tooth's FDI number from where its box sits instead of from what the
tooth looks like. If it did, it would fail on the cases that matter
clinically, such as extra (supernumerary) teeth or teeth that have drifted
out of place. This repository tests that in two parts.

**Claim A: box geometry alone predicts tooth identity.** A gradient-boosted
tree that sees only each box's center, width, height, area and aspect ratio,
and no pixels, names the tooth correctly 69.5% of the time on UFBA-425 and
68.5% on DENTEX (32 classes, 5 random splits), against a majority-class
baseline of 3.6 to 3.7%. It gets the quadrant right 96.5 to 98.0% of the
time. Shuffling the boxes among the teeth of each image drops it to chance.

**Claim B: trained detectors mostly do not rely on that shortcut.** YOLOv8x,
RT-DETR-l and Faster R-CNN, trained on the same splits, reach 93.3 to 94.5%
on the same teeth, 24 to 25 percentage points above the geometry-only
model, and beat it on all 32 tooth classes. Position still shows up in the
roughly 2% of teeth that every model gets wrong: there the three detectors
pick the same wrong tooth about 97% of the time, and that tooth matches the
geometry-only guess 84 to 88% of the time (chance: 4 to 11%). On periapical
X-rays (DenPAR), which lack the fixed framing, the geometry-only model falls
to 26.6%.

**Current numbers (uncropped five-fold cross-validation, used in the
MICCAI paper).** Most test images in the splits above were randomly cropped
copies, which weaken the position cue (RESULTS.md Section 59). On the 425
original X-rays with cross-validation grouped by X-ray, the geometry-only
model numbers 78.5% of teeth, and the detectors beat it by 14 to 17 points
on all 32 tooth numbers, replicated on three splits (Sections 65, 75, 78).
The errors all three detectors share (1.4% of teeth) match the
geometry-only answer 91% of the time and sit next to missing teeth whose
neighbors have closed the space; closing or tipping a neighbor into an
erased tooth's space on purpose reproduces them (Sections 69 to 82).

## Where to look

This is a research log, not a packaged library.

- **`paper/cjsj/`**: the short paper submitted to the Columbia Junior
  Science Journal, with `paper_numbers.py` (recomputes every number in it
  from committed results and fails on any mismatch) and
  `make_cjsj_figures.py` (builds its four figures).
- **`RESULTS.md`**: every result in the project, in order, with the script
  and split behind it. If a number anywhere else disagrees, this file wins.
- **`HANDOFF.md`**: current status and next steps.
- **`paper/miccai/`**: the MICCAI 2027 paper (`DRAFT.md`, the LNCS
  version in `latex/`, figures, and the Metrics Reloaded and CLAIM
  checklists). It uses the uncropped five-fold cross-validation in
  `cpu_repro/cv/` (RESULTS.md Sections 60 onward).
- **`benchmark/`**: the per-tooth table, splits and scorer for comparing a
  tooth-numbering model with the position-only baseline.
- **`paper/DRAFT.md`** and **`paper/figures/`**: the earlier long draft on
  the original (partly cropped) splits, kept for reference.
- **`ENVIRONMENT.md`**: setup. The project uses TensorFlow (U-Net) and
  PyTorch/Ultralytics (detectors) in one environment, and the install
  order matters. Read it before installing anything.
- **`cpu_repro/`**: the experiment code. `coord_baseline/` is the
  geometry-only model, its controls and the DENTEX replication;
  `yolo_training/` trains and evaluates the three detectors and holds their
  per-seed results; `boundary_condition/` is the DenPAR test;
  `dual_labeled_dataset/` is the supernumerary-tooth check;
  `anomaly_scan/` is annotation QA. Most folders have their own README.
- **`notebooks/`**: the upstream OralBBNet notebooks (see below), with the
  flip-augmentation bug fixed.

## Quick start

```bash
# Setup (see ENVIRONMENT.md for why the order matters)
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0 -c cpu_repro/requirements.txt
pip install -r cpu_repro/requirements.txt

# Rebuild the gitignored per-seed data folders, then rerun the main analyses
python cpu_repro/yolo_training/prepare_all_splits.py
python cpu_repro/coord_baseline/build_coord_baseline.py
python cpu_repro/yolo_training/cross_architecture_agreement.py

# Check every number in the CJSJ paper and rebuild its figures
python paper/cjsj/paper_numbers.py
python paper/cjsj/make_cjsj_figures.py

# Check the flip augmentation in the U-Net notebooks
python cpu_repro/coord_baseline/mitigation/verify_flip_augment.py
```

The detector training scripts in `cpu_repro/yolo_training/` need a GPU and
were run on Kaggle (Tesla T4) and an Apple M3; their outputs are committed,
so every analysis above runs on a CPU.

## A training pitfall worth knowing about

FDI numbers encode left and right, so a horizontally flipped X-ray puts
tooth 11 where tooth 21 belongs. Most augmentation code flips the image
without swapping the labels, which mislabels left and right teeth on every
flipped sample. The detectors here train with flipping turned off. The
upstream U-Net notebooks flipped the image, the per-tooth masks and the
per-tooth box maps without swapping their tooth channels; all four
notebooks now swap each channel with its mirror tooth's channel
(`RESULTS.md` Section 58).

## Dataset and upstream attribution

This repo is a fork of
[devichand579/Instance_seg_teeth](https://github.com/devichand579/Instance_seg_teeth)
and depends directly on that project's dataset and reference pipeline:
the [UFBA-425](https://figshare.com/articles/dataset/UFBA-425/29827475)
dataset (425 human-annotated panoramic X-rays, bounding boxes + polygons,
FDI numbering, a subset of the UFBA-UESC Dental Dataset) and the
[OralBBNet](https://arxiv.org/abs/2406.03747) paper and its training
notebooks (`notebooks/`), which this project's own detector runs
(`cpu_repro/yolo_training/`) build on. See the upstream repository for the
original project's own results tables, notebooks, and dataset category
breakdown.

If you use the dataset, cite:
```bibtex
@article{Budagam2025,
author = "Devichand Budagam and Azamat Zhanatuly Imanbayev and Iskander Rafailovich Akhmetov and Aleksandr Sinitca and Sergey Antonov and Dmitrii Kaplun",
title = "{UFBA-425}",
year = "2025",
month = "8",
url = "https://figshare.com/articles/dataset/UFBA-425/29827475",
doi = "10.6084/m9.figshare.29827475.v1"
}
```

If you use or reference the OralBBNet method, cite:
```bibtex
@misc{budagam2025oralbbnetspatiallyguideddental,
      title={OralBBNet: Spatially Guided Dental Segmentation of Panoramic X-Rays with Bounding Box Priors},
      author={Devichand Budagam and Azamat Zhanatuly Imanbayev and Iskander Rafailovich Akhmetov and Aleksandr Sinitca and Sergey Antonov and Dmitrii Kaplun},
      year={2024},
      eprint={2406.03747},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.03747},
}
```
(`year` above reflects the paper's original arXiv posting, 2024-06-06; a
later revision was posted 2025-07-02. See `paper/DRAFT.md`'s References
section for the citation-year note.)

Licensed under Apache 2.0 (`LICENSE`), inherited from the upstream
project.
