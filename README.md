# Do tooth-numbering models read anatomy or position?

Measuring the cost of engineered spatial priors on anomalous dentition.

## The question

Panoramic-radiograph tooth-numbering models place every tooth in a highly
regular, near-fixed spatial layout (32 slots, FDI numbering, quadrants
arranged the same way in almost every image). That regularity is a
detection-friendly shortcut: a model could in principle number a tooth from
where its bounding box sits in the image, largely without reading the
tooth's own anatomy. If so, the model would be expected to fail exactly on
the clinically interesting cases - supernumerary or ectopically-positioned
teeth - where position and identity come apart.

This project measures that directly, in two parts:

- **Claim A - geometry alone predicts identity.** A coordinate-only model
  (box position/size/aspect ratio, no image content at all) predicts FDI
  tooth identity at 67-72% top-1 (32-way) vs. a 3.6% majority baseline,
  replicated across two independent panoramic-radiograph datasets
  (UFBA-425, DENTEX) with 5-seed confidence intervals and a pre-stated
  falsification threshold. Well-supported.
- **Claim B - do real detectors actually use that shortcut?** Trained
  detectors are compared against the same coordinate-only baseline on
  identical splits. Across three architecturally distinct detectors
  (YOLOv8x, RT-DETR-l, Faster R-CNN - CNN single-stage, transformer
  anchor-free, and CNN two-stage/region-proposal respectively, spanning
  two independent training pipelines), real detectors substantially
  outperform the coordinate-only baseline and converge on the same
  answer: largely no, real detectors do not rely on position as their
  primary signal - with a weak-but-real residual error correlation as the
  qualifier (detectors that fail together tend to fail toward the
  position-predicted answer). See `paper/DRAFT.md` and `RESULTS.md` for
  the full numbers, caveats, and what would falsify this.

## Where to look

This repo is a research log, not a packaged library - the honest, current
state of the work lives in a few files, not in this README:

- **`RESULTS.md`** - every number produced in this project, with the
  script and split that produced it. The authoritative source; if a
  number here ever looks stale, `RESULTS.md` wins.
- **`HANDOFF.md`** - where things stand right now, what's done, what's
  next, and standing process rules for this repo (review diffs before
  committing, grep for AI-tool attribution before staging, etc.).
- **`paper/DRAFT.md`** - the in-progress MICCAI submission draft and its
  supporting notes.
- **`ENVIRONMENT.md`** - environment setup; this project runs two deep
  learning frameworks at once (TensorFlow for the U-Net segmentation
  model, PyTorch/Ultralytics for detection) with a real numpy-version
  conflict between them - read this before installing anything.
- **`cpu_repro/`** - the actual experiment code: `coord_baseline/` (the
  coordinate-only model and its README), `yolo_training/` (YOLOv8x,
  RT-DETR-l and Faster R-CNN training/eval, including
  `BACKEND_COMPARISON.md` for the Kaggle-CUDA-vs-local-MPS robustness
  check), `dual_labeled_dataset/`, `boundary_condition/`, `anomaly_scan/`,
  each with its own README where relevant. `requirements.txt` for this
  code lives at `cpu_repro/requirements.txt`.

## Dataset and upstream attribution

This repo is a fork of
[devichand579/Instance_seg_teeth](https://github.com/devichand579/Instance_seg_teeth)
and depends directly on that project's dataset and reference pipeline:
the [UFBA-425](https://figshare.com/articles/dataset/UFBA-425/29827475)
dataset (425 human-annotated panoramic X-rays, bounding boxes + polygons,
FDI numbering - a subset of the UFBA-UESC Dental Dataset) and the
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
later revision was posted 2025-07-02 - see `paper/DRAFT.md`'s References
section for the citation-year note.)

Licensed under Apache 2.0 (`LICENSE`), inherited from the upstream
project.
