# Paper draft (skeleton)

**Status:** structural draft only. Every number cited below is pulled from
`RESULTS.md` (cite the section, don't retype/round further). Anything not
yet measured is marked `[PENDING - see RESULTS.md Section 11.5]` and must
not be filled in with an assumed or estimated value - replace only once
the real script has been run and the number is in `RESULTS.md`.

## Working title

"Geometry Alone Predicts Tooth Identity: A Diagnostic Baseline for
Shortcut Risk in FDI Tooth Numbering on Panoramic Radiographs"

(placeholder - revisit once Claim B evidence exists, since a confirmed
Claim B result would justify a stronger title, e.g. replacing "Risk" with
a direct claim.)

## Abstract (draft)

Automated tooth numbering models trained on panoramic radiographs report
high accuracy, but accuracy alone cannot distinguish genuine appearance
understanding from reliance on incidental positional structure in the
data. We introduce a coordinate-only diagnostic baseline - a classifier
trained on nothing but bounding-box geometry (center, size, aspect ratio)
- as a cheap pre-registration-style sanity check that should be run before
crediting a tooth-numbering model's accuracy to image understanding. On
two independent panoramic radiograph datasets (UFBA-425, DENTEX), geometry
alone predicts FDI tooth identity at 67-72% top-1 accuracy (32-way
classification) against a 3.6% majority-class baseline, with a shuffled-
geometry negative control collapsing to majority-baseline as expected.
[PENDING: one sentence on Claim B result once measured - does a real
trained detector's error pattern correlate with the coordinate-only
model's error pattern]. [PENDING: one sentence on mitigation result once
measured]. We release the diagnostic baseline, its negative controls, and
a pre-registered falsification threshold as a reusable audit tool for
tooth-numbering research.

## 1. Introduction

- Motivate with the shortcut-learning framing from Geirhos et al. (2020,
  Nature Machine Intelligence) - models can achieve high in-distribution
  accuracy by exploiting easy, predictive-but-non-causal features.
- Ground in real medical-imaging precedent: DeGrave et al. (2021, Nature
  Machine Intelligence) on COVID chest X-ray shortcuts (hospital-specific
  markers/positioning); Winkler et al. (2019, JAMA Dermatology) on
  surgical skin markings collapsing melanoma classifier specificity from
  84.1% to 45.8%.
- State the specific question for tooth numbering: FDI numbering is a
  small (32-class), spatially-structured label space applied to a highly
  canonicalized acquisition protocol (panoramic radiographs) - exactly the
  precondition under which a position-based shortcut would be expected to
  work, per the two-precondition hypothesis (Section 11.4,
  `RESULTS.md`).
- State the contribution as a **diagnostic tool**, not a claim about any
  specific detector's internals (framing decision, `RESULTS.md`
  Section 11.1) - avoids overclaiming while the tool itself is the
  contribution.

## 2. Related work

**Shortcut learning, general.** Geirhos et al. (2020) - the general
definition and taxonomy of shortcut learning used throughout this paper.

**Shortcut learning, medical imaging.** DeGrave, Janizek & Lee (2021) -
COVID-19 chest X-ray models relying on confounders rather than pathology,
across multiple public datasets, demonstrated with explainable-AI
techniques. Winkler et al. (2019) - surgical skin markings as a shortcut
in melanoma classification, with a directly measured specificity collapse.
Lin et al. (2024, MICCAI) - "Shortcut Learning in Medical Image
Segmentation": burnt-in clinical annotations and center-crop/zero-padding
artifacts as segmentation shortcuts, with proposed and evaluated
mitigations (random cropping, annotation removal) - closest published
structural comparable to this work; unlike this paper, they measure the
shortcut inside a real trained model and evaluate a fix, which is the gap
this paper still needs to close (`RESULTS.md` Section 11.5).

**Dental/panoramic radiograph AI and dataset bias.** DENTEX (Hamamci et
al., 2023, MICCAI challenge) and its HierarchicalDet baseline - the
tooth-numbering benchmark this work's second dataset is drawn from. Zhou
et al. (2024, BMC Oral Health), "Combining public datasets for automated
tooth assessment in panoramic radiographs" - documents annotation-quality
bias across combined public dental datasets (a different bias axis than
the geometric one studied here; also the source of the Dual-Labeled
Dataset used in this project's Task 3 verification). [Add the
panoramic-radiograph-quality/detection-error correlation paper (PMC,
2025) if it strengthens the "multiple known confounds in this exact
domain" point in the intro.]

**Novelty positioning.** A citation-graph check (not a keyword web
search) of papers citing DENTEX, HierarchicalDet, and three shortcut-
learning-in-medical-imaging papers found 0 of 132 unique citing papers
combining tooth-numbering/FDI-enumeration language with shortcut/
position-bias language in title or abstract - stated here as a genuine
checked negative result, not an assumed gap. [Cite the three seed papers
by full title/author once finalized for the reference list.]

## 3. Methods

Point to, don't restate, the existing method sections:
- Coordinate-only baseline construction, features, classifiers, splits:
  `RESULTS.md` Section 2, `cpu_repro/coord_baseline/build_coord_baseline.py`.
- Negative controls (shuffle / position-only / size-only):
  `RESULTS.md` Section 4, `cpu_repro/coord_baseline/controls/`.
- Cross-dataset replication (DENTEX): `RESULTS.md` Section 10.
- Falsification threshold (pre-stated, not pre-registered before the
  first experiment - state this limitation explicitly, don't soften it):
  `RESULTS.md` Section 2 / `cpu_repro/coord_baseline/README.md`.
- Numbering-convention invariance check (Universal/Palmer):
  `RESULTS.md` Section 11.4.
- [PENDING] Error-pattern correlation method (Claim B):
  `RESULTS.md` Section 11.5, once run.
- [PENDING] Mitigation method: `cpu_repro/coord_baseline/mitigation/README.md`,
  once run.
- [PENDING] Boundary-condition dataset method: target list in
  `RESULTS.md` Section 11.5, once a dataset is chosen and run.
- Statistical testing: paired permutation tests,
  `cpu_repro/coord_baseline/significance_tests.py` (prepared, not yet
  run - see that file's docstring), supplementing the 5-seed mean ± 95%
  CI reporting already in `RESULTS.md`.

## 4. Results

Structure to mirror `RESULTS.md`'s existing numbered sections once the
paper is actually drafted in full - do not retype numbers here, cite the
section and pull at write-time so a stale copy can't drift from the
source of truth.

- 4.1 Coordinate-only baseline recovers FDI identity far above chance
  (Section 2, Section 10).
- 4.2 Negative controls confirm the signal is genuine, not an artifact
  (Section 4).
- 4.3 Quadrant vs. tooth-type: the non-trivial part of the finding
  (Section 11.3) - foreground this distinction explicitly, it is the
  paper's sharpest empirical point, not a footnote.
- 4.4 Numbering-convention invariance (Section 11.4) - a robustness check,
  not a generalization result; state this distinction clearly so it isn't
  misread as broader evidence than it is.
- 4.5 [PENDING] Error-pattern correlation with a real detector (Claim B).
- 4.6 [PENDING] Mitigation result.
- 4.7 [PENDING] Boundary-condition dataset result.

## 5. Discussion

- Explicitly separate Claim A (well-supported: geometry alone predicts
  identity) from Claim B (real detectors exploit this) per
  `RESULTS.md` Section 11.2 - state plainly, once Claim B is measured,
  which of the two the paper is actually claiming, rather than letting
  the two blur together in prose.
- Generalization boundary as a falsifiable hypothesis (Section 11.4):
  state what would need to be true elsewhere (non-panoramic modality, or
  a differently-structured label space) for the finding to transfer, and
  report the boundary-condition dataset result against that prediction
  once available, whichever direction it comes out.
- Clinical stakes: a coordinate-shortcutting detector would silently
  mislabel exactly the cases most likely to be clinically flagged -
  supernumerary or ectopically-positioned teeth, whose position deviates
  from the canonical FDI slot the shortcut relies on. Tie this to the
  UFBA-425 annotation-anomaly scan (Section 5), the Dual-Labeled Dataset
  supernumerary existence-proof (`RESULTS.md` Section 16), and the
  follow-up accuracy comparison (Section 17, run 2026-09-08). **State
  Section 17's result honestly, don't oversell it:** direction matches
  the hypothesis (accuracy near supernumerary teeth is lower - 29.5% vs.
  33.0% per-instance, 28.8% vs. 32.3% per-image) but neither test reaches
  significance (p=0.077 per-instance, p=0.36 per-image, the more
  appropriate test given image-level clustering) - report this as
  suggestive-but-inconclusive in the discussion, explicitly flagging that
  n=23 supernumerary-present images is underpowered, not as confirmed
  evidence. **A follow-up attempt to sharpen this (localized adjacency to
  the supernumerary tooth) produced an apparently dramatic, highly
  significant result that turned out to be a class-composition confound
  (supernumerary teeth cluster anatomically in the anterior maxilla, so
  their "nearest teeth" are disproportionately drawn from classes that
  are already easy/hard regardless of proximity) - withdrawn, not
  reportable as evidence either way. If this section is drafted from
  memory rather than re-reading `RESULTS.md` Section 17, do not
  accidentally resurrect the withdrawn 55.6%-near-accuracy number.**
  Controlling for teeth-count/crowding doesn't change the (still
  non-significant) whole-image conclusion either. Section 17 does have
  two genuine bonus findings worth using regardless of the supernumerary
  result: (1) Claim A transfers to this third, independent dataset (32.9%
  vs. 3.76% majority baseline), roughly half the in-domain accuracy -
  worth citing as further cross-dataset replication of Claim A alongside
  DENTEX (Section 10); (2) images with more visible teeth are easier for
  the coordinate-only model (r=0.2556, p<0.0001) - plausibly because a
  fuller arch looks more like a canonical/complete layout.

## 6. Limitations

- Falsification threshold was stated after the initial experiment, not
  pre-registered before it (state this as-is, do not soften).
- Two datasets, both panoramic FDI - generalization untested beyond that
  until the boundary-condition dataset is run.
- [PENDING, if unresolved by submission] No real detector evaluated - if
  this is still open at submission time, this becomes the single most
  important line in the limitations section, not a minor caveat.

## 7. Ethics / data statement

- UFBA-425, DENTEX: publicly released research datasets, already
  de-identified panoramic radiographs - state license/terms as documented
  by each dataset's own release.
- Dual-Labeled Dataset (Zhou et al., 2024): note its partial-release
  status (500 of 2,000 stated images) if used in the final paper, and
  the phash/filename overlap check against UFBA-425 (0 matches found)
  if used to justify treating it as an independent dataset.

## References (to finalize)

- Geirhos et al., "Shortcut Learning in Deep Neural Networks," Nature
  Machine Intelligence, 2020.
- DeGrave, Janizek & Lee, "AI for radiographic COVID-19 detection selects
  shortcuts over signal," Nature Machine Intelligence, 2021.
- Winkler et al., "Association Between Surgical Skin Markings in
  Dermoscopic Images and Diagnostic Performance of a Deep Learning CNN
  for Melanoma Recognition," JAMA Dermatology, 2019.
- Lin et al., "Shortcut Learning in Medical Image Segmentation," MICCAI
  2024 (LNCS 15008, pp. 623-633).
- Hamamci et al., "DENTEX: An Abnormal Tooth Detection with Dental
  Enumeration and Diagnosis Benchmark for Panoramic X-rays," 2023
  (arXiv:2305.19112).
- Zhou et al., "Combining public datasets for automated tooth assessment
  in panoramic radiographs," BMC Oral Health, 2024.
- [Add exact citations for the three Task-1 seed papers once resolved
  IDs are double-checked against `resolved_target_ids.json` in the
  scratch directory from that run.]
