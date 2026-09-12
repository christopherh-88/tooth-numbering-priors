# Paper draft (skeleton)

**Status:** structural draft only. Every number cited below is pulled from
`RESULTS.md` (cite the section, don't retype/round further). Anything not
yet measured is marked `[PENDING - see RESULTS.md Section 11.5]` and must
not be filled in with an assumed or estimated value - replace only once
the real script has been run and the number is in `RESULTS.md`.

## Working title

"Geometry Alone Predicts Tooth Identity: A Diagnostic Baseline for
Shortcut Risk in FDI Tooth Numbering on Panoramic Radiographs"

(placeholder - Claim B evidence now exists (`RESULTS.md` Sections 21-33)
and points the other way from what this note originally anticipated: the
real detector does *not* show shortcut-reliant behavior (it exceeds the
coordinate-only ceiling by ~25pp, concentrated in tooth-type accuracy,
and a mitigation experiment removing position/scale jitter produced no
measurable accuracy change - Sections 30/31). This does not justify
replacing "Risk" with a stronger positive claim about shortcut exploitation;
if anything it reinforces the diagnostic-tool framing as the paper's
actual contribution (`RESULTS.md` Section 24) rather than a "detector X
was gaming position" result. Title kept as-is; revisit only if the
Discussion section's framing decision changes.)

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
A real trained detector (YOLOv8, identical split) exceeds the
coordinate-only ceiling by 24.8 percentage points on average across 5
independent seeds (95.9% vs. 69.5% top-1, concentrated almost entirely in
tooth-type rather than quadrant accuracy) - a pattern inconsistent with
reliance on the geometric shortcut, though its errors do correlate with
the coordinate-only model's errors weakly more than chance (phi = 0.18,
5-seed 95% CI [0.13, 0.23]). A geometry-jitter mitigation experiment -
training the same detector with position/scale augmentation removed -
produced no statistically detectable change in accuracy (paired 95% CI
on the difference spans zero), giving no evidence the detector had been
exploiting position as a shortcut to begin with. We release the diagnostic baseline, its negative controls, and
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
structural comparable to this work. This paper closes the same structural
gap they identify (measure the shortcut inside a real trained model, then
evaluate a fix): Sections 21-33 (`RESULTS.md`) do so for tooth
numbering, with a different qualitative outcome - here the real detector
does not show shortcut-reliant behavior, so the "fix" (Sections 30/31)
is better read as a robustness check than a correction.

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

Notably, Hamamci et al. themselves flag the exact gap this paper's Task 2
is designed to close: their own Limitations section (V-C, "Evaluation
Metric Constraints") states that aggregate AP "doesn't explicitly
distinguish the type of error," that "understanding the prevalence of
specific error types, such as swapping the enumeration of adjacent
teeth, is crucial," and calls for "detailed error analysis, such as
confusion matrix for the enumeration classes" as future work. A targeted
literature check (`RESULTS.md` Section 18) confirmed no public
per-FDI-class confusion matrix exists for any DENTEX-trained detector,
including the follow-up participant papers (DentexSegAndDet, YOLOrtho,
DETDet) - only aggregate AP/AR per task is reported anywhere in this
line of work. This is cited here as the motivating gap for Task 2, not
just DENTEX-as-a-dataset.

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
- Real-detector training and evaluation (YOLOv8, GO/NO-GO protocol):
  `RESULTS.md` Section 21, `cpu_repro/yolo_training/train_yolo.py`,
  `cpu_repro/yolo_training/GO_NO_GO.md` for the pre-registered
  COMMIT/PIVOT/EXTEND thresholds.
- Error-pattern correlation method (Claim B): `RESULTS.md` Sections 22
  (error-type taxonomy comparison), 23 (case study), 26 (2x2
  agreement/phi-coefficient analysis, `error_correlation_analysis.py`).
- Mitigation method (geometry-jitter augmentation, executed):
  `cpu_repro/coord_baseline/mitigation/README.md` (design),
  `RESULTS.md` Sections 30-31 (pre-committed paired-analysis method and
  result, `cpu_repro/yolo_training/train_yolo_zerojitter.py`,
  `mitigation_analysis.py`).
- Boundary-condition dataset method: DenPAR periapical radiographs
  (Rasnayaka et al., *Scientific Data* 12:1615, 2025), chosen as the
  first real test of the two-precondition hypothesis - `RESULTS.md`
  Section 25, `cpu_repro/boundary_condition/denpar_periapical/`.
- Multi-seed replication (Claim B headline numbers, 5 seeds total):
  `RESULTS.md` Sections 32 (scoping) and 33 (result),
  `cpu_repro/yolo_training/train_yolo_seed{1,2,3,4}.py`,
  `multiseed_analysis.py`.
- Statistical testing: paired permutation tests,
  `cpu_repro/coord_baseline/significance_tests.py` (prepared, not yet
  run - see that file's docstring), supplementing the 5-seed mean ± 95%
  CI reporting already in `RESULTS.md` (image-level bootstrap
  resampling for the YOLO/DenPAR results, `mean_ci95`'s t-interval for
  cross-seed results - both documented at each point of use).

## 4. Results

Structure to mirror `RESULTS.md`'s existing numbered sections once the
paper is actually drafted in full - do not retype numbers here, cite the
section and pull at write-time so a stale copy can't drift from the
source of truth.

- 4.1 Coordinate-only baseline recovers FDI identity far above chance
  (Section 2, Section 10).
- 4.1a Feature ablation: the signal is asymmetrically position-dependent
  - `x_center` dominates (dropping it costs 47pp), `y_center` is real
  but secondary (dropping it costs 17pp, well above any shape feature's
  <0.3pp), shape features are inert (Section 20) - sharpens Claim A from
  "geometry predicts identity" to "asymmetrically position-dependent
  identity prediction," a cleaner mechanistic claim than "x_center
  only." Also motivates the mitigation experiment's choice to jitter
  position, not size/shape.
- 4.1b Robustness to test-time coordinate noise (Section 19): accuracy
  degrades smoothly, not as a cliff, and stays well above the majority
  baseline even at noise levels exceeding the natural per-class spread -
  supports reading Section 21/22's result (no shortcut-consistent error
  pattern from the real detector) as genuine, not an artifact of
  detector localization imprecision washing the signal out.
- 4.2 Negative controls support that the signal is genuine, not an
  artifact (Section 4).
- 4.3 Quadrant vs. tooth-type: the non-trivial part of the finding
  (Section 11.3) - foreground this distinction explicitly, it is the
  paper's sharpest empirical point, not a footnote.
- 4.4 Numbering-convention invariance (Section 11.4) - a robustness check,
  not a generalization result; state this distinction clearly so it isn't
  misread as broader evidence than it is.
- 4.5 A real trained detector does not show shortcut-reliant behavior
  (Section 21's GO/NO-GO result: PIVOT, YOLOv8 exceeds the
  coordinate-only ceiling by 26.3pp at seed 0, replicated at 24.8pp
  mean across 5 seeds - Section 33). The gap concentrates almost
  entirely in tooth-type accuracy (Sections 21/27), which a
  position-only signal cannot reach per the feature-ablation ceiling
  (4.1a) - the central piece of evidence against Claim B's "real
  detectors exploit this" reading. A weak-but-nonzero error correlation
  with the coordinate-only model does exist (phi = 0.16-0.18 depending
  on seed, Sections 26/33) and should be reported honestly alongside the
  gap, not omitted because it complicates a clean "no" - see Section
  24's discussion of how the two facts coexist.
- 4.6 Mitigation result: training the same detector with the
  geometry-jitter augmentation that (unintentionally) already existed
  in Section 21's run removed (translate=0/scale=0 vs. 0.1/0.5) produces
  no statistically detectable accuracy change (paired bootstrap 95% CI
  on the top-1 delta spans zero - Section 31). Reported as a clean null,
  not an inconclusive result: further evidence the detector was not
  relying on position as a shortcut, since removing an augmentation that
  would suppress such reliance had nothing to suppress.
- 4.7 Boundary-condition dataset result: on DenPAR periapical
  radiographs - where the canonicalized-acquisition-protocol
  precondition breaks while the structured-label-space precondition
  holds - the coordinate-only shortcut collapses from ~69-70%
  (panoramic) to ~27% top-1, and quadrant accuracy specifically drops
  from near-ceiling (96-98%) to 45.5%, barely above tooth-type accuracy
  on the same data (Section 25). A partial, quantified answer supporting
  the two-precondition hypothesis, not a clean falsification or
  confirmation - stated as such.
- 4.8 Multi-seed replication: the single-seed fragility concern raised
  across external review answered directly - 5 independent seeds give a
  top-1 accuracy 95.85% ± 1.01pp and a gap vs. coordinate-only of
  24.77pp ± 0.61pp (95% CI [24.16, 25.38]), nowhere near the
  pre-registered 5pp GO/PIVOT threshold on any individual seed
  (Section 33). The framing decision (4.5, Section 24) is not reopened
  by this replication.

## 5. Discussion

- Explicitly separate Claim A (well-supported: geometry alone predicts
  identity) from Claim B (real detectors exploit this - measured,
  answer is largely no, with a weak-but-real error correlation as the
  qualifier) per `RESULTS.md` Section 11.2/24 - state plainly which of
  the two the paper is actually claiming, rather than letting the two
  blur together in prose.
- Generalization boundary as a falsifiable hypothesis (Section 11.4):
  state what would need to be true elsewhere (non-panoramic modality, or
  a differently-structured label space) for the finding to transfer.
  DenPAR periapical radiographs (Section 25) is the one boundary-condition
  dataset run so far - the shortcut weakens sharply (69-70% -> ~27%
  top-1, quadrant accuracy collapsing from near-ceiling to 45.5%) but
  does not vanish to chance, a partial result supporting the
  two-precondition hypothesis without fully confirming or falsifying it.
  CBCT slices and a differently-structured label space (Section 11.5)
  remain untried, open extensions for future work, not required to
  support this paper's current claims.
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
- Two datasets, both panoramic FDI, for Claim A's main evidence; the
  boundary-condition test (DenPAR, Section 25) is a single periapical
  dataset with a substantially smaller sample (2505 instances vs.
  27,563/4,872) and its own residual label-quality noise (Arch/Site
  spreadsheet inconsistencies in ~12-35 of 633 used images, checked and
  found not to materially move the headline numbers - Section 25's
  materiality-check addition) - one data point supporting the
  two-precondition hypothesis, not a broad generalization sweep.
- Real detector evaluated (YOLOv8, 5 seeds - Sections 21/33), resolving
  what had been the paper's single biggest open item. Neither this
  detector nor the coordinate-only baseline had its hyperparameters
  tuned or searched within this project (Section 28) - both used
  un-searched configurations (sklearn defaults for the baseline,
  values copied from a pre-existing notebook for YOLO), so the exact
  magnitude of the ~25pp gap should not be read as a precisely
  calibrated number, though the asymmetry does not bias the paper's
  central *relative* claim in either direction (Section 28's disclosure
  decision).
- **Multiplicity / analysis transparency.** A number of exploratory
  checks were run across this project beyond the headline results
  (significance tests, a detectability power check, a geometric-ceiling
  check, and a supernumerary error-rate follow-up with several
  sub-analyses). State plainly that `RESULTS.md` is an append-only, dated
  log of every analysis run, including ones that didn't pan out - e.g. a
  localized-adjacency refinement of the supernumerary check
  (`RESULTS.md` Section 17) initially looked like a strong, significant
  result but was identified as a class-composition confound and is
  recorded as withdrawn rather than quietly dropped. This is offered as
  evidence against selective reporting, not as a claim that every
  possible analysis was pre-registered - it wasn't (see the falsification
  threshold caveat above).

## 7. Ethics / data statement

- UFBA-425, DENTEX: publicly released research datasets, already
  de-identified panoramic radiographs - state license/terms as documented
  by each dataset's own release.
- Dual-Labeled Dataset (Zhou et al., 2024): note its partial-release
  status (500 of 2,000 stated images) if used in the final paper, and
  the phash/filename overlap check against UFBA-425 (0 matches found)
  if used to justify treating it as an independent dataset.
- DenPAR (Rasnayaka et al., 2025): CC BY 4.0, downloaded directly from
  Zenodo with checksum verification against the published record
  (`RESULTS.md` Section 25) - attribution required per license terms if
  used in the final paper.

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
- Rasnayaka et al., *Scientific Data* 12:1615, 2025 (DenPAR dataset -
  [VERIFY EXACT TITLE before submission, not confirmed anywhere in this
  repo; RESULTS.md Section 25 records author/journal/volume/year only,
  via the Zenodo record at 10.5281/zenodo.16645076, not the paper title
  itself]).
- [Add exact citations for the three Task-1 seed papers once resolved
  IDs are double-checked against `resolved_target_ids.json` in the
  scratch directory from that run.]
