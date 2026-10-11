# Pre-registered analysis plan: interventions on DENTEX

Closing a gap, or tipping a tooth into it, in detectors trained within DENTEX.

| | |
|---|---|
| Status | **Frozen 2026-10-10, approved by the user as written, before any kernel was pushed.** |
| Written | 2026-10-10, before any intervention image has been scored |
| Extends | RESULTS.md Sections 76 to 82 (UFBA interventions), 86 to 88 (DENTEX in-domain) |
| Procedures | `CLOSURE_INTERVENTION_RULES.md` (slide), `TIPPING_RULES.md` (tip), unchanged except where this file says |
| Code | `closure_intervention.py --dataset dentex`, `kaggle/make_phase3_kernels.py` (commit recorded in the results section) |
| Results go to | RESULTS.md, a new numbered section; `cpu_repro/cv/results/kaggle_dentex_{closure,tip}_{s0,t1}/` |

## 1. Background and rationale

On UFBA, moving a tooth's distal neighbor into the space of an erased tooth
makes all three detectors give it the missing tooth's number: by sliding,
20.8% (95% CI 18.4 to 23.3) against 0.18% with the space left open
(Section 76); by tipping about its root, 9.4% (7.7 to 11.0) against 0.19%,
with a cut-and-paste-back control at 0.0% (Section 80). The slide
replicates on two further splits (Sections 79, 81) and the tip on one
(Section 82).
These interventions are the paper's evidence that closed gaps cause the
errors the detectors share, rather than merely accompany them.

All of that is one dataset. On DENTEX, a second source with its own
annotation protocol, the paper reports only the observational link: shared
errors sit next to a missing tooth 3.5 and 3.3 times as often as correct
teeth over two training seeds (Sections 86, 88), below UFBA's 5.3 and below
the pre-set bar of 4.2; part of the difference is the patient mix
(Section 87). Whether the interventions have an effect in DENTEX-trained
detectors on DENTEX X-rays is untested, and it is the main open question
about how far the mechanism claim extends.

## 2. Objectives and hypotheses

**Primary (confirmatory, training seed 0 models).**

- **H1 (slide).** Sliding the distal neighbor fully into the erased
  tooth's space raises the share of movers that all three DENTEX-trained
  detectors give the erased tooth's number (j), by an amount whose CI
  excludes 0, to at least 5%.
- **H2 (tip).** The same for tipping the neighbor about its root apex until
  its crown touches the mesial neighbor, with the paste check not
  confounded.

**Secondary (graded, same models).**

- **S1.** j(closed) is at least 18.35%, the lower end of UFBA's seed 0 CI
  for the slide (Section 76).
- **S2.** j(full tip) is at least 7.73%, the lower end of UFBA's seed 0 CI
  for the tip (Section 80).

S1 and S2 ask whether the effect is *as large* as on UFBA. They are not
required for the mechanism: DENTEX's observational ratio is lower, so a
smaller intervention effect is a possible and informative outcome.

**Replication (confirmatory, training seed 1 models, Section 88).** H1 and
H2 repeated on the second set of models, on the same targets.

## 3. Data, models and eligibility

- **Data.** DENTEX quadrant-enumeration training tier: 634 panoramic X-rays,
  18,095 labeled teeth with box, FDI number and a tooth polygon each
  (`cpu_repro/anomaly_scan/dentex_raw/train_quadrant_enumeration.json`;
  images from the public Kaggle copy `truthisneverlinear/dentex-challenge-2023`,
  attached to the kernels). The test folds are those of Sections 85 to 88
  (`dentex_folds.csv`), so each X-ray is processed by the models that did
  not train on it.
- **Models.** YOLOv8x, RT-DETR-l and Faster R-CNN trained within DENTEX
  (Section 86, training seed 0: kernels `tooth-numbering-cv-{yolov8x,
  rtdetr-l,fasterrcnn}-dentex`; Section 88, training seed 1: the same with
  `-tseed1`). No model is retrained or tuned for this study. Detections
  use confidence at least 0.5, IoU at least 0.5, confidence-first one-to-one
  matching, exactly as Section 86.
- **Position-only model.** HistGradientBoosting on the six box features,
  trained on the other four DENTEX folds (Section 86's model).
- **Targets.** Up to 3 teeth T per X-ray, drawn with `numpy.random
  .default_rng(0)` by the Section 71 procedure (`draw_targets`): not a third
  molar, both arch neighbors labeled, a polygon present. The draw gives
  1,902 targets in all 634 X-rays (UFBA: 1,248). It is new; it does not
  pair with any UFBA tooth.
- **Mover.** The distal neighbor N of T (the one farther from the
  midline); M is T's mesial neighbor.
- **Skipped targets** (counted and reported, not replaced): no polygon for
  N or M; N's mask never touches M's within the search range below; T, N
  or M labeled twice in the X-ray (added by amendment 2 in section 13).
- **Scored movers.** N is scored only if all three detectors number it
  right on the intact image (so the question is whether the intervention
  *changes* a correct answer into T's number). This conditioning is the
  one used on UFBA and is applied identically here.

## 4. Interventions

Images and polygons are resized to 640 px wide (aspect kept, area
interpolation), the width the detectors read, so the intact, open, moved
and tipped images are all processed at the size the models see. Polygons
are rasterized at that size. The pixel constants of the UFBA runs are kept
without change: the median tooth (third molars left out) is 23.9 px wide in
DENTEX at 640 px and 24.0 px in UFBA at 512 px, so a pixel constant covers
the same part of a tooth. UFBA's 512-px images are tighter on the
dentition than DENTEX's full panoramic frame, so scaling by image width
would have been wrong.

| Quantity | Value (as UFBA) |
|---|---|
| Mask dilation (ellipse) | 3 px |
| Inpainting | OpenCV Telea, radius 5 px |
| Slide search | 0 to 200 px, 1 px steps, until N's mask touches M's (dilated 1 px) |
| Tip search | 0.5 to 45 degrees, 0.5 degree steps, about the root apex of N |
| Fractions | half = 0.5 of the touching distance or angle; closed or full tip = 1.0 |

**Slide conditions:** open (T erased); half; closed (N shifted all the way
to M). **Tip conditions:** open; repaste (T erased, N cut out and pasted
back unmoved with the same inpainting); half tip; full tip. N's labeled box
moves with it (slide) or becomes the box of the rotated mask (tip); the
position-only model is also run on that box.

## 5. Outcomes and estimands

For scored mover *i* and condition *c*, let g(i, c) = 1 if all three
detectors give N the tooth number of T, 0 otherwise.

- **j(c)** = mean of g(i, c) over scored movers, in percent.
- **Δ_slide** = j(closed) − j(open). **Δ_tip** = j(full tip) − j(open).
- **Paste effect** = j(repaste) − j(open), from the tip run.

All are paired within mover. Secondary descriptive outcomes: j(half),
each detector's own fill rate, the share of movers missed, the same with
the position-only model added.

## 6. Quality control and stopping rules (before any number is read)

| # | Check | Action |
|---|---|---|
| Q1 | Position-only on unmoved boxes equals its stored answer for every mover (must be 1.0000) | The kernel stops at once (hard stop in the code, at the smoke pass). The fix and rerun are logged under section 13 |
| Q2 | 10-X-ray smoke pass completes in the same kernel before the full run | If it fails, the kernel stops before the full run |
| Q3 | Intact answers at working size against the stored full-size answers, on the same teeth (`intact_check.csv`): share identical and top-1 per detector | Reported. If a detector's top-1 at working size differs from its stored top-1 on those teeth by more than 2 points, that is stated next to the grade; nothing is changed or rerun |
| Q4 | Skipped targets | Reported with reasons. If more than 10% of targets are skipped, stated next to the grade |
| Q5 | Saved example images (intact, open, half, closed or tip) viewed | As in Sections 76 and 80; implausible edits are reported, not tuned away |

## 7. Statistical analysis

- **Estimator and uncertainty.** Percentages are pooled over scored
  movers. 95% CIs are percentile intervals from 10,000 bootstrap draws that
  resample X-rays with replacement (the unit of independence, since up to
  three movers share an X-ray and one fold model), paired by mover, with
  `numpy.random.default_rng(0)` (`boot_pair`).
- **Decisions use the CI and the point estimate** as written in section 8.
  No p-values are computed.
- **Multiplicity.** H1 and H2 concern different interventions and are each
  judged on its own; both are reported whatever the outcome, and no
  adjustment is applied. The readings in section 8 use H1 and H2
  together.
- **Precision.** At UFBA's 1,114 scored movers the CI half-widths were
  about 2.5 points (slide) and 1.7 (tip). In DENTEX, 1,603 (seed 0) and
  1,622 (seed 1) of the 1,902 targets have a mover all three detectors
  number right at full size (84 and 85%; UFBA 89%), so expect about 1,600
  scored movers and somewhat tighter intervals. The 5% bar is far above
  them.
- **Missing and skipped data.** Not imputed; skipped targets and unscored
  movers are counted (section 12).

## 8. Decision rules

| Bar | Criterion | Source |
|---|---|---|
| H1 | CI of Δ_slide excludes 0, **and** j(closed) ≥ 5% | Rule 1, `CLOSURE_INTERVENTION_RULES.md` (5% is half of the about 11% per neighbor seen at really closed gaps, Section 72) |
| H2 | CI of Δ_tip excludes 0, **and** j(full tip) ≥ 5%, **and** the paste check is not confounded | Rule 1 and paste check, `TIPPING_RULES.md` |
| S1 | j(closed) ≥ 18.35% (point estimate) | Lower end of UFBA's seed 0 CI, Section 76 |
| S2 | j(full tip) ≥ 7.73% (point estimate) | Lower end of UFBA's seed 0 CI, Section 80 |

**Paste check.** The effect is *confounded by paste artifacts* if
j(repaste) − j(open) has a CI excluding 0 and exceeds one third of the
mode's own effect (Δ_slide for H1, Δ_tip for H2). The repaste condition
comes from the tip run, on the same movers and models, and serves both
modes; the number of movers in common is reported.

**Readings, fixed in advance, and the sentence the paper will carry.**

| Outcome | Reading | Wording in the paper |
|---|---|---|
| H1 and H2 met, S1 and S2 met | The causal link replicates at the UFBA size | "Moving a neighbor into an erased tooth's space also produced the shared error in detectors trained on DENTEX, at the rate seen on UFBA." |
| H1 and H2 met, S1 or S2 not met | The mechanism is present at a smaller size | The same, with the rates and their ratio to UFBA; consistent with DENTEX's lower observational ratio |
| H1 met, H2 not met (or the reverse) | Only one intervention reproduces | Reported as such; the claim is narrowed to the intervention that reproduced |
| H1 not met and H2 not met | The UFBA intervention effect does not reproduce in DENTEX-trained detectors | "The intervention effect was not reproduced on DENTEX", added as a limit of the mechanism claim; the abstract's mechanism sentence is qualified |
| H2 confounded by paste | The tip effect cannot be separated from the paste | Reported as confounded; the slide result stands on its own bar |

## 9. Replication on the seed 1 models

Run on the same targets and mover definition, the seed 1 DENTEX models
and their position-only models. **Replicated** if H1 and H2 hold again and
j(closed) and j(full tip) fall inside the seed 0 DENTEX 95% CIs;
otherwise both are reported with the spread and the claim rests on
whichever the readings above allow. Seed 0 is the primary analysis; seed
1 does not replace it.

## 10. Secondary and exploratory analyses (no grade)

- j(half), to see whether the effect grows with closure (as Section 76).
- Per-detector fill rates in every condition.
- Tip against slide on the movers scored in both (`tip_vs_slide.py`).
- Comparison with UFBA: j(closed) and j(full tip) with CIs next to
  Sections 76 and 80, and the ratio of the DENTEX to the UFBA effect.
- Not planned, because the targets are a fresh draw: any pairing with
  UFBA teeth, or re-weighting to UFBA's mix of missing teeth (Section 87
  did that for the observational ratio only).

## 11. Threats to validity

- A sideways paste or an in-plane tip is not real drift; the paste leaves
  edges that inpainting cannot hide, and the root is rotated with the
  crown. No rotation about the long axis. The masks are polygons on a 2D
  projection, drawn under a protocol different from UFBA's.
- X-rays are resized to 640 px: the size the detectors read, but not the
  stored full-size image (Q3 measures the effect).
- The size bars (S1, S2) take UFBA's CIs, from another dataset with
  another mix of missing teeth; failing them is not failing the mechanism.
- Scored movers are those all three detectors already number right, so
  the effect is conditional on correct intact answers (as on UFBA).
- Two training seeds on one split. Section 83 and Section 88 show that
  about a third of shared-error teeth change with training randomness;
  the seed 1 replication addresses that for the intervention.
- The interventions test whether a closed space *can* produce the error
  in these detectors. They do not show that real closed gaps produce it
  by the same route.

## 12. Reporting commitments

The results section will contain, for each of the four kernels: the flow
of targets (drawn, skipped by reason, movers, scored); the table of j by
condition with CIs; Δ_slide, Δ_tip and the paste effect with CIs; the
grade of H1, H2, S1, S2 and the reading from section 8; Q1 to Q5; and the
saved examples. Every result is reported whether or not it supports the
hypotheses, and no bar, threshold, target draw or analysis is changed
after the first intervention image is scored.

## 13. Execution plan, resources and amendments

- **Kernels** (Kaggle T4, each with a 10-X-ray smoke pass first):
  `tooth-numbering-cv-dentex-closure-s0`, `-tip-s0` (seed 0 models), then
  `-closure-t1`, `-tip-t1` (seed 1 models). Embedded scripts are the files
  in `cpu_repro/cv/`; software as the training kernels (Ultralytics
  8.4.143, torchvision, OpenCV).
- **Cost.** UFBA took 0.75 h (slide) and 0.66 h (tip) for 1,248 targets;
  with 1,902 targets on larger images, an estimate of 1 to 2 h per kernel,
  4 to 8 GPU hours in all, two kernels at a time.
- **Inputs fixed.** Seed 0 models and per-tooth table: Section 86
  (`kaggle_dentex_indomain/dentex_indomain/dentex_teeth.csv`). Seed 1:
  Section 88 (`kaggle_dentex_tseed1/dentex_tseed1/dentex_teeth.csv`).
- **Amendments.** After approval this file changes only by a dated note
  below, made before the affected run, stating what changed and why. A
  fix to code that does not alter a bar or the procedure (for example a
  kernel that fails before scoring) is logged here and the rule stands.

### Approval and amendment log

| Date | Entry |
|---|---|
| 2026-10-10 | Drafted. Pixel constants kept after measuring the median tooth width in both datasets (23.9 px against 24.0 px) |
| 2026-10-10 | Approved as written by the user. Frozen. Kernels pushed after this entry |
| 2026-10-10 | **Amendment 1, before any intervention image was scored.** Both seed 0 kernels stopped at the smoke pass on Q1 (position-only agreement 0.9333, 28 of 30 movers); no GPU time beyond the smoke pass and no number was read. Cause, from a CPU diagnostic kernel (`tooth-numbering-cv-q1diag`): the position-only models had been rebuilt from `dentex_boxes.csv` (values rounded to 6 decimals, another row order), which reproduces 93.6% of the stored answers; rebuilding from the full-precision per-tooth table in its own row order, as `dentex_external.py` did, reproduces them exactly (the 34 residual differences in the diagnostic are 34 of 37 groups of duplicate labels colliding in its lookup key). Fix: the models are now built that way. A code fix only; no bar, threshold, target draw or procedure changed |
| 2026-10-10 | **Amendment 2, same time.** DENTEX has 37 groups of an FDI label given twice in one X-ray (Section 86). For a target T, mover N or mesial neighbor M with a repeated label, the tooth's box is ambiguous, so such targets are skipped and counted as a fourth skip reason ("labeled twice in the X-ray"), reported with the others in section 12. The target draw is unchanged (the skip is applied after it), so no tooth is redrawn; UFBA has no repeated labels, so the UFBA code path is unaffected |

### Outcome (2026-10-11)

Seed 0 (primary): RESULTS.md Section 90. Slide: H1 met (j(closed) 11.81%),
S1 not met. Tip: j(full tip) 4.80%, 0.2 points under the 5% bar, so H2 not
met; S2 not met. Reading from section 8: "H1 met, H2 not met".

Seed 1 (section 9): RESULTS.md Section 91. Slide: H1 met (12.05%), inside
the seed 0 CI, replicated. Tip: H2 met (5.55%, paste check clean), inside
the seed 0 CI; as seed 0 did not meet H2, the replication condition ("H1 and
H2 hold again") is not met, and per section 9 both seeds are reported with
the spread; the seed 0 grade is not changed. S1 and S2 not met on seed 1.
No bar, draw or analysis was changed after the first image was scored.
