# Go / No-Go decision rule for the YOLOv8 run

Written and frozen **before** the YOLO training run in `train_yolo.py` is
executed and before any YOLO output is looked at. This file is the
pre-registered commit-vs-pivot rule; the run is graded against it after the
fact, not the other way around.

## 1. The rule

Let:
- **C** = coordinate-only top-1 numbering accuracy, gradient-boosted tree,
  **seed-0 split specifically** (not the 5-seed aggregate - see note below):
  **C = 69.28%** (`cpu_repro/coord_baseline/per_seed_results.csv`, seed=0,
  classifier=gradient_boosted_tree, top1_acc=0.6928).
- **Y** = YOLOv8 top-1 numbering accuracy on matched detections, same
  seed-0 split, computed by the same `build_coord_baseline.evaluate()`
  function (`cpu_repro/yolo_training/eval_results/summary.csv`,
  `top1_acc`).

> **Note on C**: the coordinate baseline's headline number elsewhere in
> this repo (69.5% ± 0.8) is the mean ± 95% CI **across all 5 seeds**. Y is
> a single YOLO training run on seed 0's split only, so the correct
> apples-to-apples comparison uses seed 0's own value (69.28%), not the
> 5-seed aggregate. The difference is small (0.2 points) but the distinction
> matters for rigor and is kept explicit here rather than silently rounded
> away.

**Recommendation** (pending your confirmation):

- **COMMIT** if `(Y - C) <= 5` percentage points
- **PIVOT** if `(Y - C) >= 15` percentage points
- **EXTEND** to a second dataset if `5 < (Y - C) < 15` percentage points

### Reasoning

The coordinate baseline's own seed-to-seed noise floor is small but not
zero: across the 5 seeds already run, GBT top1_acc ranges from 68.64% to
70.25% (a ~1.6-point spread; 95% CI half-width 0.77 points). Y comes from a
**single** YOLO training run with its own unmeasured run-to-run variance
(stochastic init, augmentation, etc.) stacked on top of that. Any threshold
below roughly 3-4 points risks mistaking ordinary noise for a real signal
in either direction, so 5 points was chosen as the COMMIT ceiling
specifically because it clears that noise floor by a comfortable margin
(~3x) while still describing a genuinely modest vision contribution (single
digits, not a large jump).

15 points was chosen as the PIVOT floor because the coordinate baseline
leaves 100% - 69.28% = 30.7 points of headroom above it; a 15-point gap
closes roughly half of that remaining headroom. That's a qualitatively
different regime from "vision adds a modest correction on top of
geometry" - it means real image-derived signal (tooth shape, texture,
local anatomy, adjacent-tooth context) is doing substantial work that
position alone does not capture, which weakens the "numbering is primarily
a geometric problem" framing this project is built around.

The 5-15 point middle band is deliberately wide: it's the range where the
result is real but its *size* is ambiguous enough that a single dataset
shouldn't be trusted to generalize. UFBA-425 has a notably consistent
single-protocol imaging setup (see `cpu_repro/anomaly_scan/` - the
FDI-to-region mapping is consistent enough across the whole dataset that a
geometry-only classifier gets 69%+ accuracy on it); a moderate gap could be
a genuine property of tooth numbering, or an artifact of this dataset's
unusual positional regularity. EXTEND resolves that ambiguity with data
rather than a guess.

These are recommendations, not final - confirm or adjust the two numbers
before this file is committed.

## 2. Detection-recall guard

Matched-only comparison flatters Y: unmatched ground-truth boxes (teeth
YOLO never found) are excluded from the accuracy calculation entirely, so
a model that only detects the "easy" teeth (e.g. large, well-separated
molars) and misses crowded/ambiguous ones (e.g. incisors, which the
coordinate baseline's own controls already showed are the hardest region -
`cpu_repro/coord_baseline/README.md`) could post an inflated Y computed
over a biased, shrunken sample.

**Recommendation: minimum detection recall = 90%.**

If `detection_recall` (from `eval_results/summary.csv`) is below 90%, Y as
computed is **reported as unreliable** and must not be used for the go/no-go
call as-is. In that case, repeat the comparison with a penalized variant:
every unmatched ground-truth box counts as a misclassification (i.e.
treated as if the model predicted a wrong/no class for it), giving a
recall-adjusted Y' that fairly charges the model for teeth it never found,
rather than pretending they don't exist. Use Y' against the same thresholds
in Section 1 if this triggers.

90% was chosen because a competently trained detector on a curated,
single-protocol dataset like UFBA-425 should find the large majority of
annotated teeth even before classification is considered; recall much below
that would mean detection - not numbering - is the actual bottleneck, a
different problem than the one this comparison is designed to answer.

## 3. What each outcome means and what happens next

**COMMIT.** The trained detector adds only a modest correction on top of
what raw box geometry already predicts, on this dataset's own comparable
split. This is strong supporting evidence for the project's central
premise - that FDI tooth numbering is substantially a positional/geometric
problem rather than a visual-recognition one - and justifies spending the
MICCAI 2027 effort on formalizing and improving explicit positional priors
(e.g. a geometry-aware post-processor, or a joint geometry+vision
architecture) rather than on chasing marginal detector accuracy gains.

**PIVOT.** The detector is capturing real, substantial signal that
position alone misses. The "numbering is primarily geometric" framing is
not well supported by this dataset's own strongest available baseline, and
building a paper thesis around it as-is would be vulnerable to the obvious
rebuttal ("a trained detector already beats your baseline by a wide
margin"). Next steps: run an error analysis on cases YOLO gets right but
the coordinate model doesn't, to characterize what visual signal is doing
the work; consider reframing the contribution around using positional
priors to *improve* a vision model (regularization, error-correction,
low-data regimes) rather than showing priors are sufficient on their own;
or reconsider whether UFBA-425 is the right testbed for this specific
thesis.

**EXTEND.** The result is real (clears the noise floor) but its magnitude
is ambiguous enough that committing the paper's central claim to it would
be premature. Next step: rerun the same coordinate-baseline-vs-YOLO
comparison (same methodology, same evaluate() function, same style of
image-level split) on a second, independent panoramic X-ray tooth-numbering
dataset with a different imaging protocol/institution, before deciding
between the COMMIT and PIVOT narratives above.

## 4. This file is frozen

Once committed, the numbers in Section 1 and Section 2 do not change in
place. Any revision - including narrowing a threshold after seeing that it
would flip the outcome - requires a **new, dated entry appended below this
line**, stating what changed and why. Both the original and the revised
version must be reported together in any writeup that cites this decision;
a threshold is not allowed to quietly become whatever the observed result
needed it to be.

---

*(No amendments yet.)*

## 5. This rule applies to top-1 accuracy only, not quadrant/tooth-type

The coordinate baseline's own results show quadrant accuracy (96.5%) and
tooth-type accuracy (72.0%) diverge sharply - quadrant is already near
ceiling, tooth-type is where nearly all the remaining error lives
(`cpu_repro/coord_baseline/README.md`). Applying the *same* percentage-point
thresholds to all three metrics would be structurally unsound: quadrant
accuracy has at most ~3.5 points of headroom left above the coordinate
baseline (100% - 96.5%), so it can **never** produce a 15-point gap
regardless of how much better the vision model is - the PIVOT threshold
would be unreachable on that metric by construction, not because vision
didn't help.

**The numeric COMMIT/PIVOT/EXTEND rule in Section 1 applies to top-1
accuracy only.** Quadrant accuracy and tooth-type accuracy must still be
computed and reported for the YOLO run, in the same columns as the
coordinate baseline's `per_seed_results.csv`, and read *diagnostically*
alongside the top-1 decision rather than as independent go/no-go triggers:
- If the top-1 gap is concentrated in tooth-type accuracy (quadrant stays
  ~flat near ceiling for both C and Y), that's consistent with vision
  adding value specifically where the coordinate baseline was already
  weakest, and supports the same interpretation as the top-1 result.
- If Y's quadrant accuracy is *lower* than C's, that would be a distinct
  finding (a trained detector doing worse than raw geometry at the
  near-solved sub-task) worth flagging on its own, independent of the
  top-1 outcome.
