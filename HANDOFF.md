# Handoff (last updated 2026-09-08)

Read this first in the next session before doing anything else. It exists
so nothing has to be re-derived from chat history.

## Where things stand

**Target:** MICCAI. Current work is workshop-strength (FAIMI/UNSURE-type
venue) as-is; main-conference strength requires closing Claim B (see
`RESULTS.md` Section 11.2) - real detector evidence, not just the
coordinate-only diagnostic.

**Git state as of this write-up:** the MIRROR_MAP fix, significance
tests, positional-variance measurement, and Task 2 power check are all
**committed** (commits `7f420b2`, `2423646`). The geometric-ceiling
check (Section 15 below) may still be uncommitted depending on when this
is read - run `git status --short` first thing next session to check,
don't assume.

**Standing rules still in force** (given by the user earlier, still
apply): show diffs before committing anything; grep for AI attribution
(`claude|anthropic|generated with|co-authored-by|ai-generated|written by ai`,
case-insensitive) across changed files before staging; don't commit until
the user explicitly says so.

## What's actually done (verified, not just claimed)

1. **Coordinate-only baseline (Claim A): well-supported.** Geometry alone
   predicts FDI tooth identity at 67-72% top-1 (32-way) vs. 3.6% majority
   baseline, replicated on UFBA-425 and DENTEX, clean negative controls,
   5-seed CIs, pre-stated falsification threshold. `RESULTS.md`
   Sections 2, 4, 10.
2. **Novelty check (Task 1, real citation-graph check, not a web search):**
   132 unique citing papers across 5 seed papers (DENTEX, HierarchicalDet,
   the Hill/Koback/Schilling shortcutting paper, SHORTKIT-ML, "Shortcut
   Learning in Medical Image Segmentation") scanned for numbering-terms x
   shortcut-terms co-occurrence. **0 hits** - genuine negative result,
   supports novelty.
3. **Dual-Labeled Dataset verification (Task 3):** label 91 (supernumerary)
   present, but only 24 usable instances across 23 images (dataset is an
   admitted partial release, 500 of 2,000 stated images). 0 filename or
   perceptual-hash overlap with UFBA-425. **Findings reported in chat only
   - per explicit earlier instruction, NOT written into RESULTS.md.**
   Still an open decision whether/how to incorporate (see
   `paper/DRAFT.md` Section 7).
4. **Falsification threshold:** added to `RESULTS.md` Section 2 and
   `cpu_repro/coord_baseline/README.md`, committed.
5. **Flip/label-mismatch bug in the training notebook: found, confirmed,
   fixed, re-verified** (see below) - committed.
6. **Mitigation experiment: designed, not run.**
   `cpu_repro/coord_baseline/mitigation/README.md` - geometry-jitter
   augmentation (primary) or a decorrelation loss (fallback), evaluation
   protocol, falsification threshold. Blocked on GPU.
7. **Paper draft skeleton:** `paper/DRAFT.md` - abstract, related work
   (real citations: Geirhos 2020, DeGrave et al. 2021, Winkler et al.
   2019, Lin et al. MICCAI 2024, Zhou et al. 2024 BMC Oral Health, DENTEX/
   HierarchicalDet), methods pointers, results outline with `[PENDING]`
   markers, discussion, limitations, ethics statement. Committed.
8. **Statistical testing script:** `cpu_repro/coord_baseline/significance_tests.py`
   - paired permutation tests (real vs. shuffled control; top1 vs.
   majority baseline), written against the actual CSV columns, **not yet
   run**. Committed.
9. **Boundary-condition dataset candidates** (not yet searched for or
   downloaded - deliberately not started, see sequencing note below),
   noted in `RESULTS.md` Section 11.5: bitewing/periapical radiographs
   (leading candidate - breaks the acquisition-canonicalization
   precondition cleanly), CBCT slices (messier, adds volumetric
   complexity), a differently-structured label space on panoramic
   radiographs (isolates the other precondition).
10. **Natural positional variance measured** (`RESULTS.md` Section 13) -
    mean per-class std 0.0267 (x) / 0.0490 (y), normalized. Sizes the
    mitigation experiment's jitter magnitude (>= 0.0535 x / 0.0979 y).
11. **Task 2 pre-registered detectability check** (`RESULTS.md`
    Section 14) - confirms Task 2 is statistically well-powered across a
    75-95% plausible detector-accuracy range (minimum detectable gap
    2.8-6.4pp). Task 2 is worth running as planned, not underpowered.
12. **Geometric ceiling check** (`RESULTS.md` Section 15) - a k-NN
    nonparametric check found no evidence GBT is leaving headroom
    unexploited (best k-NN 65.6% vs. GBT 69.3%, k-NN does not exceed
    GBT). Supports reading the accuracy number as genuine geometric class
    overlap, not an under-fit classifier - stated as a moderate,
    non-overstated claim (not a rigorous Bayes-error proof).

## The flip/label-mismatch bug, in detail

`notebooks/yolov8+unet/yolov8+unet_training.ipynb` encodes bounding boxes
as a fixed-channel array (`binary_map[class_id, y1:y2, x1:x2] = 1`, cell
13) where channel index = FDI class. The only augmentation (cell 21) was
a 50%-probability `np.fliplr` that mirrored box content but never
remapped which channel it lived in - so on ~50% of augmented training
views, a flipped tooth's box ended up spatially in its mirror-quadrant's
position while still labeled under its original (now wrong) channel.
Confirmed with a minimal non-training CPU reproduction using real UFBA-425
box coordinates (FDI 11 / FDI 21 pair). Fixed by adding a `MIRROR_MAP`
channel permutation (built from the existing `MIRROR_QUADRANTS` relation
already in `build_coord_baseline.py`) applied in lockstep with the flip.
Re-verified: the fix is correct (full before/after evidence and the
exact diff are in `cpu_repro/coord_baseline/mitigation/README.md`).

**Why this matters for Friday:** Task 2 (error-pattern correlation
check, the thing that would close Claim B) depends on training through
this now-fixed pipeline. Training on the pre-fix version would confound
any measured shortcut-reliance signal with this labeling artifact - so
the fix needs to be committed and actually used, not just sitting in the
working tree, before the GPU run.

## Exact next steps, in order (per the user's explicit sequencing -
don't reorder or add items ahead of this without asking)

All CPU-only prep (fix, significance tests, positional variance, Task 2
power check, geometric ceiling check) is done. The MIRROR_MAP fix is
committed and confirmed well-powered to detect a real effect once
trained. **Nothing further is CPU-blocked - the only remaining blocker
for the next step is GPU access.**

1. **GPU training run (Task 2 setup), once Kaggle access resumes Friday:**
   run `notebooks/yolov8+unet/yolov8+unet_training.ipynb` (now with the
   MIRROR_MAP fix in place - confirm `git log` shows commit `7f420b2` or
   later is checked out) or the plain YOLOv8 notebook, to get a real
   trained detector's predictions.
2. **Task 2 itself:** run those predictions through the existing
   `is_mirror_quadrant_error` / `is_neighbor_error` taxonomy in
   `build_coord_baseline.py` (already static-audited, no bugs found) and
   compare against the coordinate-only model's error pattern. This is
   the result that determines whether Claim B has any support. The
   power check (Section 14) confirms this comparison is well-powered
   even if the detector turns out quite accurate.
3. **Only after 1-2:** the mitigation experiment
   (`cpu_repro/coord_baseline/mitigation/README.md` - jitter magnitude
   already sized in Section 13) and the boundary-condition dataset work
   (candidates listed, none searched for/downloaded yet - deliberately
   not started; the user was explicit that these come after Task 2, so
   don't front-run them without asking first even though they're
   CPU-only and could technically be done now).

## Things NOT to re-litigate or redo

- Don't re-run the Semantic Scholar novelty check (Task 1) - it's done,
  cached results exist, real negative result.
- Don't re-download or re-verify the Dual-Labeled Dataset (Task 3) - done,
  findings are above and in chat history, just needs a decision on
  whether/how to write into RESULTS.md, not more data-gathering.
- Don't re-audit `is_mirror_quadrant_error`/`is_neighbor_error` - already
  statically audited, clean, documented in `mitigation/README.md`.
- Don't propose a different mitigation design from scratch - one is
  already specified with a stated falsification threshold; only change it
  if new information warrants it.
