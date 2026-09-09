# Handoff (last synced 2026-09-08, evening)

Read this first in the next session before doing anything else. It exists
so nothing has to be re-derived from chat history.

## Where things stand

**Target:** MICCAI. Current work is workshop-strength (FAIMI/UNSURE-type
venue) as-is; main-conference strength requires closing Claim B (see
`RESULTS.md` Section 11.2) - real detector evidence, not just the
coordinate-only diagnostic.

**Git state as of this sync:** everything through the noise-robustness/
feature-ablation/literature-gap round is **committed** (commits
`3e36846`, `16e6f95`, `e8dcc26`, `395055d`, `d68d4e0`, `6ff9100`,
`3c91497`). Working tree was clean as of this sync - run
`git status --short` first thing next session to confirm nothing's
changed since.

**Standing rules still in force** (given by the user earlier, still
apply): show diffs before committing anything; grep for third-party AI
tool attribution across changed files before staging; don't commit until
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
3. **Dual-Labeled Dataset verification (Task 3): written into
   `RESULTS.md` Section 16, resolved.** Label 91 (supernumerary) present,
   24 usable instances across 23 images (dataset is an admitted partial
   release, 500 of 2,000 stated images - license "unknown," re-verified
   live against the dataset's own Kaggle metadata). 0 filename or
   perceptual-hash overlap with UFBA-425. Scripts and raw output in
   `cpu_repro/dual_labeled_dataset/`.
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
13. **Dual-Labeled Dataset supernumerary error-rate follow-up**
    (`RESULTS.md` Section 17, script
    `cpu_repro/dual_labeled_dataset/supernumerary_error_check.py`) - does
    the coordinate-only model (trained on 100% of UFBA-425) do worse on
    standard teeth in images that also contain a supernumerary tooth?
    **Coordinate-convention check passed first** (required before any
    cross-dataset comparison, per `cpu_repro/CONVENTIONS.md`). Whole-image
    result: direction matches the clinical-stakes hypothesis (29.5% vs.
    33.0% per-instance, 28.8% vs. 32.3% per-image) but **not significant**
    (p=0.077 per-instance, p=0.36 per-image - underpowered at n=23
    supernumerary-present images). Persists after controlling for
    teeth-count/crowding (OLS), still not significant.
14. **Bonus finding, same script:** Claim A transfers to this third,
    independent dataset (32.9% accuracy vs. 3.76% majority baseline,
    roughly half UFBA-425's in-domain 69.3%) - cite as further
    cross-dataset replication alongside DENTEX. Also: images with more
    visible teeth are easier for the coordinate-only model (r=0.2556,
    p<0.0001) - a fuller arch looks more canonical.
15. **A withdrawn result - do not resurrect this number.** A "localized
    adjacency" refinement (comparing teeth nearest to each supernumerary
    tooth vs. farther-away teeth) initially looked like a strong,
    significant finding (55.6% near-accuracy vs. 26.0%/33.0%, p<0.0001) -
    but a class-composition audit showed it's confounded: supernumerary
    teeth cluster anatomically in the anterior maxilla, so their "nearest
    teeth" are disproportionately anterior classes (11/12/13/21/22/23)
    that are already easy/hard regardless of proximity. **This 55.6%
    number is withdrawn, not evidence, and must not be cited or
    reintroduced** if this section of the paper gets drafted from memory
    instead of re-reading `RESULTS.md` Section 17 directly.
16. **Literature check: no public per-FDI-class confusion matrix exists
    for any real trained detector** (`RESULTS.md` Section 18, verified
    live against arXiv:2305.19112v2 - not secondhand). DENTEX's own
    authors (Hamamci et al.) explicitly flag this as missing and name
    "swapping the enumeration of adjacent teeth" as the exact
    unaddressed error type - a strong, specific motivating citation for
    Task 2. **Confirms this cannot substitute for Task 2** if the
    Friday GPU run is delayed - there's no public data to fall back on,
    so don't re-attempt this search expecting a different answer.
17. **Noise-robustness curve** (`RESULTS.md` Section 19,
    `cpu_repro/coord_baseline/noise_robustness.py`) - the coordinate-only
    signal degrades smoothly under test-time Gaussian coordinate noise,
    no cliff, and stays 5-10x above majority baseline even at noise
    levels exceeding the natural per-class positional spread (Section
    13). Practical reading: ordinary detector localization imprecision
    is unlikely to erase the signal outright, so a future Task-2 null
    result would be a genuine finding, not a noise-washout artifact.
18. **Feature ablation** (`RESULTS.md` Section 20,
    `cpu_repro/coord_baseline/feature_ablation.py`) - the signal is
    **asymmetrically position-dependent**: `x_center` dominates
    (dropping it costs 47pp, kept alone worth 36pp), `y_center` is real
    but secondary (dropping it costs 17pp, kept alone worth only 11pp),
    shape features (width/height/area/aspect_ratio) are inert (<0.3pp
    each). **Say "asymmetrically position-dependent," not "x_center
    only" or "position generally"** - both wordings were explicitly
    checked and corrected once already this session, don't regress to
    either. A flagged (not fully decomposed) follow-on: x_center +
    y_center together (0.6196) is superadditive vs. their marginal sum
    (0.4662) and closes 89% of the gap to all-six (0.6949) - suggests
    most of the "extra" signal beyond x_center alone is joint x/y
    interaction (e.g. quadrant), not a hidden shape contribution, but
    this isn't rigorously decomposed yet.

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
power check, geometric ceiling check, Dual-Labeled Dataset verification
and its supernumerary follow-up, DENTEX literature-gap check,
noise-robustness curve, feature ablation) is done. The MIRROR_MAP fix is
committed and confirmed well-powered to detect a real effect once
trained. **Nothing further is CPU-blocked - the only remaining blocker
for the next step is GPU access.** Don't re-run any of the above CPU
diagnostics expecting new information - they've each been checked,
written up, and (where relevant) had their wording explicitly corrected
once already; re-deriving them from scratch would just reproduce the
same numbers already in `RESULTS.md` Sections 18-20.

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
- Don't re-download or re-verify the Dual-Labeled Dataset (Task 3) -
  fully done and written into `RESULTS.md` Sections 16-17, including the
  supernumerary error-rate follow-up. No open decision remains here.
- Don't re-attempt the localized-adjacency refinement or re-derive the
  55.6% near-accuracy number as if it were a real finding - it's a
  documented, withdrawn confound (item 15 above / Section 17).
- Don't re-audit `is_mirror_quadrant_error`/`is_neighbor_error` - already
  statically audited, clean, documented in `mitigation/README.md`.
- Don't propose a different mitigation design from scratch - one is
  already specified with a stated falsification threshold; only change it
  if new information warrants it.
- Don't re-search the literature for a public per-FDI-class confusion
  matrix to substitute for Task 2 - checked directly against the DENTEX
  paper's primary source and its follow-up participant papers, confirmed
  absent (item 16 above / Section 18).
- Don't re-run the noise-robustness or feature-ablation scripts expecting
  different numbers, and don't describe the feature-ablation result as
  "x_center only" or "position generally, x and y equally" - the precise
  wording ("asymmetrically position-dependent") was deliberately checked
  and corrected once already; regressing to either simpler phrasing would
  undo that correction (item 18 above / Section 20).
