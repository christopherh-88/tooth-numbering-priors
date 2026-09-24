# Handoff (last synced 2026-09-23)

Read this first in the next session before doing anything else. It exists
so nothing has to be re-derived from chat history. **The section below is
current; the "What's actually done" / "Exact next steps" sections further
down are the 2026-09-08 snapshot, kept for the early Claim-A groundwork
detail - don't read them as describing where things stand now.**

## Where things stand (2026-09-23)

**Target:** MICCAI. Claim B (`RESULTS.md` Section 11.2) is now
well-supported, not open: three architecturally distinct detectors
(YOLOv8x, RT-DETR-l, Faster R-CNN - CNN single-stage, transformer
anchor-free, CNN two-stage, two independent training pipelines) all
converge on the same answer - real detectors substantially outperform
the coordinate-only prior and largely do not rely on it, with a
weak-but-real residual error correlation as the qualifier. See
`RESULTS.md` Sections 33/40/45 (the headline gap per architecture),
46/51 (the error-agreement mechanism, now 10 seeds), 49 (CUDA-vs-MPS
backend robustness, all three architectures), and 50 (per-tooth
head-to-head vs. the prior, paired bootstrap). `paper/DRAFT.md`'s
Discussion section already states this position.

**Git state as of this sync:** the backend-comparison/Section-46-
extension/per-tooth-head-to-head work is committed (commits `65cc970`,
`16c69a5`, `a2f62e4`, `7f27388`); `16c69a5` is pushed to `origin/main`,
the other three are local-only pending `git push`. This sync's own
documentation fixes (this file, `README.md` rewrite, `paper/DRAFT.md`
References addition for UFBA-425/OralBBNet, `RESULTS.md` Sections
49-51) are **staged, not committed** as of this sync - run
`git status --short` and `git --no-pager diff --cached --stat` first
thing next session to see exactly what's pending review.

**Kaggle status: on hold, paused by the user.** Do not push, download,
or poll Kaggle kernel status until the user explicitly says to resume -
this includes not re-checking kernel status "just to see," since that
can itself hit rate limits. Two things are queued for whenever Kaggle
resumes, in this order: (1) rerun RT-DETR-l seed 10 (the first attempt
was canceled by Kaggle at its 12h timeout with no output - give the
rerun a shorter timeout so a repeat hang doesn't burn 12h of the weekly
GPU quota again); (2) CUDA seeds 12-14 for all three detectors (9
kernels, blocked on the weekly GPU quota, which was exhausted and
resets on a schedule Kaggle doesn't publish - check the quota in the
web UI before pushing). `queue2.sh`-style pushing skips already-pushed
slugs, so a restart is safe once the user says go.

**Optional, low-priority, not on the critical path:** the Dual-Labeled
Dataset supernumerary follow-up (Section 17) is underpowered at n=23
images because only 500 of the dataset's stated 2,000 images are
public. The remaining data requires emailing the dataset author
(wbzhou23@mails.jlu.edu.cn, per the dataset's own Kaggle metadata) -
slow, no response guaranteed, and even a full reply only strengthens
one Discussion-section paragraph, not either core claim. Not worth
spending session time on unless the user asks.

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

## Exact next steps as of 2026-09-08 (superseded - kept for history only)

Everything in this subsection was written before Task 2 (detector
training) happened. It is all done now - see "Where things stand"
above and `RESULTS.md` Sections 21-51. Left here only so the sequencing
reasoning that led to Task 2 isn't lost; don't treat it as pending.

1. ~~GPU training run (Task 2 setup)~~ - done, three architectures
   (Sections 33, 40, 44-45), all with the MIRROR_MAP fix in place.
2. ~~Task 2 itself~~ - done; Claim B answer is "largely no, with a
   weak-but-real error correlation" (Sections 33/40/45/46/49/50/51).
3. **Mitigation experiment: done, clean null** (Sections 30-31, YOLOv8x
   only - never extended to RT-DETR/Faster R-CNN, see Section 31's own
   scope note if that gap matters for the final paper).
   **Boundary-condition dataset: done for DenPAR** (Section 25 - the
   shortcut weakens sharply off-domain but doesn't vanish). CBCT slices
   and a differently-structured label space remain untried, open
   extensions, not required for current claims.

## Actual next steps (2026-09-23)

1. **Resume Kaggle only when the user says so** (see "Where things
   stand" above for the exact queue: RT-DETR-l seed 10 rerun, then CUDA
   seeds 12-14). Don't poll kernel status in the meantime.
2. **Fold Sections 49-51 into `paper/DRAFT.md`** if the paper is being
   actively drafted - the Discussion section's Claim-B framing already
   matches these results in spirit, but the backend-robustness check
   (Section 49) and the 10-seed Section 46 extension (Section 51)
   aren't cited in the draft's own text yet, only in `RESULTS.md`.
3. **`git push`** the three local-only commits, and this sync's staged
   documentation fixes once the user reviews and commits them
   (commands are printed at commit time, not run automatically - see
   standing rules above).
4. **Optional, not prioritized:** the supernumerary-dataset email (see
   "Where things stand" above) and the `requirements.txt` CUDA-wheel
   ordering bug (flagged Section 36, still unfixed, needs a Linux
   machine to verify against - don't fix blind).

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
- Don't re-run the CUDA-vs-MPS backend comparison for seeds already
  covered (YOLOv8x/RT-DETR-l/Faster R-CNN, seeds 0-11 CUDA, 5-9 MPS) -
  done, verified, written up (`RESULTS.md` Section 49,
  `cpu_repro/yolo_training/BACKEND_COMPARISON.md`). Only seeds 10 (RT-
  DETR-l rerun) and 12-14 remain, and those are Kaggle-blocked, not a
  local re-run.
- Don't redo the per-tooth head-to-head (Section 50) or the Section 46
  ten-seed extension (Section 51) - both done, both verified against
  their own summary.csv/joined-table totals. Extending either to seeds
  10-14 needs CUDA per-tooth predictions that don't exist yet (only
  summary.csv was downloaded from those Kaggle runs).
- Don't re-add the UFBA-425/OralBBNet citation to `paper/DRAFT.md`'s
  References - added and independently verified 2026-09-23 (title,
  authors, DOI, and the arXiv-version/year discrepancy are all checked
  and noted inline).
