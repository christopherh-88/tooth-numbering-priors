# Phase 2, batch 1: decision rules (frozen 2026-10-01)

Written 2026-10-01, before any of these analyses is run, and approved by
the user the same day, before running. This file is frozen: results are graded against it, and any
later change is added as a dated note at the end, not an edit above.

All three analyses reuse saved per-tooth predictions only. No training,
no Kaggle, and no existing result file is overwritten. New outputs go to
new files, and the writeup becomes a new RESULTS.md section.

## Inputs

`eval_results/{multiseed,rtdetr_multiseed,fasterrcnn_multiseed}/joined_seed{N}.csv`,
seeds 0-9 and 11-14 (14 seeds, all three detectors). Seed 10 has no
three-detector join (RESULTS.md Section 52) and is left out. Each row is
one labeled test tooth with the detector's prediction (empty if missed)
and the position-only model's prediction on the same tooth.

Seeds 0-4 (Kaggle T4) stay the headline group. Seeds 5-9 (M3) and 11-14
(T4) are reported as separate replication groups and never pooled with
0-4, as in Sections 49-54.

## Fact found while writing this plan

Each seed's test set holds **85 source X-rays but about 200 image files**:
the public UFBA-425 release includes up to 3 augmented copies of each
X-ray, and the grouped split keeps all copies of a test X-ray in test.
So the roughly 5,500 test teeth per seed are not 5,500 independent
observations. The independent unit is the source X-ray (`image_id` in
the joined files, 85 per seed).

Checked existing bootstraps:
- Per-seed gap CIs (`rtdetr_multiseed_analysis.py`,
  `fasterrcnn_multiseed_analysis.py`), Sections 21/26
  (`robustness_analysis.py`), per-class head-to-head
  (`per_tooth_head_to_head.py`) and DenPAR: resample source X-rays.
  Correct.
- **Augmentation test (`mitigation_analysis.py`, Section 31, in the CJSJ
  paper as -0.18 pp, CI -0.74 to +0.34):** resamples the 201 image files,
  so copies of one X-ray are drawn independently. Its CI is likely too
  narrow. Analysis A below reruns it by source X-ray.
- The headline gap CIs (e.g. 24.8 [24.2, 25.4]) are t-intervals over 5
  seeds, not bootstraps. Their test sets overlap across seeds, so they
  describe run-to-run spread, not uncertainty over X-rays. A adds the
  X-ray-level view.

## A. Gap CIs by source X-ray

**Method.** For each detector and seed: draw 85 source X-rays with
replacement, keep every tooth on every copy of each drawn X-ray, and
compute detector top-1 minus position-only top-1 on those teeth (paired,
same teeth). 10,000 draws, 95% percentile CI. Missed teeth count as wrong,
as in the paper.

Sensitivity check: the same gap using only the first copy of each source
X-ray (about 85 images and 2,400 teeth per seed), so each X-ray counts
once.

Also: rerun the Section 31 augmentation comparison with source-X-ray
resampling (same 10,000 draws), and report the old and new CIs side by
side.

**Decision rule.**
1. If the 95% lower bound clears 15 pp for every seed in every group,
   the paper's "every run cleared the pre-set threshold" stands, with the
   new CIs reported.
2. If any lower bound falls between 5 and 15 pp, the sentence changes to
   the count that clear 15 pp at the X-ray level, and the per-seed CI
   becomes the thing reported, not the run count.
3. If any lower bound falls below 5 pp, stop and review before any
   wording changes; that would mean the gap is not reliable on some
   splits.
4. The headline "24 to 25 pp" stays if the one-copy-per-X-ray mean gap
   for seeds 0-4 rounds into 24 to 25 for all three detectors. If it
   moves outside that, the headline uses the one-copy number and says
   why.
5. Augmentation test: if the new CI still contains 0, the null result
   stands and the paper quotes the wider CI. If it excludes 0, the
   "unchanged" claim is withdrawn and reported as a measured difference.

## B. Missed vs. misnumbered

**Method.** Each labeled tooth is exactly one of: correct (matched at
IoU >= 0.5, confidence >= 0.5, right FDI number), misnumbered (matched,
wrong number), or missed (no matched detection). Per detector and seed,
report the three rates, plus:
- all-GT top-1 = correct / all teeth (the paper's headline metric);
- matched-only top-1 = correct / (correct + misnumbered);
- matched-only gap = matched-only detector top-1 minus position-only
  top-1 **on the same matched teeth**.
CIs by source X-ray, as in A.

**Decision rule.**
1. If the matched-only gap is within 2 pp of the all-GT gap for seeds
   0-4 (all three detectors), the headline is not driven by detection
   failures. Report both; no wording change.
2. If the matched-only gap is larger by more than 2 pp, the all-GT
   headline is the conservative one. Keep it, and say so.
3. If the matched-only gap is smaller by more than 2 pp, part of the
   headline comes from which teeth the detector skips. The paper then
   reports both gaps side by side as co-headlines.
4. Missed rate per detector is reported descriptively. The YOLOv8x seed
   7/9 spikes (Section 53) are already explained and are not re-litigated.

## C. Geirhos error-consistency kappa next to phi

**Method.** For each detector paired with the position-only model, per
seed: observed agreement c_obs = share of teeth where both are right or
both are wrong; expected agreement from accuracies alone
c_exp = p1*p2 + (1-p1)*(1-p2); kappa = (c_obs - c_exp) / (1 - c_exp)
(Geirhos et al., NeurIPS 2020). Also report kappa_max, the largest kappa
possible given the two accuracies, since with 94% vs. 69% accuracy kappa
cannot reach 1. CIs by source X-ray. The same is computed for each pair
of detectors.

**Decision rule.**
1. If the kappa CI excludes 0 in every seed 0-4 run, kappa confirms phi's
   "small but above zero" reading, and the paper reports kappa next to phi.
2. If the kappa CI includes 0 in two or more of the 15 seed 0-4 runs
   (3 detectors x 5 seeds), the "above zero in every run" sentence is
   withdrawn and replaced by the count.
3. Detector-vs-detector kappa is descriptive: it shows whether the
   detectors' errors line up with each other more than with the
   position-only model. No claim depends on it.

## Outputs (all new files)

- `image_level_reanalysis.py` (one script for A, B and C)
- `eval_results/phase2_gap_by_xray.csv`,
  `eval_results/phase2_missed_vs_misnumbered.csv`,
  `eval_results/phase2_error_consistency.csv`,
  `eval_results/phase2_augmentation_by_xray.csv`
- A new RESULTS.md section. Any number that changes in the CJSJ paper
  goes into `paper/cjsj/REVISIONS.md` and `paper_numbers.py`.
