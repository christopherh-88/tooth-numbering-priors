# External test on DENTEX: do the floor, the margin and the shared errors carry over?

Approved 2026-10-06 with the thresholds tightened at approval (asked to
strengthen them), frozen before any of it is run.

## Question

Every MICCAI result so far is on UFBA-425. Run the UFBA-trained models,
unchanged, on DENTEX X-rays from other clinics and machines: does position
still set a high floor, do the detectors still beat it, and do the errors
all three share still follow position and sit next to missing teeth?

## Data

DENTEX `quadrant_enumeration` training tier: 634 panoramic X-rays, 18,095
teeth, every tooth boxed and numbered (the disease tier labels only
diseased teeth, about 5 per X-ray, so it cannot test numbering). Labels:
`cpu_repro/anomaly_scan/dentex_raw/train_quadrant_enumeration.json`
(quadrant and position names give the FDI number, convention checked in
`cpu_repro/CONVENTIONS.md`). Images: the public Kaggle copy
`truthisneverlinear/dentex-challenge-2023`, attached to the kernel, never
downloaded locally. No DENTEX X-ray was used in training or tuning
anything in this project.

## Models and scoring

- Detectors: the seed 0 UFBA fold models of YOLOv8x, RT-DETR-l and Faster
  R-CNN (Section 65), unchanged. Each DENTEX X-ray goes to one fold's
  models, assigned by a fixed random permutation (seed 0), so every tooth
  gets one answer per detector, as in cross-validation.
- Position-only, primary: the same 6-feature gradient-boosted model,
  five-fold cross-validated within DENTEX, folds grouped by X-ray
  (seed 0). This is the floor position sets on DENTEX itself.
- Position-only, also reported: the UFBA-trained fold model, applied to
  DENTEX boxes (how far the UFBA position map transfers).
- Scoring as everywhere else: confidence at least 0.5, IoU at least 0.5,
  confidence-first one-to-one matching, a missed tooth counts as wrong;
  95% CIs from 10,000 bootstrap draws of X-rays.

## Graded

Each bar is tied to the UFBA result it has to reproduce.

1. **Floor:** the lower end of the CI of DENTEX position-only top-1 is at
   least 68.5% (UFBA's 78.5% minus 10 points, which is also the earlier
   pooled DENTEX estimate, README Claim A). The whole interval, not just
   the point, must clear it.
2. **Margin:** for each detector, DENTEX top-1 minus DENTEX position-only
   has a CI whose lower end is at least 5 points (a third of the smallest
   UFBA margin, 14.3). The size is reported next to the UFBA 14 to 17
   points.
3. **Shared errors follow position:** among teeth that all three detectors
   number wrong with the same answer and position-only also gets wrong,
   the share where position-only gives that same answer falls inside the
   UFBA seed 0 CI (85.71 to 96.00%), the same replication test used for
   the seed 1 and 2 splits (Sections 70, 75, 78), or above it.
4. **Shared errors sit at gaps:** those teeth have a missing labeled
   neighbor (FDI positions 1 to 8, as Section 69) more often than teeth all
   three detectors get right, with the ratio at least 4.2 (the lower end of
   the UFBA CI, 4.2 to 6.6) and its own CI above 3.

**Replicated** if all four hold (rules 2 for all three detectors). If fewer
than 50 shared-error teeth occur, rules 3 and 4 are reported as
underpowered, not graded.

Also reported, no rule: each detector's top-1, missed and misnumbered
rates on DENTEX against UFBA; the UFBA-trained position-only model on
DENTEX; the Section 84 gap flag on DENTEX; how often the shared wrong answer
is the number of a tooth with no label in that X-ray (UFBA: 93% of those
next to a gap).

## Cost

One Kaggle GPU kernel (inference only, 634 X-rays, 3 detectors, 5 folds
loaded once each), about 30 to 60 minutes, then the CPU analysis in the
same kernel.
