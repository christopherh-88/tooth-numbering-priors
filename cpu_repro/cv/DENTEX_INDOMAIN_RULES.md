# Training within DENTEX: do the margin and the shared errors replicate without a domain shift?

Approved 2026-10-06 with the margin bar raised to 10 points at approval
(the stricter of the two offered), frozen before any of it is run.

## Question

Section 85 ran the UFBA-trained models on DENTEX. Shared errors still
followed position, but two detectors lost their margin (YOLOv8x mostly by
missing teeth under the shift, Faster R-CNN by numbering one position too
distal) and shared errors sat at gaps 2.3 times as often as correct teeth,
not 5. Those losses mix two things: a domain shift and a different dataset.
Training the same three detectors within DENTEX removes the shift: do the
UFBA findings hold on a second dataset when the models are trained on it?

## Data

DENTEX `quadrant_enumeration` training tier, as Section 85: 634 X-rays,
18,095 teeth. Labels:
`cpu_repro/anomaly_scan/dentex_raw/train_quadrant_enumeration.json`.
Images: the public Kaggle copy `truthisneverlinear/dentex-challenge-2023`,
attached to the kernels, never downloaded locally. The 37 duplicate FDI
labels within X-rays are kept as labeled.

## Split

- Test folds: the Section 85 assignment (random permutation, seed 0, five
  folds), so every X-ray is tested by exactly the models that did not see
  it, and the position-only floor is the one Section 85 already reports
  (66.9%, CI 65.5 to 68.2).
- Inside each fold's training X-rays, 15% (seed 0) are held out for epoch
  selection, as `PHASE2_CV_RULES.md`. DENTEX has no image categories, so
  the split is not stratified.

## Models

YOLOv8x, RT-DETR-l and Faster R-CNN v2 with exactly the UFBA settings
(`train_cv.py`: 30 epochs, image size 640, same augmentation, same
epoch-selection rule, Ultralytics 8.4.143). Nothing is tuned on DENTEX.
One training seed (0). Scoring as everywhere else: confidence at least
0.5, IoU at least 0.5, confidence-first one-to-one matching, a missed
tooth counts as wrong, 95% CIs from 10,000 bootstrap draws of X-rays.

## Graded

Rules 2 and 3 use the bars of the external test
(`DENTEX_EXTERNAL_RULES.md`), so the two DENTEX results can be read side by
side. The margin bar is stricter than there, since training within DENTEX
removes the domain shift.

1. **Margin:** for each detector, DENTEX top-1 minus DENTEX position-only
   has a CI whose lower end is at least 10 points (about two thirds of the
   smallest UFBA margin, 14.3; the external test used 5).
2. **Shared errors follow position:** among teeth that all three detectors
   number wrong with the same answer and position-only also gets wrong,
   the share where position-only gives that same answer is at least
   85.71% (the lower end of the UFBA seed 0 CI).
3. **Shared errors sit at gaps:** those teeth have a missing labeled
   neighbor (positions 1 to 8, as Section 69) more often than teeth all
   three detectors get right, with the ratio at least 4.2 (the lower end of
   the UFBA CI) and its own CI above 3.

**Replicated** if all three hold (rule 1 for all three detectors). If fewer
than 50 shared-error teeth occur, rules 2 and 3 are reported as
underpowered, not graded. The floor is not graded again: it was measured
on these folds in Section 85 and is restated.

Also reported, no rule: each detector's top-1, missed and misnumbered
rates within DENTEX next to Section 85 (the UFBA-trained models); whether
Faster R-CNN's one-position distal shift remains; how often the shared
wrong answer is the number of a tooth with no label in that X-ray; the
Section 84 gap flag; how many shared-error teeth are also shared errors in
Section 85.

## Cost

Three Kaggle GPU kernels (one per detector, five folds each, at most two
at a time), each with a smoke pass first and the usual 11-hour watchdog
that keeps finished folds. About 6 to 10 GPU hours in all, roughly a third
of the weekly Kaggle GPU quota. Then one CPU kernel scores the saved
detections with the Section 85 scorer (`dentex_external.py`, reading the
saved detections instead of running inference). Wall time about 6 to 12
hours.

## Code changes before the run

- `train_cv.py` gains options to read a boxes file, a folds file and an
  image folder other than UFBA's. The defaults are unchanged, so UFBA runs
  reproduce exactly.
- A small script writes `dentex_boxes.csv` and `dentex_folds.csv` from the
  labels with the split above.
- `dentex_external.py` gains an option to score saved detections instead
  of running the UFBA models.
