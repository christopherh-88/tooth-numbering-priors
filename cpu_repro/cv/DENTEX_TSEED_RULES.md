# Training within DENTEX, second training seed: do Section 86's numbers hold up?

Draft 2026-10-07, to be frozen at approval, before any of it is run.

## Question

Section 86 trained the three detectors within DENTEX once (training seed
0). The position share (85.5%) missed its bar by 0.2 points and the gap
ratio (3.5) missed by 0.7. On UFBA, retraining on the same split with a
new seed changed about a third of the shared-error teeth (Section 83,
overlap 0.63). Is Section 86's result a property of DENTEX or of one
training run?

## Design

Exactly as `DENTEX_INDOMAIN_RULES.md` (same folds, same 15% validation
X-rays, same settings, same scorer), with training seed 1 (`train_cv.py
--train-seed 1`, as Section 83 did for UFBA). Kaggle GPU kernels
`tooth-numbering-cv-{yolov8x,rtdetr-l,fasterrcnn}-dentex-tseed1`, each with
a CPU smoke pass; then one CPU scoring kernel.

## Graded (seed 1 alone, the bars of DENTEX_INDOMAIN_RULES.md, unchanged)

1. **Margin:** for each detector, the CI lower end of seed 1 top-1 minus
   position-only is at least 10 points.
2. **Shared errors follow position:** at least 85.71% of seed 1's shared
   errors give position-only's answer (graded only with 50 or more).
3. **Shared errors at gaps:** ratio at least 4.2 with its CI above 3.

**Seed 1 meets the bars** if all three hold. Whatever the outcome, Section
86's grade stands; this run is reported next to it, not instead of it.

## Also reported, no rule

- Section 86 against seed 1, side by side, each with its CI.
- Both seeds pooled (shared-error sets from both runs, bootstrapping
  X-rays jointly) for the position share and the gap ratio, as a more
  precise estimate. Reported, not graded, because the pooled bar would be
  set after seeing seed 0.
- Overlap of the two shared-error sets (UFBA same split, new seed: 0.63).
- Each detector's top-1, missed and misnumbered rates against Section 86.

## Cost

About 11 GPU hours (Section 86 took 1.7, 1.5 and 7.8 h), a third or more
of Kaggle's weekly 30-hour GPU quota, which this week has already used
about 13 hours on DENTEX. If the quota runs out mid-run, the finished
folds are saved in that kernel's output, and the missing folds are run in
a new kernel (`--folds`) after the weekly reset. Wall time about 8 to 10
hours with two kernels at a time.

## Code changes before the run

- `make_kernels.py --dentex --suffix=-tseed1 --train-args="--train-seed 1"`
  names the kernels with the suffix (output dirs already get `_tseed1`).
- `dentex_external.py --saved-tag _dentex_tseed1` scores those outputs.
