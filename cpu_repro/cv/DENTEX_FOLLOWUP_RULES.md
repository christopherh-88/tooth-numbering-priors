# DENTEX interventions, two follow-ups: stored X-ray size, and a third training seed

**Status: approved as drafted by the user 2026-10-11 (both follow-ups, N and
S), frozen before any kernel is pushed. It changes only by a dated note
(section 9).** Follows `DENTEX_INTERVENTION_RULES.md` (Sections 90, 91), whose
bars, targets, constants and analysis are reused unchanged.

## 1. Why

Sections 90 and 91 leave two open questions a reviewer will ask.

1. **Is the smaller DENTEX effect partly the 640-px resize?** The
   interventions were done on X-rays resized to 640 px wide. Faster R-CNN
   lost 8.6 to 9 top-1 points there (Q3: 91.6 to 83.0 and 91.9 to 82.9),
   and without Faster R-CNN the DENTEX effect is 0.66 of UFBA's instead of
   0.51 to 0.57. If the resize is a cause, the DENTEX effect at the stored
   size is larger.
2. **Does the tip clear its 5% bar?** Seeds 0 and 1 give 4.80% (not met) and
   5.55% (met). A bar inside the run-to-run spread cannot be settled by two
   runs.

## 2. Follow-up N: stored size (training seed 0 models)

Same 1,902 targets, same mover definition, same detectors, same
position-only models, same procedure and same conditions as Section 90,
with one change: the X-ray is **not** resized to 640 px. It is read at its
own stored size (widths 1,976 to 3,003 px, median 2,868; none resampled),
tooth polygons rasterized at that size, and the pixel constants scaled by
that X-ray's width over 640 (4.5 at the median width: dilation 13 px,
inpaint radius 22 px, largest shift 896 px, and the 1-px touch tolerance 4
or 5 px). The tooth is then about 4.5 times larger in pixels and the same
size relative to the X-ray, so the geometry is the 640-px run's, at higher
resolution. (The first draft of this paragraph said "2,943 px wide, the
modal width"; that was wrong, see amendment 1.) The
detectors resize internally as they did when scored at full size in Section
86 (Q3 then compares like with like and is expected to pass).

Kernels `tooth-numbering-cv-dentex-{closure,tip}-s0n`, same smoke pass and
Q1 stop as before, preceded by a **regression gate**: the script was
restructured after Sections 90 and 91 (it now builds and scores the
altered images for 40 X-rays at a time, because at the stored size all
altered images no longer fit in memory), so each kernel first reruns 10
X-rays at 640 px and compares with the Section 90 rows of the same mode:
the geometry columns must be identical and at least 99% of the detectors'
answers the same, otherwise the kernel stops. A local check, run before
the kernels (old and new script, 16 X-rays, both modes, stand-in
detectors depending only on image bytes), gave byte-identical rows.

**Reading, fixed now.** Graded by the same bars as Section 90 on the same
quantities (H1, H2, S1, S2), but **the 640-px run stays the primary
analysis** and N is a sensitivity analysis: it can add a result, not
replace one.

| Outcome of N | Reading | Sentence in the paper |
|---|---|---|
| Q3 not met (Faster R-CNN intact top-1 more than 2 points under stored) | The size effect is not removed; N is reported as run, uninterpretable on this point | Stated as a limit |
| j(closed) and j(full tip) inside the 640-px seed 0 CIs (10.12 to 13.54; 3.69 to 6.01) | The resize does not explain the smaller effect | "Running at the stored size gives the same rates (x% and y%)" |
| Either above the upper end of that CI | The resize lowers the effect; part of the UFBA gap is a resolution artifact | Both reported; the effect size claim for DENTEX is the stored-size one, and the 640-px runs are labeled a lower bound |
| Either below the lower end | The effect is lower at the stored size | Both reported, with the spread |

H2 may become met or stay unmet on N; either is reported and neither
changes the seed 0 or seed 1 grades.

## 3. Follow-up S: third training seed

Train the three detectors within DENTEX again, as
`DENTEX_INDOMAIN_RULES.md` and `DENTEX_TSEED_RULES.md` (same folds, same
validation X-rays, same settings), with `--train-seed 2`; one CPU scoring
kernel (`dentex_external.py --saved`) as for seed 1; then the two
intervention kernels at 640 px, exactly as for seed 1 (`-closure-t2`,
`-tip-t2`). Kernels `tooth-numbering-cv-{yolov8x,rtdetr-l,fasterrcnn}-dentex-tseed2`.

**Reading, fixed now.** No new bar. Reported:

- j(closed) and j(full tip) with CIs on seed 2; H1 and H2 graded with the
  rule's bars as for each seed.
- Across the three seeds: the three values, their range and mean, and
  how many seeds meet H2 ("met on k of 3").
- Observational: Section 88's comparison extended to three seeds (position
  share, gap ratio, and the three-way overlap of shared-error sets).

| Outcome | Sentence in the paper |
|---|---|
| H2 met on 2 or 3 of 3 seeds | "The tip effect on DENTEX is about 5% and met our 5% bar on k of 3 training seeds" |
| H2 met on 1 of 3 | "The tip effect on DENTEX is 4 to 6%, straddling our 5% bar (met on 1 of 3 seeds)" |
| H2 met on 0 of 3 | Seed 1's 5.55% is described as the upper end of a spread of x to y%; the tip is "a smaller effect" on DENTEX |

The seed 0 grade ("H1 met, H2 not met") stands whatever is found, as in
Section 91: seed 0 is the primary analysis and the pooled spread is
descriptive.

## 4. Quality control (before any number is read)

Q1 (position-only agreement 1.0000, hard stop), Q3 (intact answers against
stored: flagged above 2 points), Q4 (skips under 10%), Q5 (examples looked
at) as in `DENTEX_INTERVENTION_RULES.md` section 6. For N, Q3 is the check
that the stored-size run is what it claims to be; a failed Q3 is a result,
not a reason to rerun with other settings.

## 5. Statistical analysis

As Section 90: 10,000 bootstrap draws of X-rays, percentile CIs, paired by
mover, rng seed 0. Per-seed and per-run CIs are not combined into a
meta-analytic interval; the three seeds are shown side by side.

## 6. Threats to validity

- N changes resolution only if the detectors' internal resizing makes the
  input equal; YOLOv8x and RT-DETR-l letterbox to 640 px either way, so for
  them N changes how the edit was made (at higher resolution, then
  downsampled), not what they read. A null for N is therefore expected for
  those two and informative mostly for Faster R-CNN.
- Telea inpainting at about 4.5 times the radius is not the same texture
  synthesis at the same relative scale.
- Three seeds on one split: the spread is the seed spread on this split,
  not the spread across splits.
- All limits of `DENTEX_INTERVENTION_RULES.md` section 11.

## 7. Resources and order

| Item | Estimated GPU hours (T4) |
|---|---|
| N: two kernels, plus smoke passes | 3 to 6 |
| S: three training kernels (Section 88 took about 5.4 h for a seed) | 5.4 |
| S: two intervention kernels (about 0.6 h each) | 1.2 to 2 |
| Total | 10 to 14 of the roughly 16 h left |

N and the S training kernels are independent and run first, two at a time;
scoring and the S intervention kernels follow. If the quota runs short, S's
`-closure-t2` is dropped before `-tip-t2` (the tip is the open question),
and that is said in the write-up.

## 8. Reporting commitments

RESULTS.md Section 92 (N) and Section 93 (S), each with the flow, QC, the
table of j by condition with CIs, the grade of H1, H2, S1, S2, the reading
from this file, and the saved examples; paper supplement S2 extended; all
results reported whether or not they support the hypotheses. No bar,
threshold, target draw or analysis is changed after the first image of a
run is scored.

## 9. Approval and amendment log

| Date | Entry |
|---|---|
| 2026-10-11 | Drafted |
| 2026-10-11 | Approved as drafted by the user (both N and S). Frozen. Kernels pushed after this entry |
| 2026-10-11 | **Amendment 1, before any kernel was pushed.** (a) The draft said the stored size was "2,943 px wide, the modal width". Checking the labels file: widths run 1,976 to 3,003 px (median 2,868; the most common size, 52 X-rays, is 1,976 x 976). N now keeps each X-ray at its own stored size and scales the constants by its own width over 640, which is what "stored size" means and needs no resampling. (b) The regression gate and the 40-X-ray chunking described in section 2 were added. No bar, threshold, target draw, condition or reading changed |
| 2026-10-12 | **Outcome note, no bar, rule or reading changed.** N ran as written (Section 92; regression gate and Q1 passed, Q3 met). S: seed 2 trained and scored (Section 93). The seed 2 intervention kernels `-tip-t2` and `-closure-t2` were not run: Kaggle refused the pushes at the weekly GPU quota (30 h). They run, tip first, when the quota allows; if they are not run before submission the paper states the tip as two 640-px seeds plus the stored-size run |
