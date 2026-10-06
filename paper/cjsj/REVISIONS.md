# CJSJ revision notes (Phase 1)

Planned text edits to `HuangChristopher_paper.docx` (Sep 30 version, kept
outside the repo). Writing only, no new experiments. Nothing here has been
applied to the docx yet. Every number below is already in the submitted
paper and checked by `paper_numbers.py`; no new numbers are introduced.

Goals:
1. Mechanism (position as tie-breaker) in the second half of the abstract and
   the first sentence of the conclusion, with the "exploratory" caveat.
2. A "two implications" paragraph.
3. The 24 to 25 pp gap as the one repeated number; cut the 480-combination
   sentence, the narrowest-margin sentence and extra feature-ablation detail.
4. Joint-failure result (97% same wrong tooth, 84 to 88% position match)
   earlier in Results.
5. Repo DOI next to the repo link (see the end of this file).

## 1. Abstract (replace in full)

> AI systems that number teeth on dental X-rays report high accuracy, and a
> wrong number attaches a finding to the wrong tooth. A model could get many
> numbers right from where a tooth sits in the image instead of what it looks
> like, a shortcut that would fail on teeth in unusual positions. On two
> public panoramic X-ray datasets, including DENTEX, the benchmark from the
> MICCAI 2023 dental challenge, a classifier given only each tooth's
> bounding-box coordinates, with no image pixels, named the tooth correctly
> about 69% of the time, against under 4% for always guessing the most common
> tooth. On periapical X-rays, which lack the panoramic format's fixed
> framing, it fell to 26.6%. Three object detectors of different designs
> (YOLOv8x, RT-DETR-l, and Faster R-CNN), each trained on five random data
> splits, beat the position-only model by 24 to 25 percentage points on the
> same teeth and on every tooth class, clearing a threshold fixed before
> training. Position still shaped where they failed. On the roughly 2% of
> teeth that every model got wrong, the three detectors picked the same wrong
> tooth 97% of the time, and that tooth matched the position-only guess 84 to
> 88% of the time, against 4 to 11% by chance. We designed this analysis
> after seeing the main results, so it is exploratory, but it points to one
> mechanism: the detectors identify teeth by appearance and use typical
> position only to break ties between neighbors that look alike. A
> position-only baseline trains in seconds and shows how much a
> tooth-numbering model adds beyond position.

Changes: periapical result moved up so the abstract ends on the mechanism;
"exploratory" caveat added (the old abstract had none); "93 to 95%" cut so the
gap is the only detector number; "3.6 to 3.7%" became "under 4%".

## 2. Introduction, last paragraph of "We tested two claims"

Current last sentence:
> We first measure how much position alone predicts, then check whether three
> detectors do better and where position still shows up in their errors.

Replace with:
> We first measure how much position alone predicts, then check whether three
> detectors do better. The most telling result is in their errors: on the
> teeth every model got wrong, the detectors mostly gave the answer that
> position alone would give.

## 3. Results reorder (moves the joint-failure result earlier)

New order of Results and Discussion:
1. Position Alone Predicts Tooth Number (Claim A), trimmed (see 4a)
2. Detectors Go Well Beyond Position (Claim B), first paragraph and Table I
   only, trimmed (see 4b)
3. **Where Position Still Shows Up** (moved up from after the augmentation
   paragraph), unchanged text, with Fig. 3 and the replication-group
   paragraph
4. The Fig. 1b case paragraph (tooth 11 vs. 21), now read as an example of a
   look-alike pair, opening with: "Fig. 1b shows the kind of pair where
   position could break a tie."
5. The augmentation paragraph (see 4c)
6. Boundary Condition, Clinical Check, Limitations: unchanged

Fig. 3 will then be cited before Fig. 1b in the text. If the journal wants
figures cited in order, renumber Fig. 3 as Fig. 2 and the current Fig. 2 as
Fig. 3.

## 4. Cuts

### 4a. Claim A, feature ablation

Current:
> In the feature ablation, the two center coordinates alone reached 62.0%,
> and the four size and shape features alone reached 14.3%. Removing the
> horizontal coordinate cost 47.5 points and removing the vertical one cost
> 16.8. No single size or shape feature changed accuracy by more than 0.3
> points, because they overlap, but together they added 7.5. Most of the
> information is in a tooth's left-to-right position along the arch.

Replace with:
> In a feature ablation, the two center coordinates alone reached 62.0%, and
> removing the horizontal coordinate alone cost 47.5 points: most of the
> information is in a tooth's left-to-right position along the arch.

### 4b. Claim B paragraph

Current:
> On the same test teeth, the detectors reached 94.3% (YOLOv8x), 94.5%
> (RT-DETR-l), and 93.3% (Faster R-CNN) top-1 accuracy (Fig. 2a, Table I),
> 24.8, 25.0, and 23.8 pp above the position-only model, with overlapping
> CIs. All 15 runs cleared the pre-set 15-point threshold, and the smallest
> single-run gap, 22.7 pp, had a CI starting at 19.3 pp. On the teeth they
> detected, all three named the quadrant correctly 99.6 to 99.7% of the time
> (position-only: 96.5% on all teeth). Pooled over the five seeds, each
> detector beat the position-only model on all 32 FDI classes, with every
> per-class CI above zero (Fig. 2b), and the same held in both replication
> groups. The narrowest margin was YOLOv8x on tooth 38, a lower wisdom tooth,
> at +7.2 pp (CI +3.5 to +10.9). In single seeds, the position-only model came
> out ahead once in 480 detector, seed, and class combinations, by 1.6 pp on
> tooth 38 (YOLOv8x, seed 1).

Replace with:
> On the same test teeth, all three detectors beat the position-only model by
> 24 to 25 pp, with overlapping CIs (Fig. 2a, Table I), and every one of the
> 15 runs cleared the pre-set 15-point threshold. Pooled over the five seeds,
> each detector beat the position-only model on all 32 FDI classes, with
> every per-class CI above zero (Fig. 2b), and the same held in both
> replication groups.

Per-detector accuracies stay in Table I. Dropped: the smallest single-run
gap, the detector quadrant accuracy, the tooth 38 narrowest margin and the
480-combination sentence.

### 4c. Augmentation paragraph

Current ending:
> ... but it does rule out that augmentation as the source of the 24.8-point
> gap.

Replace with:
> ... but it does rule out that augmentation as the source of YOLOv8x's gap.

## 5. Conclusion

### Opener (replaces the current first paragraph)

> Three detectors with very different designs beat a position-only model by
> 24 to 25 points on every tooth class, and when they all failed, their
> errors pointed to one mechanism: they read the teeth and fall back on
> typical position only to break ties between look-alike neighbors. That
> reading comes from an analysis we designed after the main results. Position
> alone still predicts tooth number far above chance on panoramic X-rays,
> which is why a position-only baseline is worth reporting.

The quadrant figures (96.5 to 98.0%) move out of the conclusion; they stay in
Results.

### Two implications (replaces the first four sentences of the current second paragraph)

> The results have two implications. For readers of tooth-numbering papers,
> high accuracy does not by itself show that a model reads the image, so it
> is fair to ask for a position-only baseline: it needs no image pixels,
> trains in seconds on a laptop, and shows how much the detector adds. For
> people building detectors, two checks come first. Any left-right flipping
> during training should swap left and right tooth numbers, or about half of
> the flipped training views carry wrong labels. And when checking a model's
> output, look first at confusions between neighboring teeth, since nearly
> all the errors shared by every model here named a tooth next to the true
> one.

Keep the two "Future work should..." sentences that follow, unchanged.

## 6. Repo DOI (pending)

Current:
> Code, per-seed results, analysis scripts, and a script that recomputes
> every number in this paper are available at
> github.com/christopherh-88/tooth-numbering-priors.

After a Zenodo-archived release exists:
> ... are available at github.com/christopherh-88/tooth-numbering-priors
> (archived version: doi:10.5281/zenodo.XXXXXXX).

Blocked until the Zenodo setup is done.

## 7. Numbers that Phases 2 and 3 change (added 2026-10-01)

Sections 1 to 6 above use only numbers already in the submitted paper.
Phases 2 and 3 (RESULTS.md Sections 59 to 65) changed some of them. Which
set the CJSJ version reports is the author's call; these are the
replacements if it moves to the uncropped cross-validation results. Not
applied to the docx.

| Claim in the paper | Submitted | Uncropped 5-fold CV, confidence-first matching (Sections 60, 64, 65) |
|---|---|---|
| Detector minus position-only gap | 24 to 25 pp | 14.3 to 16.7 pp (CIs 13.0 to 18.1) |
| Position-only top-1 | about 69% | 78.5% (76.9 to 80.0) |
| Detector top-1 | 93 to 95% | 92.8 to 95.2% |
| Teeth every model got wrong | about 2% | 1.4% (161 of 11,602) |
| Same wrong tooth across the three detectors | 97% | 99.4% |
| Shared wrong answer = position-only answer | 84 to 88% | 91.3% (chance p95 6.8%) |
| Shared wrong answer is a neighbor | nearly all | 94.4% |

Why the gap shrank: most test images in the submitted split are
Roboflow's randomly cropped copies (Section 59). Cropping moves teeth
within the frame, which costs the position-only model about 8 pp (68.4% on
cropped copies against 76.8% on uncropped originals) while the detectors
barely change, so it inflates the gap. The uncropped CV uses only the
original X-rays.

New findings that could support the mechanism sentence:
- Detectors are confidently wrong on these teeth (AUROC 0.53 to 0.66,
  Section 65), so a confidence threshold would not catch them.
- About 10% of the shared errors are teeth whose label is disputed by a
  second annotation (Section 61, 65); the position match holds without them
  (91.7%).
- Shifting the whole image 10% sideways barely changes the detectors'
  answers (under 0.4 pp, Section 63), so "typical position" in the
  mechanism sentence should read as position relative to the neighboring
  teeth, not position in the frame. Shift test v2 and context masking
  (running) will sharpen this.

## 8. Mechanism wording after the interventions (added 2026-10-01)

Sections 67 to 69 of RESULTS.md change what the mechanism sentence can
claim. Not applied to the docx.

Current (abstract, section 1 above):
> ... it points to one mechanism: the detectors identify teeth by
> appearance and use typical position only to break ties between neighbors
> that look alike.

The interventions do not show the detectors using position: shifting the
image barely changes their answers, and masking the far context leaves
the shared wrong answers in place. What they show instead is where the
shared errors happen. Suggested replacement:

> ... it points to one failure mode: the shared errors cluster next to
> missing teeth (45% of them, against 9% of correctly numbered teeth), and
> there every model, with or without image pixels, gives the tooth the
> number of the missing one.

Suggested addition to "two implications" (section 5 above), replacing
"look first at confusions between neighboring teeth":

> ... check the numbering of every tooth next to a gap in the arch, where
> the shared errors concentrate.

Both use numbers from the uncropped cross-validation (section 7), so they
only fit if the paper moves to that set.

## 9. Second split (added 2026-10-05)

RESULTS.md Section 70. A new five-fold split (seed 1) with YOLOv8x and
RT-DETR-l retrained gives gaps of 15.8 and 15.9 pp (seed 0: 16.5 and
16.7) and the same 91% position match for the shared errors (seed 0:
90%), all inside the seed 0 confidence intervals. Suggested sentence for
the limitations or methods paragraph, if the paper moves to the uncropped
set:

> On a second random split, with YOLOv8x and RT-DETR-l retrained, both
> gaps and the position match of the shared errors fell inside the first
> split's 95% confidence intervals.

## 10. Gap intervention, and a limit on section 8 (added 2026-10-05)

RESULTS.md Section 71. Erasing a tooth from the image makes each detector
sometimes give a neighbor the erased tooth's number (2.0% YOLOv8x, 4.4%
RT-DETR-l, 5.7% Faster R-CNN, against near 0 for the controls). All
three do it to the same tooth only 4 times in 2,260. So an empty slot
alone does not produce the shared errors of section 8.

The section 8 wording should not say that a gap causes the shared
errors. Suggested replacement for its last clause:

> ... and there every model, with or without image pixels, gives the
> tooth the number of the missing one. Erasing a tooth from the image
> makes each detector do this only occasionally and seldom together,
> so the shared errors likely depend on the neighbor having moved into
> the space, which is where the position-only model also places the
> missing tooth.

## 11. Closed gaps (added 2026-10-05)

RESULTS.md Section 72 tests the explanation suggested in section 10.
Where a tooth is missing and both its neighbors are present, the shared
errors happen where the neighbors have closed the space: the space left
is a predictor of a shared error with AUROC 0.78 (0.70 to 0.84), and 56%
of these errors sit at gaps where the neighbors' outlines touch, against
20% of gaps without one. This replaces the "likely" in section 10's
suggested wording:

> ... and there every model, with or without image pixels, gives the
> tooth the number of the missing one. This happens where the teeth on
> either side have closed the space, so the arch shows nothing missing.
> Erasing a tooth, which leaves a full-width space, seldom produces it.

Suggested review rule, replacing section 8's: check the numbering next to
every missing tooth whose space has closed. Note for the text: most gaps
in this data are at least partly closed, so the rule is broad (it would
catch 97% of these errors but only 13% of flagged gaps hold one).

## 12. Headline decision (added 2026-10-05)

Decided: the headline numbers come from the uncropped 5-fold
cross-validation with confidence-first matching (RESULTS.md Sections 60,
65, 70), not the submitted split. The submitted 24 to 25 pp gap and the
"about 69%" position-only figure were measured on a test set that was
mostly randomly cropped copies (Section 59), which inflates the gap.
Sections 1 and 7 above become:

| | Old headline | New headline |
|---|---|---|
| Position-only top-1 | about 69% | 78.5% (76.9 to 80.0) |
| Detector minus position-only | 24 to 25 pp | 14.3 to 16.7 pp, on all 32 tooth numbers |
| Detector top-1 | 93 to 95% | 92.8 to 95.2% |
| Shared errors | 97% same wrong tooth; 84 to 88% match position | 1.4% of teeth; 91.3% match position (chance p95 6.8%) |
| Replication | five random splits of the cropped set | a second 5-fold split (Section 70) |
| Mechanism | position breaks ties between look-alike neighbors | closed gaps: the arch looks complete next to a missing tooth (Sections 69 to 72) |

DENTEX (68.5%) and periapical (26.6%) position-only numbers were measured
under the original protocol, not this one. If they stay in the CJSJ
paper, say so in the sentence ("on DENTEX, under the original splits").
The MICCAI draft leaves them out.

Replacement abstract for section 1:

> AI systems that number teeth on dental X-rays report high accuracy, and
> a wrong number attaches a finding to the wrong tooth. A model could get
> many numbers right from where a tooth sits in the image rather than what
> it looks like. On 425 uncropped panoramic X-rays (11,602 teeth), with
> five-fold cross-validation grouped by X-ray, a model given only each
> tooth's bounding box, with no image pixels, named the tooth correctly
> 78.5% of the time. Three object detectors of different designs
> (YOLOv8x, RT-DETR-l and Faster R-CNN) beat it by 14 to 17 percentage
> points, on every one of the 32 tooth numbers, and the result held on a
> second random split. Position still shaped where they failed. On 1.4%
> of teeth all three detectors gave the same wrong number, and the
> position-only model gave that number too 91% of the time, against under
> 7% by chance. These shared errors sat next to a missing tooth five
> times as often as correctly numbered teeth, usually took the missing
> tooth's number, and happened where the neighboring teeth had closed the
> space. Erasing a tooth from the image, which leaves the space open,
> rarely produced them. Where a gap has closed, the arch looks complete,
> and every model numbers the tooth as if it had moved into the space.
> A position-only baseline trains in seconds and shows how much a
> tooth-numbering model adds beyond position, and checking the numbering
> next to closed gaps targets the errors that models share.
