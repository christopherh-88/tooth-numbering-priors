# MICCAI 2027 draft

Status: annotated working copy of the paper. The text matches
`paper/miccai/latex/main.tex` (rewritten in plainer language 2026-10-25);
this file adds a trailing `[R##]` after each number naming the RESULTS.md
section it comes from, and the longer tables. Anonymized for double-blind
review: no author names, no repo link (the code link goes in after
acceptance).

---

## Title

When Tooth-Numbering Detectors Fail Together: A Position-Only Baseline
and Closed Gaps on Panoramic X-rays

## Abstract

Tooth-numbering detectors for panoramic X-rays reach high accuracy, but the
format puts each tooth in a predictable place, so some of that accuracy
could come from position alone. We measure this with a
position-only baseline: a classifier that sees each tooth's bounding box and
no pixels. On 425 uncropped X-rays (11,602 teeth), with
cross-validation grouped by X-ray, it numbers 78.5% of teeth correctly
[R65]. Three detectors (YOLOv8x, RT-DETR-l, Faster
R-CNN) beat it by 14.3 to 16.7 percentage points, on all 32 tooth numbers,
and this holds on two more splits [R65, R74, R75, R78]. Their rare shared
mistakes still follow position. On 1.4% of teeth the three detectors and the
baseline are all wrong and the detectors agree on one wrong number; in 91%
of these the baseline gives the same number, against under 7% by chance
[R65]. These teeth are five times as likely as correct teeth to sit next to
a missing tooth, and the wrong number is usually the missing tooth's number
[R69]. They occur where the neighbors have closed the space [R72]. Closing a gap on purpose reproduces them: when a neighbor is
moved into an erased tooth's space, all three detectors give it the missing
number 21% of the time, against 0.2% when it is left open [R71, R76].
On 634 DENTEX X-rays, detectors trained there beat position alone by 22 to
25 points, their shared errors again give the baseline's answer (85.5%), and
closing a gap reproduces them 12 to 16% of the time [R86, R90, R91, R92]. We release the per-tooth table and a scorer.

Keywords: tooth numbering, panoramic radiography, shortcut learning,
error consistency, object detection

---

## 1 Introduction

Assigning each tooth its FDI number is the first step of most automated
dental charting: a caries or a periapical lesion found by a detector is
only useful once it is attached to the right tooth. Public benchmarks such
as DENTEX [Hamamci et al.] score numbering together with detection, and detectors reach
high numbering accuracy (Table 1).

Panoramic X-rays are a standardized format. The patient bites on a guide,
the arch runs left to right across the image, and each tooth number has a
typical location. This invites a shortcut [Geirhos et al. 2020]: a model
can get many numbers right from where a box sits, without reading the
tooth. Such a shortcut would not show in overall accuracy, and it would
fail on patients who differ from the typical arch, for example after
extractions.

We ask how much of tooth numbering position alone explains, and whether
detectors fail where position fails. Our contributions are:

1. A position-only baseline, trained in seconds on six box features,
   that numbers 78.5% of teeth correctly under grouped five-fold
   cross-validation. Three detectors beat it by 14 to 17 pp on every
   tooth number, and the result replicates on two further splits.
2. An error analysis showing that the errors the detectors share are
   position-like.
3. Experiments (image shift, context masking, tooth erasure, and moving a
   neighbor into the space of an erased tooth) that place these errors at
   closed gaps, next to a missing tooth whose neighbors have moved
   together so that the arch looks complete.
4. A released per-tooth benchmark with folds, reference predictions and
   a scorer that reports the gap over the baseline and accuracy on the
   shared failures.

## 2 Related work

**Tooth numbering.** Detection-based numbering on panoramic X-rays is
standard, with DENTEX [Hamamci et al.] and diffusion-based
detectors [Hamamci et al., HierarchicalDet] as recent examples. DENTEX
challenge entries combine segmentation with detection [He et al.], build a
YOLO-based framework [Mei et al.], or use Mask R-CNN with a DiffusionDet and
DINO ensemble [Choi et al.]. Several methods use the arch layout explicitly
as a prior [Budagam et al., OralBBNet]. Evaluations report accuracy or
mAP; we found none that compares a model with what position alone
achieves on the same teeth.

**Shortcut learning in medical imaging.** Models can rely on cues that
correlate with the label in the training data, from skin markings in
dermoscopy [Winkler et al.] to hospital-specific features in chest
X-rays [DeGrave et al.] and segmentation [Lin et al.]; see also [Hill et
al.] and [Cajas et al.]. These studies usually show a shortcut by
removing or altering the cue. We use the same logic (shift, masking,
erasure) for a cue that is partly legitimate: position does carry
information about tooth number.

**Error consistency.** Whether two models fail on the same inputs is
measured beyond chance by error-consistency kappa [Geirhos et al. 2020,
NeurIPS]. We report it with permutation baselines and go one
step further, comparing the wrong answers themselves.

## 3 Methods

**Data and splits.** UFBA-425 [Budagam et al., dataset] has 425 panoramic
X-rays (512 x 512 pixels) with a mask for each tooth, labeled by FDI number
(32 classes, third molars included). A tooth's box is the full extent of its
mask, which gives 11,602 teeth. We use the original X-rays only. An
augmented export of this data adds randomly cropped copies, which move
teeth within the frame and lowered position-only accuracy by about 8 points
in our earlier splits [R59]. The X-rays are split into five folds balanced
by the dataset's categories. Each training fold holds out 15% of its X-rays
for model selection, and every X-ray is tested once, by models that never
saw it. Two further splits (other random seeds) share at most 26% and 32% of any test fold with the first [R70, R96].

**Data use.** Both datasets are public and were used as released; we
collected no patient data. UFBA-425 is a set of anonymized X-rays on Figshare
under CC BY 4.0; we found no ethics statement for it. DENTEX is released by
its authors under CC BY-NC-SA 4.0, and they report ethics committee approval
for the X-rays from three hospitals [Hamamci et al.].

**Models.** The position-only baseline is scikit-learn's gradient-boosted
tree classifier (HistGradientBoosting, default settings) on six features of
the labeled box: center x and y, width, height, area and aspect ratio, all
relative to image size. It never sees pixels. The detectors are YOLOv8x
[Jocher et al.], RT-DETR-l [Zhao et al.] and Faster R-CNN with a ResNet-50
FPN backbone [Ren et al.; Li et al.], all starting from COCO weights. They
are a single-stage CNN, a transformer and a two-stage CNN, from two separate
code bases. Each is trained for 30 epochs with horizontal flips off (a flip
would swap left and right tooth numbers). YOLOv8x and RT-DETR-l read
640-pixel input; Faster R-CNN uses torchvision's default resizing (short
side 800 pixels) and SGD with a learning rate scaled to its batch of 2. We
keep the epoch with the best validation fitness (0.1 mAP50 + 0.9
mAP50:95).

**Scoring.** Detections with confidence of at least 0.5 are matched
one-to-one to labeled teeth at IoU of at least 0.5, highest confidence
first, whatever number they give. A tooth with no match counts as wrong.
Top-1 is the share of labeled teeth that are matched and given the right
number; this puts the detectors and the baseline, which always answers, on
the same teeth. We also report numbered F1, which also penalizes extra boxes
[R73]. AP50 and AP50:95 are in the code release [R89]; top-1 stays the
headline because the baseline has no scores. CIs are 95% intervals from
10,000 bootstrap draws of X-rays, paired when two models are compared. The
decision rule for each experiment was written and frozen before it was run.

A tooth is a **joint failure** when the position-only model and all three
detectors are wrong, no detector missed the tooth, and the three detectors
give the same number. The **position match** is the share of joint failures
where the position-only model gives that same number. Its chance level comes
from shuffling the position-only answers among the joint failures (10,000
permutations).

**Interventions.**

- **Shift.** The whole image is moved sideways by 10% of its width (about
  two tooth widths), and the labels move with it.
- **Context masking.** Everything outside a tooth's box, widened by k box
  widths and heights (k = 0.5, 1, 2), is blacked out.
- **Missing neighbor.** The next position in the arch has no labeled tooth.
- **Tooth erasure.** For up to three teeth per X-ray with both neighbors
  present, the tooth's mask (grown by 3 px) is filled in by inpainting
  [Telea 2004], and we ask whether a neighbor that was numbered right now
  takes the erased tooth's number. Controls are the teeth two places away
  and a sham inpaint over bone.
- **Space closure.** For each missing tooth with both neighbors present, S
  is the shortest distance between the neighbors' masks divided by the
  missing tooth's median width (S = 0 when they touch).
- **Gap closure.** After erasing a tooth, its distal neighbor (the next one
  toward the back of the mouth) is cut out and pasted back shifted toward
  the gap, by half or all of the distance at which it touches the tooth on
  the other side, with its box moved to match [R76]. In a variant, tipping,
  the neighbor is instead rotated about its root apex until its crown
  touches [R80].

## 4 Results

Table 1. Top-1 on 11,602 teeth, first split (95% CI) [R65, R73, R75, R78].

| | Top-1 | Gap over position-only (pp) | Numbered F1 | Gap, two more splits (pp) |
|---|---|---|---|---|
| Position-only | 78.5 (77.0 to 80.0) | | | |
| YOLOv8x | 95.0 (94.2 to 95.8) | 16.5 (15.1 to 17.9) | 95.0 | 15.8, 16.4 |
| RT-DETR-l | 95.2 (94.4 to 95.9) | 16.7 (15.3 to 18.1) | 92.3 | 15.9, 16.4 |
| Faster R-CNN | 92.8 (91.9 to 93.7) | 14.3 (13.0 to 15.7) | 89.6 | 13.8, 14.3 |

**Position explains most; detectors add 14 to 17 points.** Position alone
numbers more than three in four teeth. Each detector beats it on all 32
tooth numbers. The gap is smallest at the third molars (4.3 to 5.8 points)
and largest at the lower central incisors (up to 25.5) [R74]. If the
baseline is given the detectors' own boxes instead of the labeled ones, it
does slightly better (79.0 to 80.2%) and the gap is 14.5 to 16.2 points, so
the labeled boxes do not inflate its score [R66]. On the two further
splits, every gap falls inside the first split's CI [R75, R78]. YOLOv8x and
RT-DETR-l tie on top-1 (difference 0.17 points, where the smallest
difference we could detect is 0.50) [R66], but RT-DETR-l leaves about 1.9
unmatched boxes per X-ray against 0.4 for YOLOv8x, which lowers its F1 by
2.7 points [R73].

**Shared errors follow position.** 161 teeth (1.4%) are joint failures
[R65]. In 91.3% of them the position-only model gives the same wrong number
as the detectors; shuffling its answers gives at most 6.8% (95th
percentile). In 94.4% the wrong number is the next tooth over in the same
quadrant. The detectors are about as confident on these teeth as on correct
ones (AUROC 0.53 to 0.66, where 0.5 is chance), so a confidence cutoff does
not find them [R65], and neither does a rule based on the detectors' second
choices [R66]. About 10% of joint failures may be label errors [Northcutt et
al.]; without them the position match is 91.7% [R65].

**Not position in the frame, not distant teeth.** Shifting the image by two
tooth widths leaves the position-only model right on under 9% of teeth, but
changes the detectors' top-1 by under 0.3 points [R67]. Blacking out
everything beyond one box width costs Faster R-CNN and RT-DETR-l about 26
points (YOLOv8x 9.6), counting teeth that are no longer detected as wrong.
Even so, among the joint failures that are still detected, 74 to 93% keep
the shared wrong number [R68].

**Next to missing teeth.** 44.7% of joint failures have a missing neighbor,
against 8.5% of teeth that all three detectors get right (ratio 5.3, CI 4.2
to 6.6) [R69]. In 67 of the 72 joint failures next to a gap (93%), the wrong
number is the missing tooth's. Teeth that only the position-only model gets
wrong have a missing neighbor 10.5% of the time, so gaps are not where
position fails in general, only where position and the detectors fail
together.

**At closed gaps.** Table 2 and Fig. 1. Across 372 missing teeth with both
neighbors present, the space left between the neighbors separates gaps with
a joint failure from gaps without one (AUROC 0.78, CI 0.70 to 0.84; median S
0.00 against 0.32) [R72]. 56% of these errors sit at gaps whose neighbors
touch, against 20% of other gaps. Each detector's own errors follow closure
about as well (AUROC 0.74 to 0.79), and better than the position-only
model's (0.68).

Table 2. Space closure S at 372 gaps [R72].

| Outcome at the gap | Gaps | Median S, with error | Median S, without | AUROC (95% CI) |
|---|---|---|---|---|
| Joint failure | 36 | 0.00 | 0.32 | 0.78 (0.70 to 0.84) |
| YOLOv8x error | 78 | 0.03 | 0.35 | 0.76 (0.69 to 0.82) |
| RT-DETR-l error | 97 | 0.04 | 0.38 | 0.79 (0.73 to 0.84) |
| Faster R-CNN error | 98 | 0.04 | 0.36 | 0.74 (0.68 to 0.80) |
| Position-only error | 134 | 0.10 | 0.38 | 0.68 (0.62 to 0.74) |

**An open gap is not enough.** After a tooth is erased, a correctly numbered
neighbor takes its number in 2.0% (YOLOv8x), 4.4% (RT-DETR-l) and 5.7%
(Faster R-CNN) of cases, against 0 to 0.2% for both controls [R71]. But all
three detectors do it to the same neighbor in only 4 of 2,260 cases (0.18%).

**Closing the gap reproduces the shared error.** For the same erased teeth,
we moved the distal neighbor sideways until it touched the tooth on the
other side [R76]. Of 1,114 moved teeth that all three detectors numbered
right on the intact X-ray, all three gave the missing tooth's number for
0.18% with the space open, 2.9% when moved halfway and 20.8% when moved all
the way (20.6 points above open, CI 18.2 to 23.1). Each detector alone did
so 38 to 42% of the time. Models trained on the two further splits give the
same result (22.5% and 21.8% at full closure, at most 0.1% open) [R79,
R81]. Tipping the neighbor, which is closer to how real teeth drift, gives
9.4% (CI 7.7 to 11.0; 9.3% on the second split's models) [R80, R82]. At real gaps whose neighbors almost touch (S at most 0.1), 22% hold a joint failure, about 11% per neighbor [R72]. Cutting the tooth
out and pasting it back unmoved gives 0.0%, so the pasted edges do not cause
the effect. In these cases the moved tooth looks the same as before, yet the detectors number it by the slot it now occupies.

**A second dataset.** We repeated the main steps on 634 DENTEX X-rays
(18,095 teeth) from other clinics [Hamamci et al.]. Position alone numbers
66.9% of these teeth. The UFBA-trained detectors, run unchanged, lose
accuracy under this shift (YOLOv8x misses many teeth; Faster R-CNN is often
one position off), though their shared errors still follow position (87.5%)
[R85]. We therefore also trained all three on DENTEX with five-fold
cross-validation. They then beat position alone by 22 to 25 points. Their
303 shared errors give the baseline's answer 85.5% of the time (UFBA: 91.3%)
and sit next to a missing tooth 3.5 times as often as correct teeth (UFBA:
5.3) [R86]. Both fall short of the targets we set in advance (85.7% and 4.2), so we
call this a partial replication. Two more training runs on the same folds give
the same picture: margins of 24 to 26 points, a position match of 85.5% and
84.2%, and a gap ratio of 3.3 and 3.2 [R88, R93]. Part of the weaker gap link depends on
which X-rays are in the data: after weighting DENTEX to UFBA's mix of missing
teeth per X-ray, the ratio is 3.9 [R87, R94].

**Interventions on DENTEX.** They also work there, with smaller
effects (tables in the code release). At each X-ray's stored size,
sliding a neighbor into the erased tooth's space gives the missing number
15.7% of the time (0.4% open) and tipping it 7.0% (0.4% open), about three
quarters of the UFBA rates (one training run) [R92]. On X-rays shrunk to 640
pixels wide the rates are lower, mostly because Faster R-CNN reads the shrunken
X-rays less well: slide 11.8%, 12.0% and 12.6%, tip 4.8%, 5.5% and 6.5%
on three training runs. Our target of 5% for the tip was met on two of the
three [R90, R91, R92, R93].

## 5 Discussion

The detectors read teeth: shifting the image barely changes their answers, and they beat the position-only model on every tooth number. Yet their rare
shared errors are the position-only model's errors, and our results suggest
where they come from. When a tooth is missing and its neighbors have moved
together, the arch shows no gap. Counting along it gives the tooth next to
the closed space the missing tooth's number, and that tooth also sits
roughly where the missing one would. Appearance, context and position then
point the same way, and every model follows. An erased tooth leaves a
full-width empty space, which does not create this agreement; closing the
space does [R76].

Two practical points follow. First, a position-only baseline is cheap and
should be reported with tooth-numbering results: on standardized panoramic
X-rays it sets a high floor, and the margin above it shows what a model
adds. Second, confidence does not find the errors that models share. A check
from outside the model works better: look at the numbering next to every
missing tooth, especially where the space has closed. In our data this check
covers 97% of the shared errors at gaps, but it also flags many gaps without
one (precision 13%), because most gaps are partly closed [R72]. A version
that uses only a model's own output flags teeth next to a position the model
left empty, and teeth that share a number. It flags 12 to 16% of teeth and
catches 41 to 47% of the shared errors on every split, against 17 to 32% for
the same number of least-confident teeth [R84]. We had set a target of at
least 1.5 times as many as confidence, which it reached on two of three
splits. Training with simulated missing and closed teeth did not remove the
error: YOLOv8x numbered teeth next to real gaps no better than a retrained
control (+0.1 points, CI -1.6 to 1.7) [R77].

**Limitations.** UFBA is one dataset from one source, with a single
annotation per tooth and no patient data. The DENTEX replication is partial,
and its stored-size intervention results rest on one training run. The
interventions move or tip a tooth within the image plane only, but real
teeth also rotate, and outlines that touch in a flat image may not touch in
the mouth. Third molars and longer gaps, which hold most missing positions,
are outside the closure analysis. The rates are stable across splits and
training runs, but the individual teeth are not: about a third of the joint
failures change when the same model is retrained (overlap 0.63, against 0.54
to 0.57 across splits) [R78, R83]. About 10% of joint failures may be label
errors.

## 6 Conclusion

On uncropped panoramic X-rays, position alone numbers 78.5% of teeth and
detectors add 14 to 17 points. The few errors they share follow position and
concentrate next to missing teeth whose neighbors have closed the space. On a
second dataset, DENTEX, the same pattern holds with a weaker link to gaps and
smaller intervention effects. We release the per-tooth table and scorer so
that new models can be measured against the baseline and on these shared
failures.

---

## Figures

- Fig. 1: `paper/miccai/fig_closed_gap.png` / `.pdf`, from
  `make_fig_closed_gap.py`: (a) a closed gap where every model numbers
  tooth 17 as the missing 16; (b) cumulative share of gaps by space S,
  with and without a joint failure.
- The causal diagram (`causal_diagram.png`) is for the supplement or a
  talk; it does not fit in 8 pages.

## References

All references are in `paper/miccai/latex/refs.bib`. All were checked against
their sources on 2026-10-25 (Crossref for the DOIs, arXiv for the preprints,
the publisher or proceedings pages otherwise). Corrections made then:
Telea's page range is 23-34 in the DOI record (the bib had 25-36, which is
the author's PDF pagination); the page ranges of Geirhos et al. (NeurIPS 2020)
and Ren et al. (NIPS 2015) were dropped because no official source gives them;
the Shortkit-ML entry gained its DOI (10.64898/2026.04.29.26352053); author
names were completed. Checked and unchanged: the DENTEX preprint (arXiv v2,
2025), Hamamci et al. (MICCAI 2023), He et al., Mei et al., Choi et al.,
Budagam et al. (OralBBNet; UFBA-425 on Figshare, DOI 10.6084/m9.figshare.29827475.v1),
Zhao et al. (CVPR 2024, pp. 16965-16974), Li et al. (arXiv:2111.11429), Winkler
et al. (JAMA Dermatology 155(10):1135-1141), DeGrave et al., Geirhos et al.
(Nature Machine Intelligence 2:665-673), Lin et al. (MICCAI 2024, LNCS 15008,
pp. 623-633), Hill et al., Northcutt et al. (JAIR 70:1373-1411), Maier-Hein et
al. (Nature Methods 21:195-212). Jocher et al. (Ultralytics YOLOv8, 2023) is a
software citation; its authors (in the order of the project's CITATION.cff), version 8.0.0 (released 2023-01-10) and repository URL were checked against that file, whose title is "Ultralytics YOLO"; the bib names the YOLOv8 release.

LaTeX version: `paper/miccai/latex/main.tex`, compiled on Kaggle by
`make_compile_kernel.py`.

## Open before submission

- Keep this file and `latex/main.tex` in step (main.tex is the text that is submitted).
- Fill funding and conflict statements (author).
- Code and data link after acceptance (anonymized now).
