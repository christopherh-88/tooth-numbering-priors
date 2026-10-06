# MICCAI 2027 draft

Status: first full draft, 2026-10-05. Written in Markdown to settle the
content; it moves to the LNCS LaTeX template (8 pages plus references)
once the text is stable. Anonymized for double-blind review: no author
names, no repo link (the code link goes in after acceptance). Every
number cites the RESULTS.md section it comes from in a trailing comment
`[R##]`; strip these before conversion. References marked were
not checked against the source in this repo's citation audits.

---

## Title

When Tooth-Numbering Detectors Fail Together: A Position-Only Baseline
and Closed Gaps on Panoramic X-rays

## Abstract

Tooth-numbering models for panoramic X-rays report high accuracy, but
the X-ray format places each tooth in a predictable spot, so part of that
accuracy could come from position rather than appearance. We measure
this with a position-only baseline: a classifier that sees each tooth's
bounding box and no pixels. On 425 uncropped panoramic X-rays (11,602
teeth) with five-fold cross-validation grouped by X-ray, it numbers
78.5% of teeth correctly [R65]. Three detectors of different design
(YOLOv8x, RT-DETR-l, Faster R-CNN) beat it by 14.3 to 16.7 percentage
points, on all 32 tooth numbers, and the gap replicates on two further
splits [R65, R75, R78, R74]. Their errors still follow position. On 1.4% of
teeth all three detectors give the same wrong number, and in 91% of
these the position-only model gives it too, against under 7% by chance
[R65]. A sequence of interventions locates these shared errors. Shifting
the whole image barely changes the detectors' answers, and masking
distant context rarely fixes them [R67, R68]. They sit next to a missing
tooth five times as often as correct teeth and usually take the missing
tooth's number [R69], and they occur where the neighbors have closed the
space (AUROC 0.78) [R72]. Closing a gap on purpose reproduces them: when
a neighbor is moved into an erased tooth's space, all three detectors give
it the missing number 21% of the time, against 0.2% when the space is left
open [R71, R76]. We release the per-tooth table and a
scorer so new models can be compared with the baseline and on the shared
failures.

Keywords: tooth numbering, panoramic radiography, shortcut learning,
error consistency, object detection

---

## 1 Introduction

Assigning each tooth its FDI number is the first step of most automated
dental charting: a caries or a periapical lesion found by a detector is
only useful once it is attached to the right tooth. Public benchmarks such
as DENTEX [Hamamci et al.] score numbering together with detection, and
reported accuracies are high.

Panoramic X-rays are a strongly standardized format. The patient bites on
a guide, the arch runs left to right across the image, and each tooth
number has a typical location. This invites a shortcut [Geirhos et al.
2020]: a model can get many numbers right from where a box sits, without
reading the tooth. A shortcut of this kind would be invisible in
aggregate accuracy and would fail exactly on the patients who differ from
the typical arch, for example after extractions.

We ask two questions. How much of tooth numbering does position alone
explain? And when detectors fail, do they fail where position fails? Our
contributions:

1. A position-only baseline, trained in seconds on six box features,
   that numbers 78.5% of teeth correctly under grouped five-fold
   cross-validation. Three detectors beat it by 14 to 17 pp on every
   tooth number, and the result replicates on two further splits.
2. An error analysis showing that the detectors' shared errors are
   position-like: when all three give the same wrong number, the
   position-only model gives that number 91% of the time.
3. Interventions and checks (image shift, context masking, tooth
   erasure, a space-closure measure, and closing a gap on purpose) that place these errors at
   closed gaps: next to a missing tooth whose neighbors have moved
   together, so the arch looks complete.
4. A released per-tooth benchmark with folds, reference predictions and
   a scorer that reports the gap over the baseline and accuracy on the
   shared failures.

## 2 Related work

**Tooth numbering.** Detection-based numbering on panoramic X-rays is
standard, with DENTEX [Hamamci et al.] and hierarchical diffusion
detectors [Hamamci et al., HierarchicalDet] as recent examples and many
DENTEX challenge entries building on YOLO or DETR variants [He et al.;
Mei et al.; Choi et al.]. Several methods use the arch layout explicitly
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

### 3.1 Data and splits

UFBA-425 [Budagam et al., dataset] contains 425 panoramic X-rays at
512 x 512 pixels with per-tooth masks labeled by FDI number (32 classes,
third molars included). We derive each tooth's box from the full extent
of its mask, giving 11,602 teeth. We use the original X-rays only; an augmented export of this
data adds randomly cropped copies, which move teeth within the frame and
lowered position-only accuracy by about 8 pp in our earlier splits
[R59]. X-rays are split into five folds stratified by the dataset's
category labels; each training fold holds out 15% of its X-rays for model
selection, and each X-ray is tested once, by models that never saw it. Two
further splits (seeds 1 and 2) share at most about a quarter of any test
fold with the first [R70, R78].

### 3.2 Models

**Position-only baseline.** A gradient-boosted tree classifier
(scikit-learn HistGradientBoosting, default settings) on six features of
the labeled box: center x and y, width, height, area and aspect ratio,
all relative to image size. It never sees pixels. It is trained on the
training folds' labeled boxes and predicts on the test fold.

**Detectors.** YOLOv8x [Jocher et al.], RT-DETR-l [Zhao et al.]
and Faster R-CNN with a ResNet-50 FPN backbone [Ren et al.;
Li et al.], all from COCO-pretrained weights, chosen as a
single-stage CNN, a transformer and a two-stage CNN with two separate
training code bases. All are trained for 30 epochs at 640 pixels with
horizontal flipping off (a flip would swap left and right tooth numbers);
Faster R-CNN uses SGD with a learning rate scaled to its batch of 2. The
epoch with the best validation fitness (0.1 mAP50 + 0.9 mAP50-95) is kept.

### 3.3 Scoring

Each labeled tooth is matched to at most one detection: detections with
confidence of at least 0.5 are matched one-to-one at IoU of at least 0.5,
highest confidence first, regardless of class. A tooth with no match
counts as wrong. Top-1 is the share of labeled teeth matched and given
the right number, which puts detectors and the baseline (which always
answers) on the same teeth. We also report numbered F1, which counts
extra boxes [R73]. All CIs are 95% intervals from 10,000 bootstrap draws
of X-rays, paired where two models are compared. Each experiment's
decision rule was written and frozen before it was run.

A tooth is a **joint failure** when all three detectors and the
position-only model are wrong and the three detectors give the same
answer. The **position match** is the share of joint failures where the
position-only model also gives that answer. Its chance level comes from
shuffling the position-only model's answers among the joint failures
(10,000 permutations).

### 3.4 Interventions

- **Shift.** The whole image is translated by plus or minus 10% of its
  width, two tooth widths, with labels moved to match. A position-based
  detector should change its answers; one that reads teeth should not.
- **Context masking.** Everything outside a tooth's box widened by k box
  widths and heights (k = 0.5, 1, 2) is blacked out.
- **Missing neighbor.** A tooth has a gap next to it when the adjacent
  position in the arch has no labeled tooth.
- **Tooth erasure.** For up to three teeth per X-ray with both neighbors
  present, we inpaint the tooth's mask (dilated 3 px, Telea's method
  [Telea 2004]) and ask whether a neighbor that was numbered
  right now takes the erased tooth's number. Controls: teeth two places
  away, and a sham inpaint of the same shape over bone.
- **Space closure.** For each missing tooth with both neighbors present,
  S is the shortest distance between the two neighbors' masks divided by
  the missing tooth's median width (0 when they touch). We test whether S
  separates gaps with a joint failure from gaps without one.
- **Gap closure.** After erasing a target tooth, its distal neighbor is
  cut out and pasted shifted toward the gap by half or all of the
  distance at which its mask touches the tooth on the other side, with its
  box moved to match [R76].

## 4 Results

### 4.1 Position explains most, detectors add 14 to 17 pp

Table 1. Top-1 on 11,602 teeth, split 0 (95% CI) [R65, R73, R75, R78].

| | Top-1 | Gap over position-only (pp) | Numbered F1 | Splits 2, 3 gap (pp) |
|---|---|---|---|---|
| Position-only | 78.5 (76.9 to 80.0) | | | |
| YOLOv8x | 95.0 (94.2 to 95.8) | 16.5 (15.1 to 17.9) | 95.0 | 15.8, 16.4 |
| RT-DETR-l | 95.2 (94.4 to 95.9) | 16.7 (15.3 to 18.1) | 92.3 | 15.9, 16.4 |
| Faster R-CNN | 92.8 (91.9 to 93.7) | 14.3 (13.0 to 15.7) | 89.6 | 13.8, 14.3 |

Position alone numbers more than three in four teeth. Each detector beats
it on all 32 tooth numbers; the per-number gap is smallest at the third
molars (4.3 to 5.8 pp) and largest at the lower central incisors (up to
25.5 pp) [R74]. Fed the detectors' own predicted boxes instead of labeled
ones, the baseline does slightly better (79.0 to 80.2%) and the gap is
14.5 to 16.2 pp, so labeled boxes do not flatter it [R66]. On two further
splits every gap falls inside the split 0 CI [R75, R78]. YOLOv8x and RT-DETR-l tie on
top-1 (difference 0.17 pp against a minimum detectable difference of
0.50 pp) [R66], but RT-DETR-l leaves about 1.9 unmatched boxes per X-ray
against 0.4, which lowers its F1 by 2.7 pp [R73].

### 4.2 Shared errors follow position

161 teeth (1.4%) are joint failures [R65]. In 91.3% of joint failures the position-only model gives the same wrong
number, against a permutation 95th percentile of 6.8% [R65]. 94.4% of
the shared wrong numbers are one place away in the same quadrant. The detectors are
about as confident on these teeth as on correct ones (AUROC 0.53 to
0.66), so a confidence threshold does not find them [R65], and neither
does a rule based on the detectors' second choices [R66]. About 10% of
joint failures may be label errors by a confident-learning check; the
position match is 91.7% without them [R65].

### 4.3 Where the shared errors happen

**Not absolute position.** Shifting the image by two tooth widths, which
leaves the position-only model right on under 9% of teeth, changes the
detectors' top-1 by under 0.3 pp [R67].

**Not distant context.** Masking everything beyond one box width costs
Faster R-CNN and RT-DETR-l about 26 pp of top-1 on correctly numbered
teeth (YOLOv8x 9.6 pp), but among detected joint failures 74 to 93% keep
the shared wrong number [R68].

**Next to missing teeth.** 44.7% of joint failures have a missing
neighbor, against 8.5% of teeth all three detectors get right (ratio
5.3, 4.2 to 6.6) [R69]. 67 of the 72 joint failures next to a gap (93%)
take the number of the missing tooth. Teeth that only the position-only
model gets wrong have a missing neighbor 10.5% of the time, so gaps are
not where position fails in general but where position and the detectors
fail together.

**At closed gaps.** Table 2 and Fig. 1. Across 372 missing teeth with
both neighbors present, the space left between the neighbors separates
gaps with a joint failure from gaps without one (AUROC 0.78, 0.70 to
0.84) [R72]. 56% of these errors sit at gaps whose neighbors touch,
against 20% of other gaps. Each detector alone tracks closure at least as
well (AUROC 0.74 to 0.79) as the position-only model (0.68).

Table 2. Space closure S at 372 gaps [R72].

| Outcome at the gap | Gaps | Median S, with error | Median S, without | AUROC (95% CI) |
|---|---|---|---|---|
| Joint failure | 36 | 0.00 | 0.32 | 0.78 (0.70 to 0.84) |
| YOLOv8x error | 78 | 0.03 | 0.35 | 0.76 (0.69 to 0.82) |
| RT-DETR-l error | 97 | 0.04 | 0.38 | 0.79 (0.73 to 0.84) |
| Faster R-CNN error | 98 | 0.04 | 0.36 | 0.74 (0.68 to 0.80) |
| Position-only error | 134 | 0.10 | 0.38 | 0.68 (0.62 to 0.74) |

**An open gap is not enough.** Erasing a tooth by inpainting makes a
correctly numbered neighbor take its number in 2.0% (YOLOv8x), 4.4%
(RT-DETR-l) and 5.7% (Faster R-CNN) of cases, against 0 to 0.2% for both
controls [R71]. But all three detectors do it to the same neighbor in
only 4 of 2,260 cases (0.18%), while at real gaps 72 joint failures show
all three agreeing.

**Closing the gap reproduces the shared error.** For the same erased
teeth, we moved the distal neighbor sideways until it touched the tooth
on the other side [R76]. Of 1,114 moved teeth that all three detectors
numbered right on the intact X-ray, all three gave the missing tooth's
number to 0.18% with the space open, 2.9% when moved halfway and 20.8%
when moved all the way (difference 20.7 points, CI 18.2 to 23.1); each
detector alone did so 38 to 42% of the time. Models trained on a second
split give the same result (22.5% at full closure, 0.0% open) [R79].
Tipping the neighbor about its root apex until its crown touches, which
is closer to how real teeth drift, gives 9.4% (CI 7.7 to 11.0), near the
11% per neighbor at real closed gaps; cutting the tooth out and pasting
it back unmoved gives 0.0%, so paste edges do not cause the effect [R80]. A moved tooth keeps its own
shape, so the detectors number it by the slot it occupies.

## 5 Discussion

The detectors read teeth; they do not number by position in the frame,
and they beat a position-only model on every tooth. Yet their rare
shared errors are the position-only model's errors. Our results explain
why: these errors happen where the visible evidence and the typical
layout agree on the wrong answer. When a tooth is missing and its
neighbors have moved together, the arch shows no gap; counting along it
gives the tooth next to the closed space the missing tooth's number, and
that tooth also sits roughly where the missing one would. Appearance,
context and position then point the same way, and every model follows.
An inpainted gap, which leaves a full-width space, does not create this
agreement; closing the gap does [R76].

Two practical points follow. First, a position-only baseline is cheap
and should be reported with any tooth-numbering result: on standardized
panoramic X-rays it sets a high floor, and the margin above it, not the
raw accuracy, is what a model contributes. Second, the errors that
different models share are not found by confidence. A review rule from
outside the model works better: check the numbering next to every
missing tooth, in particular where the space has closed. In this data
such a rule would catch nearly all shared errors at gaps (97%) but flag
many gaps without one (precision 13%), since most gaps are partly closed
[R72]. Training does not remove the error easily: YOLOv8x retrained with
simulated missing and closed teeth numbered teeth next to real gaps no
better than a retrained control (+0.1 points, CI -1.6 to 1.7) [R77].

**Limitations.** One dataset from one source, with a single annotation
per tooth and no demographic information; we did not validate on an
external dataset such as DENTEX. The interventions slide or tip a tooth
within the image plane only (real teeth also rotate about their long
axis), and masks on a 2D projection can touch without the teeth
touching. Third-molar
gaps and longer gaps, which hold most missing positions, are outside the
closure analysis. The detectors were trained once per split; two further
splits replicate the main numbers but change split and training
randomness together. About 10% of joint failures may be label errors.

## 6 Conclusion

On uncropped panoramic X-rays, position alone numbers 78.5% of teeth,
and detectors add 14 to 17 pp. The few errors they share follow
position and concentrate next to missing teeth whose neighbors have
closed the space. We release the per-tooth table and scorer so that new
models can be measured against the baseline and on these shared
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

All references are in `paper/miccai/latex/refs.bib`. The nine marked
[verify] in earlier versions were checked by web search on 2026-10-05
(Geirhos et al. NeurIPS 2020, pp. 13890-13902; Zhao et al. CVPR 2024,
pp. 16965-16974; Ren et al. NIPS 2015, pp. 91-99; Li et al.
arXiv:2111.11429, which torchvision cites for Faster R-CNN v2; Telea,
J. Graphics Tools 9(1):25-36, 2004; Northcutt et al. JAIR 70:1373-1411,
2021; Maier-Hein et al. Nat. Methods 21:195-212, 2024; Tejani et al.
Radiol. AI 6(4):e240300, 2024; Jocher et al., Ultralytics YOLOv8,
2023). Where first names were not verified, the .bib entry uses surnames
only.

LaTeX version: `paper/miccai/latex/main.tex`, compiled on Kaggle by
`make_compile_kernel.py`.

## Open before submission

- Convert to LNCS LaTeX and check the 8-page limit.
- Fill funding and conflict statements (author).
- Code and data link after acceptance (anonymized now).
