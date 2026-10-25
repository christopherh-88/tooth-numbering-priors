# Supplementary material

Released with the code. Anonymized for double-blind review. The numbers come
from the released result files in `cpu_repro/cv/results/`.

## S1. Average precision of the three detectors

Average precision (AP) is the usual detection score. Ours is class-aware: a
detection counts for the tooth number it gives, so a misnumbered box is a
false positive of that number and leaves the labeled tooth unmatched. We use
101-point interpolation, average over the 32 tooth numbers, and report AP50
(IoU 0.50) and AP50:95 (IoU 0.50 to 0.95 in steps of 0.05). 95% CIs come from
1,000 bootstrap draws of X-rays. AP is computed from the detections saved at
confidence 0.05 or higher, pooled over the five test folds. Our
implementation matches `pycocotools` to three decimals on synthetic
detections (RESULTS.md Section 97). The position-only model gives no scores,
so it has no AP.

| Set | Detector | AP50 (95% CI) | AP50:95 (95% CI) |
|---|---|---|---|
| UFBA split 0 | YOLOv8x | 94.9 (93.9 to 95.9) | 69.7 (68.9 to 70.9) |
| | RT-DETR-l | 96.7 (96.1 to 97.4) | 71.3 (70.7 to 72.4) |
| | Faster R-CNN | 95.1 (94.3 to 96.0) | 66.4 (65.7 to 67.5) |
| UFBA split 1 | YOLOv8x | 94.8 (93.8 to 95.8) | 69.7 (68.9 to 70.9) |
| | RT-DETR-l | 96.7 (96.2 to 97.4) | 71.4 (70.8 to 72.5) |
| | Faster R-CNN | 95.0 (94.1 to 96.0) | 66.5 (65.8 to 67.6) |
| UFBA split 2 | YOLOv8x | 94.7 (93.8 to 95.9) | 69.6 (68.8 to 70.9) |
| | RT-DETR-l | 96.7 (96.2 to 97.4) | 71.5 (70.9 to 72.5) |
| | Faster R-CNN | 95.0 (94.2 to 96.0) | 66.4 (65.8 to 67.5) |
| DENTEX, training seed 0 | YOLOv8x | 91.4 (90.2 to 92.7) | 51.2 (50.1 to 52.8) |
| | RT-DETR-l | 93.5 (92.7 to 94.5) | 52.5 (51.5 to 54.0) |
| | Faster R-CNN | 91.5 (90.3 to 92.9) | 49.3 (48.4 to 50.8) |
| DENTEX, training seed 1 | YOLOv8x | 91.7 (90.5 to 93.0) | 52.0 (51.0 to 53.6) |
| | RT-DETR-l | 93.5 (92.5 to 94.6) | 52.3 (51.3 to 53.8) |
| | Faster R-CNN | 91.9 (90.9 to 93.2) | 49.5 (48.7 to 51.0) |

AP and the top-1 of the main text answer different questions. AP uses every
detection from confidence 0.05 up, so it also reflects how many
low-confidence boxes a detector emits (RT-DETR-l about 290 per X-ray,
YOLOv8x about 29). Top-1 uses the single cutoff of 0.5 and puts the
detectors and the position-only baseline on the same teeth.

## S2. Interventions on DENTEX X-rays, detectors trained within DENTEX

The main text moves a neighbor into the space of an erased tooth. We repeated
this on DENTEX with the three detectors trained within DENTEX and their
position-only model, for three training seeds (S2.1 adds the stored X-ray
size). We drew 1,902 teeth from the 634 DENTEX X-rays, three per X-ray. The
procedure is the one of the main text: the tooth is erased by inpainting and
its distal neighbor is moved into the space. In the slide, the neighbor moves
sideways until it touches the tooth on the other side. In the tip, it is
rotated about its root apex until the crown touches. X-rays are resized to
640 px wide, the size YOLOv8x and RT-DETR-l read (Faster R-CNN resizes to 800
px on the short side). The pixel constants are those of UFBA (a DENTEX tooth
is 23.9 px wide against 24.0 on UFBA). The quantity j is the share of movers
(teeth that all three detectors number right on the intact X-ray) to which all
three detectors give the erased tooth's number. 95% CIs come from 10,000
bootstrap draws of X-rays. The procedure, decision rules and amendments were
fixed before any kernel ran.

| | Seed 0 | Seed 1 | Seed 2 | UFBA, seed 0 |
|---|---|---|---|---|
| Slide movers | 1,456 | 1,461 | 1,519 | 1,114 |
| j, open | 0.34 (0.07 to 0.76) | 0.21 (0.00 to 0.48) | 0.07 (0.00 to 0.20) | 0.18 (0.00 to 0.45) |
| j, half | 2.61 (1.81 to 3.43) | 1.98 (1.30 to 2.73) | 2.04 (1.32 to 2.81) | 2.87 (1.88 to 3.94) |
| **j, closed** | **11.81 (10.12 to 13.54)** | **12.05 (10.37 to 13.76)** | **12.57 (10.80 to 14.38)** | 20.83 (18.35 to 23.30) |
| Closed minus open | 11.47 (9.76 to 13.24) | 11.84 (10.18 to 13.56) | 12.51 (10.75 to 14.30) | 20.65 (18.16 to 23.14) |
| Tip movers | 1,416 | 1,424 | 1,476 | |
| j, open | 0.35 (0.07 to 0.77) | 0.21 (0.00 to 0.49) | 0.07 (0.00 to 0.21) | 0.19 |
| j, repaste (cut, pasted back unmoved) | 0.14 (0.00 to 0.36) | 0.14 (0.00 to 0.35) | 0.14 (0.00 to 0.34) | 0.0 |
| j, half tip | 1.48 (0.91 to 2.13) | 0.98 (0.50 to 1.51) | 1.02 (0.54 to 1.56) | |
| **j, full tip** | **4.80 (3.69 to 6.01)** | **5.55 (4.38 to 6.77)** | **6.50 (5.24 to 7.78)** | 9.4 (7.73 to 11.03) |
| Full tip minus open | 4.45 (3.28 to 5.68) | 5.34 (4.16 to 6.58) | 6.44 (5.17 to 7.70) | |

Our pre-set bars: the effect over the open gap must exclude 0, j must be at
least 5%, and the paste control must not explain the effect. The slide meets
them on all three seeds. The tip meets them on seeds 1 and 2 (5.55% and
6.50%) and misses by 0.2 points on seed 0 (4.80%), so it meets them on 2 of 3
seeds. We report the tip as about 5% (4.8 to 6.5%). Seed 0 is the primary
analysis, and the other seeds do not change its grade. Neither mode reaches
the lower end of UFBA's confidence interval (18.35 and 7.73%) on any seed.

Quality control: position-only agreement with the stored answers is 1.0000 on
every run. Faster R-CNN's top-1 on intact X-rays falls from 91.6, 91.9 and
91.7% at full size to 83.0, 82.9 and 85.2% at 640 px, while YOLOv8x and
RT-DETR-l change by under 0.5 points on seeds 0 and 1 and by about 0.2 points
on seed 2. The all-three condition is therefore on movers Faster R-CNN still
numbers right at that size. Without Faster R-CNN, the full-closure rates on
seed 0 are 18.9% (slide) and 11.9% (tip), against 28.5 and 18.1% on UFBA.

Which teeth fill changes between seeds. Of 266 slide fills in either of seeds
0 and 1, 82 are in both; for the tip it is 22 of 125. Over all three seeds,
57 of 341 slide fills and 12 of 174 tip fills occur in all three. Eight
targets with a repeated label were skipped on every seed.

### S2.1 Stored X-ray size

The 640-px resize costs Faster R-CNN 8.6 points of intact top-1 on seed 0
(91.6 to 83.0%). We therefore repeated the interventions on the seed 0 models
with each X-ray read at its own stored size (1,976 to 3,003 px wide; the pixel
constants scaled by the X-ray's width over 640; the detectors resize
internally as in the main runs). The same 1,902 targets and the same bars
apply. Quality control passed: intact answers at the stored size match the
stored-size scoring of the main text (Faster R-CNN 91.61 against 91.61%), and
position-only agreement is 1.0000. The 640-px runs are the pre-set primary
analysis. The stored-size run was pre-set as a sensitivity analysis: if it
came out higher, its numbers would be the effect-size claim for DENTEX and
the 640-px runs a lower bound.

| | Stored size | 640 px (seed 0) | UFBA |
|---|---|---|---|
| Slide movers | 1,599 | 1,456 | 1,114 |
| j, open | 0.38 (0.12 to 0.69) | 0.34 (0.07 to 0.76) | 0.18 |
| **j, closed** | **15.70 (13.91 to 17.55)** | 11.81 (10.12 to 13.54) | 20.83 (18.35 to 23.30) |
| Tip movers | 1,553 | 1,416 | |
| j, open | 0.39 (0.13 to 0.71) | 0.35 (0.07 to 0.77) | 0.19 |
| j, repaste | 0.19 (0.00 to 0.45) | 0.14 (0.00 to 0.36) | 0.0 |
| **j, full tip** | **7.02 (5.78 to 8.29)** | 4.80 (3.69 to 6.01) | 9.4 (7.73 to 11.03) |
| Full tip minus open | 6.63 (5.39 to 7.90) | 4.45 (3.28 to 5.68) | |

Both effects meet the main bars at the stored size (including the tip's 5%
bar, with a clean paste control) and are 75% of UFBA's, against 57 and 51% at
640 px. Neither reaches the lower end of UFBA's interval. A post hoc check,
defined after the results, shows that the increase is not an artifact of which
movers are scored and that it mostly comes from Faster R-CNN. With YOLOv8x and
RT-DETR-l alone, which letterbox to 640 px either way, the full tip is 12.06
against 11.91% and the full slide 20.07 against 18.93% (RESULTS.md Section
95). With those two detectors, about a third of UFBA's effect is still missing
on DENTEX, so the resize explains the Faster R-CNN part of the difference and
not all of it.

### S2.2 Third training seed (observational result)

A third training seed on the same folds gives the same result as seeds 0 and
1: margins of 24.8 to 26.2 points, a position match of 84.2% (bar 85.71%) and
a gap ratio of 3.22 (bar 4.2). So 1 of 3 bars is met in all three runs. 213 of
the 410 teeth that are a shared error in any run are shared errors in all
three. Its slide and tip results are in the table of S2.
