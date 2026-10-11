# Supplementary material

Anonymized for double-blind review. Numbers are from the released result
files (`cpu_repro/cv/results/ap/`).

## S1. Average precision of the three detectors

COCO-style AP, class-aware: a detection counts for the tooth number it
gives, so a misnumbered box is a false positive of that number and leaves
the labeled tooth unmatched. 101-point interpolation, mean over the 32 tooth
numbers; AP50 uses IoU 0.50 and AP50:95 averages IoU 0.50 to 0.95 in steps
of 0.05. 95% CIs from 1,000 bootstrap draws of X-rays. Computed from the
detections saved at confidence 0.05 or higher, pooled over the five test
folds; the implementation reproduces `pycocotools` to three decimals on
synthetic cases. The position-only model gives no scores and has no AP.

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
YOLOv8x about 29); top-1 uses the single cutoff of 0.5 and puts the
detectors and the position-only baseline on the same teeth.

## S2. Interventions on DENTEX X-rays, detectors trained within DENTEX

The slide and the tip of the main text, repeated on 1,902 teeth drawn in
the 634 DENTEX X-rays (three per X-ray), with the three detectors trained
within DENTEX and their position-only model, for two training seeds. The
procedure is the one of the main text: the tooth is erased by inpainting,
and its distal neighbor is moved into the space (slide: sideways until it
touches the tooth on the other side; tip: about its root apex until the
crown touches). X-rays are resized to 640 px wide, the size the detectors
read; the pixel constants are those of UFBA (a DENTEX tooth is 23.9 px
wide against 24.0 on UFBA). j is the share of movers, teeth that all three
detectors number right on the intact X-ray, to which all three give the
erased tooth's number. 95% CIs from 10,000 bootstrap draws of X-rays.
Procedure, decision rules and amendments were fixed before any kernel ran.

| | Seed 0 | Seed 1 | UFBA, seed 0 |
|---|---|---|---|
| Slide movers | 1,456 | 1,461 | 1,114 |
| j, open | 0.34 (0.07 to 0.76) | 0.21 (0.00 to 0.48) | 0.18 (0.00 to 0.45) |
| j, half | 2.61 (1.81 to 3.43) | 1.98 (1.30 to 2.73) | 2.87 (1.88 to 3.94) |
| **j, closed** | **11.81 (10.12 to 13.54)** | **12.05 (10.37 to 13.76)** | 20.83 (18.35 to 23.30) |
| Closed minus open | 11.47 (9.76 to 13.24) | 11.84 (10.18 to 13.56) | 20.65 (18.16 to 23.14) |
| Tip movers | 1,416 | 1,424 | |
| j, open | 0.35 (0.07 to 0.77) | 0.21 (0.00 to 0.49) | 0.19 |
| j, repaste (cut, pasted back unmoved) | 0.14 (0.00 to 0.36) | 0.14 (0.00 to 0.35) | 0.0 |
| j, half tip | 1.48 (0.91 to 2.13) | 0.98 (0.50 to 1.51) | |
| **j, full tip** | **4.80 (3.69 to 6.01)** | **5.55 (4.38 to 6.77)** | 9.4 (7.73 to 11.03) |
| Full tip minus open | 4.45 (3.28 to 5.68) | 5.34 (4.16 to 6.58) | |

Pre-set bars: the effect over open excludes 0 and j is at least 5%, with the
paste control not confounded. The slide meets it on both seeds. The tip meets
it on seed 1 (5.55%) and misses it on seed 0 by 0.2 points (4.80%), so the
tip is reported as a smaller effect that straddles the bar; seed 0 is the
primary analysis and its grade is not changed by seed 1. Neither mode reaches
the lower end of UFBA's confidence interval (18.35 and 7.73%). Quality
control: position-only agreement with the stored answers 1.0000 on every run;
Faster R-CNN's top-1 on intact X-rays falls from 91.6 and 91.9% at full size to
83.0 and 82.9% at 640 px (YOLOv8x and RT-DETR-l change by under 0.5 points),
so the all-three condition is on movers Faster R-CNN still numbers right at
that size; without Faster R-CNN the full-closure rates are 18.9% (slide) and
11.9% (tip) on seed 0 against 28.5 and 18.1% on UFBA. Teeth that fill
differ between seeds: of 266 slide fills in either seed, 82 are in both; of
125 tip fills, 22. 8 targets with a repeated label were skipped.
