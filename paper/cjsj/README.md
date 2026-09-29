# CJSJ paper

"Geometry Alone Predicts Tooth Identity in Panoramic Dental X-Rays, but Real
Detectors Largely Do Not Exploit It" (C. Huang and K. Dua), the short version
of this project written for the Columbia Junior Science Journal (original
research, two-column CJSJ template, 2 to 3 pages plus figures and references).
The manuscript itself (`HuangChristopher_paper.docx`) and the figures file
(`HuangChristopher_figures.pptx`) are kept with the submission, not in this
repository.

## Files

- `paper_numbers.py` recomputes every number in the paper from committed
  result files and prints it next to the value written in the paper. It exits
  with status 1 on any mismatch. The only numbers it cannot recompute are the
  supernumerary-tooth check, which needs the third-party dual-labeled dataset
  (`RESULTS.md` Section 17); it prints those with their source.
- `make_cjsj_figures.py` builds `fig1.png` to `fig4.png` (600 dpi, one column
  wide, Times New Roman when installed) from committed files and one UFBA-425
  image.
- `fig1.png` to `fig4.png` are the figures as submitted.

```bash
python paper/cjsj/paper_numbers.py
python paper/cjsj/make_cjsj_figures.py
```

## Where each figure comes from

| Figure | Content | Source |
|---|---|---|
| 1 | UFBA-425 image `cate8-00390` with its 31 labeled boxes, colored by FDI quadrant | `Dataset/yolo_train_dataset/train/{images,labels}/cate8-00390_jpg.rf.eb1ab1028af98aa88e55c7302014dc54.*` |
| 2 | Top-1 accuracy on identical held-out teeth, T4 seeds 0 to 4 (bars: mean, error bars: range) | `eval_results/{multiseed,rtdetr_multiseed,fasterrcnn_multiseed}/*_summary.csv`, `coord_baseline/summary.csv`; `RESULTS.md` Sections 33, 40, 45 |
| 3 | Close-up of the Fig. 1 image: tooth 11 (position-only model said 21, YOLOv8x said 11) | `eval_results/case_study_crops/manifest.csv` (case 07); Section 23 |
| 4 | Same-wrong-tooth agreement and geometry match on teeth every model got wrong, three seed groups, with the permutation chance range | `eval_results/cross_architecture_agreement_summary.csv`, `cross_architecture_agreement_nulls.csv`; Sections 46, 51, 52, 58 |

"Seeds 0 to 4" are the main Kaggle T4 runs, "5 to 9" the Apple M3 replication
and "11 to 14" the extra T4 replication. Seed 10 has no three-detector table
(Section 52), so it is not in Fig. 4.
