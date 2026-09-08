# Annotation screening scan

A screening pass over `Dataset/yolo_train_dataset/` looking for annotations
where "position implies FDI code" should break: duplicate codes, missing
codes, and boxes whose position contradicts their FDI quadrant.

**This is a data-quality screen, not a clinical diagnosis.** It flags
candidates for a human (or a more careful downstream check) to look at; it
says nothing about the patient.

## Results at a glance

- 1021 label files scanned (425 unique source X-rays, incl. Roboflow
  augmented crops - see "Unit of analysis" below).
- 692 / 1021 have a nonzero anomaly score, but 586 of those are flagged
  *only* for missing FDI codes (see below - this is expected/common in this
  dataset and barely moves the score).
- Genuinely notable counts (`primary_reason`, i.e. the single largest
  contributor to that image's score):

  | primary_reason | count | meaning |
  |---|---|---|
  | missing_codes | 586 | least-erupted-wisdom-tooth-type gaps; expected, low weight |
  | sparse_annotation | 41 | fewer than 8 teeth annotated at all |
  | single_axis_position | 35 | one or more boxes clearly on the wrong side/level for their code - see caveat below |
  | duplicate_codes | 20 | same FDI code used more than once in one image |
  | wrong_quadrant | 10 | a box wrong on **both** axes - sits in the anatomically opposite region entirely |

Full ranked list: `full_scan.csv` (all 1021). Top 100: `candidates.csv`.

## What each check does

1. **Total tooth count** - `total_tooth_count`, boxes annotated in that image.
2. **Duplicate FDI codes** - `duplicate_codes` / `n_duplicate_codes`: the
   same class_id used on more than one box in one image. Two teeth cannot
   share one FDI number, so any repeat is a genuine labeling contradiction.
3. **Missing FDI codes** - `missing_codes` / `n_missing_codes`, relative to
   the full 32-code adult dentition. This is reported for every image but
   weighted very lightly (0.05/code) in the score, because partial
   dentition is common and expected by design in UFBA-425 (several of the
   dataset's own categories are explicitly missing teeth / implants / >32
   teeth - see the repo's top-level README) - it's context, not evidence of
   a labeling error on its own.
4. **Position-vs-code inconsistency** - `position_violation_detail` /
   `n_position_violations`: does a box's (x, y) sit in the region its FDI
   quadrant implies? The convention was confirmed empirically from the
   dataset's own bulk statistics (`reference_quadrant_stats.csv`), not
   assumed:

   | FDI quadrant | expected image region |
   |---|---|
   | 1 (upper right) | image-left, image-upper |
   | 2 (upper left) | image-right, image-upper |
   | 3 (lower left) | image-right, image-lower |
   | 4 (lower right) | image-left, image-lower |

   (This is the standard mirrored radiographic convention: patient's right
   appears on the viewer's left.) A box is flagged if it's clearly on the
   wrong side of x=0.5 and/or the wrong side of y=0.5, using a ~0.05 buffer
   past the dataset's own natural 5th/95th-percentile range per quadrant, so
   ordinary variation near the midline or occlusal plane isn't flagged -
   only boxes that land solidly in the opposite region. Wrong on **both**
   axes (`n_wrong_quadrant_both_axes`) - e.g. an upper-left code found deep
   in the lower-left region - is the clearest, highest-confidence case and
   is weighted highest.

## Important caveat: `likely_global_framing`

Manually reviewing the top hits surfaced a real confound, so a column was
added for it rather than leaving it implicit. Some images have *several*
boxes all crossing the position threshold on the *same axis* by a small,
similar margin (e.g. five different upper-quadrant teeth all sitting at
y=0.55-0.58, just past the y<=0.55 cutoff). That pattern looks like the
whole source X-ray is rotated, tilted, or cropped slightly - shifting where
the true upper/lower or left/right boundary falls for that one image - far
more than it looks like five independent per-tooth labeling mistakes.

`likely_global_framing` is `True` when 3+ single-axis violations in the same
image land within 0.15 of each other on that axis. In this scan, **every**
`single_axis_position`-primary image in the top 20 is flagged
`likely_global_framing=True`, while every `duplicate_codes` and
`wrong_quadrant` case is `False`. Recommended reading order for manual
review:

1. `wrong_quadrant` (both axes) - highest confidence, genuine per-tooth
   contradictions (example: a box labeled `23`, upper-left canine, found at
   x=0.127, y=0.776 - the diagonally opposite corner of the image).
2. `duplicate_codes` - unambiguous: the same code cannot appear twice.
3. `single_axis_position` with `likely_global_framing=False` - isolated,
   larger single-tooth deviations; still worth a look.
4. `single_axis_position` with `likely_global_framing=True` - probably a
   rotated/cropped/tilted source image, not per-tooth errors. Lower
   priority for annotation fixes; possibly useful as a separate "framing
   outlier" list if you're doing image-level QA instead of label QA.
5. `sparse_annotation` / `missing_codes` - context, not contradictions.

## Anomaly score

A simple, additive, fully documented heuristic for sorting candidates - not
a statistical test. See `WEIGHTS` at the top of `scan_annotations.py`:

| signal | weight |
|---|---|
| each redundant box beyond the first, per duplicated code | +3.0 |
| each box wrong on both axes (opposite-quadrant contradiction) | +5.0 |
| each box wrong on exactly one axis | +2.0 |
| each missing FDI code | +0.05 |
| total tooth count < 8 (flat penalty) | +5.0 |
| each tooth beyond 32 (anatomically impossible count) | +2.0 |

`primary_reason` is the single largest contributor to that image's score;
`reason_breakdown` shows every contributor.

## Unit of analysis

One row per YOLO label **file**, not per source X-ray. Roboflow generated
~2-3 augmented crops per source image (e.g. `cate1-00002_jpg.rf.<hash>.txt`
appears 3x with slightly different box coordinates, all derived from the
same original `cate1-00002.jpg`) - each is scanned as its own entry because
each is literally a separate image a detector would see, and because
collapsing them would require picking one arbitrary "canonical" copy for no
clear benefit. The `image_id` column (filename before `_jpg.rf.<hash>`)
lets you tell which rows are augmented copies of the same source X-ray -
e.g. `cate2-00109` appears 3x in the top 5 duplicate-code hits, meaning the
underlying annotation error was present in the original labeling and
propagated through all of Roboflow's augmented copies of it (not something
the augmentation introduced).

## Rerun

```bash
cd /Users/christopherhuang/Documents/GitHub/tooth-numbering-priors
source .venv312/bin/activate   # or your own env with cpu_repro/requirements.txt installed
python cpu_repro/anomaly_scan/scan_annotations.py
```

Takes a few seconds. Edit the threshold/weight constants at the top of
`scan_annotations.py` to adjust sensitivity.

## Files

- `scan_annotations.py` - the script.
- `full_scan.csv` - every label file (1021 rows), ranked by anomaly score.
- `candidates.csv` - top 100 of the above, the "candidate list."
- `reference_quadrant_stats.csv` - the empirical median/mean/std of
  (x_center, y_center) per FDI quadrant used to confirm the position
  convention above.
