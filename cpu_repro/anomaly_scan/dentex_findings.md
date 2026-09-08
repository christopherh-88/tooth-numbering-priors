# DENTEX (arXiv 2305.19112, MICCAI 2023 challenge) - annotation scan findings

**Bottom line up front: YES. DENTEX contains real cases where tooth position
and tooth identity dissociate, with a full FDI label still attached to the
box. 101 of 1358 fully-FDI-coded images (7.4%) show a duplicate FDI code
and/or a position-vs-quadrant inconsistency; 27 images have at least one
box whose position contradicts its FDI quadrant, including 2 boxes wrong on
both axes (opposite-quadrant placement) and one image where 7 upper-right
teeth (11-17) are both duplicated AND all sitting ~0.65-0.68 down the image
- squarely in lower-jaw territory despite carrying upper-right FDI codes.**

Everything below was checked against the actual downloaded annotation
files, not the paper's description - including one real footgun found
along the way (Section 2).

## What was downloaded

Training and validation annotations only - not the 10.2 GB of images,
which aren't needed to answer any of the six questions except the overlap
check in Section 6 (for which a 40-image sample was pulled). Files came
from `https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/
DENTEX/`:
- `training_data.zip` (10.2 GB) - only its 3 annotation JSONs were
  extracted, via HTTP range requests (`remotezip`) against the zip's
  central directory, without downloading the archive itself:
  `training_data/quadrant/train_quadrant.json`,
  `training_data/quadrant_enumeration/train_quadrant_enumeration.json`,
  `training_data/quadrant-enumeration-disease/
  train_quadrant_enumeration_disease.json`.
- `validation_triple.json` (64 KB, downloaded directly).
- 40 sample images from `training_data.zip` (via the same range-request
  approach) for the perceptual-hash overlap check in Section 6 only;
  deleted after use, not kept in the repo.

Kept in `dentex_raw/`: the three training JSONs, `validation_triple.json`,
`build_table.py` (builds the combined per-tooth table used by the scan),
and the resulting `dentex_full_fdi_table.csv`. License: DENTEX data is
CC-BY-NC-SA per the challenge page - fine to keep small annotation JSONs
for research documentation, not for redistribution.

## 1. Schema: does enumeration give a full FDI code, joined to diagnosis on the same box?

Yes, confirmed directly from the JSON, for the fully-annotated tier only.
Each annotation in `train_quadrant_enumeration_disease.json` and
`validation_triple.json` carries `category_id_1` (quadrant, resolved
through that file's own `categories_1` id->name list), `category_id_2`
(position-in-quadrant 1-8, via `categories_2`), and `category_id_3`
(diagnosis, via `categories_3`) **all three on the same annotation
dict** - e.g. `{"category_id_1": 3, "category_id_2": 7, "category_id_3": 0,
"bbox": [...]}`. No join across files or tables is needed or possible to
get wrong - it's already one record. `quadrant_enumeration_disease` and
`validation_triple` together give **quadrant + full enumeration + diagnosis
per box**; `quadrant_enumeration` gives quadrant + full enumeration with no
diagnosis field at all; `quadrant`-only gives quadrant alone, no
enumeration digit, no diagnosis.

## 2. What's labeled in each of the three subsets, and how many teeth carry a full FDI code

Counts matched exactly what you cited (693 / 634 / 1005, with the 1005
split 705 train + 50 validation + 250 held-out test):

| tier | images | annotations | fields present | full FDI code? |
|---|---|---|---|---|
| quadrant-only | 693 | 2772 | quadrant digit only | **No** - no enumeration digit exists in this tier at all |
| quadrant+enumeration | 634 | 18095 | quadrant + position (full FDI) | Yes |
| quadrant+enumeration+diagnosis (train) | 705 | 3529 | quadrant + position + diagnosis | Yes |
| quadrant+enumeration+diagnosis (validation) | 50 | 182 | quadrant + position + diagnosis | Yes |
| quadrant+enumeration+diagnosis (test) | 250 | not public | - | held out for challenge scoring, not checked |

**21,806 teeth carry a full FDI code** across the three publicly-available
FDI-coded splits (18095 + 3529 + 182), spanning 1358 distinct images.

A characterization worth stating plainly because it affects how to read
"missing codes" in Section 4: the enumeration-only tier averages **28.5
teeth per image** (near-full-mouth enumeration, as expected), but the
diagnosis tier averages only **5.2 teeth/image in train and ~4.0 in
validation**. The diagnosis tier does not enumerate the whole mouth - it
only boxes teeth that have a diagnosable finding. That's not an annotation
failure, it's the tier's design, and it means most of that tier's "missing
FDI code" and "sparse annotation" flags in Section 4 are expected, not
anomalous - the real signal there is duplicates and position violations,
not counts.

A second, separate schema inconsistency, found during a later re-review and
not caught in the first pass: `train_quadrant_enumeration.json`'s
`categories_1` list gives category **names as integers** (`1`, `2`, `3`,
`4`), while every other file in DENTEX (`train_quadrant.json`,
`train_quadrant_enumeration_disease.json`, `validation_triple.json`) gives
them as **strings** (`'1'`, `'2'`, ...). `build_table.py` handles this
correctly (`str(q_lookup[...])` casts explicitly before concatenating into
an FDI code), but this is exactly the kind of thing that breaks silently:
code that did `category_name == '1'` instead of casting first would work on
three of DENTEX's four annotation files and silently fail to match on the
fourth, with no error - just wrong-looking counts. Verified directly:
`type(train_quadrant_enumeration.json's categories_1[0]['name'])` is `int`;
the same field in the other three files is `str`.

Image pools across tiers are **not simply additive** - checked by MD5, not
assumed. All 634 `quadrant_enumeration` filenames also exist by name in the
`quadrant` folder (which has 59 extra files), but hashing a random sample
of 40 same-named pairs found only **one** byte-identical match
(`train_0.png`) - the rest (39/40) are different physical images that
happen to reuse the same sequential index. Read as: these are two
largely-independent image pools with a filename-numbering coincidence, not
a fully shared pool - and one apparent stray duplicate at index 0. This
wasn't asked for directly but is exactly the kind of silent-corruption trap
Section 5/6 is about, one level in from where you'd expect to look for it.

## 3. Impacted teeth: count and whether they still carry an FDI number

**644 impacted-tooth annotations** (`category_id_3 == 0`, the "Impacted"
diagnosis category): 604 in the training split, 40 in validation. **Every
single one carries both `category_id_1` and `category_id_2`** - checked by
filtering for null/missing fields, found zero. Impacted teeth are not
labeled differently or left un-enumerated; they get a full FDI code exactly
like every other diagnosed tooth. This matters directly for the
position-vs-identity question: an impacted tooth is one of the few
clinically real scenarios where a tooth can be genuinely, non-erroneously
out of its textbook position (impaction often means displaced or angled
eruption) while still correctly carrying its proper FDI identity - so
DENTEX's impacted-tooth boxes are worth a specific look before assuming
every position-vs-quadrant flag in Section 4 is a labeling error rather
than a real clinical impaction. This scan does not attempt to distinguish
the two (see caveat in Section 4).

## 4. The same annotation scan (`scan_dentex.py`)

Run on all 1358 images across the two FDI-coded tiers (`quadrant_enumeration`
+ `quadrant_enumeration_disease` train/validation). Same logic as
`scan_annotations.py` (UFBA-425): duplicate-code detection, missing-code
detection (against the 32-code permanent dentition only - see Section 5 on
why not 51-85), position-vs-quadrant consistency with the buffered
threshold, and the `likely_global_framing` clustering flag. Full results:
`dentex_results/full_scan_dentex.csv` (all 1358), `candidates_dentex.csv`
(top 100).

| signal | images | total instances |
|---|---|---|
| duplicate FDI codes | 81 | 111 extra/duplicate boxes |
| any position-vs-quadrant violation | 27 | 84 violating boxes |
| ...of which wrong on **both** axes (opposite-quadrant placement) | 1 image | 2 boxes |
| duplicate codes AND a position violation in the same image | 7 | - |
| **any of the above (duplicate and/or position)** | **101 / 1358 (7.4%)** | - |

1254/1358 images have *some* nonzero score, but as flagged in Section 2,
618 are driven by `missing_codes` and 570 by `sparse_annotation` - almost
entirely the diagnosis tier's by-design partial coverage, not real
anomalies. The number that matters for "does position dissociate from
identity" is the 101 above.

Three concrete examples, read directly from `position_violation_detail`:
- **train_506.png** (score 40.0, highest in the dataset): FDI codes 31, 32,
  33, 34, 35, 36, 37, 41 each appear **exactly twice** - the entire
  lower-left quadrant plus one lower-right tooth, systematically
  double-boxed. No position violation - this looks like a duplicated
  annotation pass over the same teeth, not a position/identity mismatch.
- **train_507.png** (score 37.45): FDI codes 11-17 (upper right quadrant,
  full range) each duplicated, **and** every one of them sits at
  y=0.63-0.68 - solidly in lower-image territory, when quadrant 1 should be
  y<=0.55. This is the clearest case of the thing you asked about: seven
  boxes carrying upper-right FDI identities while sitting where the lower
  jaw is.
- **train_258.png** (score 18.5): codes 14 and 15 are each flagged **wrong
  on both axes** - found at x=0.58-0.61, y=0.65-0.66 (image-right,
  image-lower) when quadrant 1 requires image-left, image-upper. Also each
  duplicated. This is the single cleanest "opposite corner" case in the
  dataset, the DENTEX analogue of the UFBA-425 example in
  `cpu_repro/anomaly_scan/README.md`.

Caveat carried over from the UFBA-425 scan, and sharper here because DENTEX
explicitly labels impaction: a position-vs-quadrant flag is not
automatically a labeling error. An impacted tooth can be legitimately
displaced. This scan does not cross-reference `category_id_3` against
position violations to separate "mislabeled" from "genuinely impacted and
displaced" - that would be a reasonable next check before treating the 27
position-flagged images as pure label-error candidates, but it hasn't been
done here.

## 5. Quadrant convention - verified empirically, not assumed

Computed the same way as UFBA-425's `reference_quadrant_stats.csv`: median/
mean/percentile (x_center, y_center) per quadrant, from DENTEX's own 21,806
instances, checked **before** reusing any UFBA-425 threshold.

| quadrant | median x | median y | UFBA-425's own median (for comparison) |
|---|---|---|---|
| 1 | 0.397 (image-left) | 0.415 (image-upper) | x=0.366, y=0.415 |
| 2 | 0.609 (image-right) | 0.416 (image-upper) | x=0.656, y=0.420 |
| 3 | 0.606 (image-right) | 0.660 (image-lower) | x=0.627, y=0.659 |
| 4 | 0.403 (image-left) | 0.661 (image-lower) | x=0.390, y=0.655 |

**DENTEX uses the same convention as UFBA-425**: quadrants 1,4 -> image-left;
2,3 -> image-right; 1,2 -> image-upper; 3,4 -> image-lower (the standard
mirrored radiographic layout). This was checked, not assumed - see
`cpu_repro/CONVENTIONS.md` for the general reconciliation protocol this
check follows. The natural per-quadrant spread (5th/95th percentile) is
similar to or slightly tighter than UFBA-425's own, so `scan_dentex.py`
reuses UFBA-425's exact buffered thresholds (0.45/0.55) rather than
deriving new ones - a checked decision, documented in the script's
docstring, not a silent copy.

Also worth noting for anyone tempted to add primary-tooth (51-85) codes
later: DENTEX's `categories_1` never contains quadrant values above 4 in
any of the three tiers - the schema has no representation for primary
teeth at all, same conclusion as the Children's Dental Panoramic
Radiographs dataset but for a different reason (that one has no tooth
identity of any kind; this one has tooth identity but only for the
permanent-dentition quadrant range).

## 6. Image overlap with UFBA-425

**Filename convention**: trivially no overlap. DENTEX uses `train_N.png` /
`val_N.png`; UFBA-425 uses `cateN-NNNNN.jpg`. No shared naming scheme, no
shared IDs.

**Perceptual hash**: checked directly, not skipped as "obviously fine."
Computed 64-bit pHash (`imagehash.phash`, grayscale) for a random sample of
40 DENTEX images (spread across all three annotation tiers) and 60 UFBA-425
images (`Dataset/bb_u_net_dataset/panoramic_x_rays/`), then compared all
2400 cross-pairs. **Minimum Hamming distance found: 14** (out of 64 bits);
**zero pairs at or below distance 10**, the conventional near-duplicate
threshold for this hash size. No evidence of overlap in this sample.

This is a sample-based check (40 of DENTEX's ~3600 images, 60 of UFBA-425's
425), not exhaustive - it cannot rule out a shared image somewhere in the
unsampled majority with certainty. Given the datasets come from different
published sources (DENTEX: Hamamci et al., MICCAI 2023, institution not
UFBA/UESC; UFBA-425: Budagam et al., 2025, from the UFBA-UESC collection)
and show no filename or hash overlap in this check, treat them as
independent unless a future full-corpus hash comparison says otherwise.

## Practical implication for this project

DENTEX is the real second FDI-coded dataset the Children's Dental
Panoramic Radiographs dataset couldn't be (see
`children_dataset_findings.md`). It has genuine, countable position/
identity dissociation cases (101 images, both duplicate-driven and
position-driven), a verified-matching quadrant convention with UFBA-425,
and no detected image overlap - it's usable as an EXTEND target per
`cpu_repro/yolo_training/GO_NO_GO.md` if that path is triggered, and
`scan_dentex.py` / `dentex_full_fdi_table.csv` are ready for that without
rework.
