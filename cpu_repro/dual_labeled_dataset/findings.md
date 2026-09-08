# Dual-Labeled Dataset (Zhou et al., BMC Oral Health 2024) - findings

**Source:** Kaggle `zwbzwb12341234/a-dual-labeled-dataset` (paper:
"Combining public datasets for automated tooth assessment in panoramic
radiographs," BMC Oral Health, 2024). License: "unknown" per the
dataset's own Kaggle metadata (re-verified live, 2026-09-08) - state this
plainly in any writeup that cites it, don't assume a permissive license.
**Downloaded and analyzed:** 2026-09-08.

## Partial-release status (verify before assuming full 2,000-image access)

Exact quote from the dataset's own `dataset-metadata.json` description
(re-fetched live 2026-09-08 via `kaggle datasets metadata`, not relied on
from memory):

> "Our dataset comprises a total of 2,000 panoramic radiographs... A
> total of 33 different numbering labels were used... Currently, 500
> panoramic images and their label files have been uploaded. For the
> remaining data, please contact the author to obtain it
> (wbzhou23@mails.jlu.edu.cn)."

Only the 500-image tranche was downloaded and is reflected below - this
is an admitted partial release, not the full stated dataset.

## Label 91 (supernumerary) coverage

**Script:** `inspect_labels.py` · run output:

- 2,066 total label files present in the download; 500 have a paired
  image in `images1/` (the other 1,566 label files have no matching
  image in this 500-image tranche).
- 58,269 total annotated shapes/instances across all label files; 33
  distinct label values (32 standard FDI codes + "91").
- **Label "91": 53 total instances across 49 distinct label files.**
- **Of those 49 files, only 23 have a paired image in this download** -
  those 23 files contain **24 total "91" instances** (one file,
  `4252.json`, has 2). The other 26 files (30 instances) have no image in
  this tranche and are not usable.
- **Usable supernumerary evidence in this download: 24 instances across
  23 images** - not "292 across 30" or any other figure not derived from
  this run; this is the real number this specific 500-image tranche
  supports.

Full per-file breakdown and label-value counts: see the captured run
output at the bottom of this file (script is reproducible against a
fresh download - see `inspect_labels.py`'s docstring).

## Overlap with UFBA-425

**Script:** `phash_overlap.py` · compares all 500 downloaded images
against UFBA-425's 425 panoramic X-rays (`Dataset/bb_u_net_dataset/panoramic_x_rays/`),
by exact filename-stem match and by perceptual hash (`imagehash.phash`,
hash_size=16, Hamming distance <= 8 threshold for a near-duplicate).

- **Exact filename-stem overlap: 0.**
- **Perceptual-hash near-duplicate matches: 0** (out of 500 x 425 =
  212,500 pairwise comparisons).

No detected overlap with UFBA-425 in this downloaded tranche. This
supports treating the two as independent datasets for any future
cross-dataset use, though it does not rule out overlap in the
undownloaded remainder of the 2,000-image set.

## What this does and does not support for the clinical-stakes argument

This is **existence-proof, not measured evidence**: it confirms 24 real,
annotated supernumerary-tooth instances exist in a public dataset, which
is relevant substrate for the clinical-stakes argument in `paper/DRAFT.md`
(a coordinate-shortcutting model would plausibly mislabel exactly these
anomalous-position cases). **No accuracy comparison has been run on
them** - nothing here shows the coordinate-only model (or any detector)
is actually worse on these 24 instances than on typical ones. Turning
this into real evidence would require running the existing coordinate-
only model's features against these 24 instances (after converting the
labelme polygon annotations to the same bounding-box feature
representation used elsewhere in this project) and checking whether
predicted-FDI-vs-true-FDI error rates are elevated on them specifically -
not yet done, and not attempted as part of this write-up.

## Raw run output (inspect_labels.py, 2026-09-08)

```
Total label files: 2066
Total image files in images1/: 500
Label files with matching image in images1/: 500
Label files WITHOUT matching image in images1/: 1566
Total shapes/instances across all label files: 58269

Distinct label values seen: 33
Label '91' (supernumerary) total instance count: 53
Number of DISTINCT image/label files containing at least one '91': 49
Of those, how many have a matching image present in images1/: 23
Of those, how many do NOT have a matching image in this download: 26
```

## Raw run output (phash_overlap.py, 2026-09-08)

```
hashing dual-labeled images1/ ...
  500 hashed
hashing UFBA-425 panoramic_x_rays/ ...
  425 hashed

Exact filename-stem overlap: 0
Perceptual-hash near-duplicate matches (Hamming distance <= 8, hash_size=16 -> max dist 256): 0
```
