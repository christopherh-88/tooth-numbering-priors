# Tooth-numbering coordinate convention: verification protocol and case log

**Rule: never combine or compare two tooth-numbering datasets' coordinates
without independently verifying each one's own quadrant-to-image-region
convention first.** Datasets differ on this (mirrored vs. non-mirrored
image orientation, coordinate origin, patient-left-vs-viewer-left
conventions), and a silent mismatch doesn't error out - it just quietly
flips which teeth look anomalous, which quadrant looks "hardest," and
which images the position-consistency scan flags, for every downstream
number in this project. This file is the record of every convention check
run so far, so a future addition doesn't have to re-derive the protocol or
re-litigate whether a given dataset was actually checked.

## The protocol

Never assume a convention from a paper's description, a class-name list,
or "it's the same modality so it must be the same layout." Verify from the
dataset's own bulk statistics, the same way every time:

1. For every annotated instance, resolve its FDI quadrant digit through
   **that file's own category id -> name mapping** - never assume
   `category_id + 1 == quadrant digit`. This is not a theoretical
   precaution: DENTEX's own `quadrant`-only tier has category id 0 mapped
   to quadrant name `'2'` and id 1 mapped to `'1'` - swapped - while its
   sibling `quadrant_enumeration` and `quadrant_enumeration_disease` tiers
   *are* in aligned order. Three files from the same paper, same release,
   two different orderings. See `anomaly_scan/dentex_findings.md` Section
   2. Always look up by name, never by arithmetic on the id. A second,
   independent footgun in the same dataset: DENTEX's `quadrant_enumeration`
   tier gives category names as Python `int`s (`1`), while its three
   sibling files give them as `str`s (`'1'`) - code that compares a
   category name to a string literal without casting first would work on 3
   of 4 files and silently fail on the fourth. Always cast explicitly
   (`str(...)`) when building an FDI code string from a looked-up name,
   regardless of what type the source JSON seems to use elsewhere in the
   same dataset.
2. Compute the median (and 5th/95th percentile, for a natural-range sanity
   check) of normalized `(x_center, y_center)` grouped by quadrant digit,
   across the whole dataset (or as much of it as is FDI-coded).
3. Compare against the expected mirrored-radiographic layout - quadrants
   1,4 on the patient's right (image-left in a standard mirrored
   panoramic view), quadrants 2,3 on the patient's left (image-right); 1,2
   upper (maxillary), 3,4 lower (mandibular) - **as a hypothesis to check,
   not a default to assume**. A dataset that isn't mirrored, or that
   flips the vertical axis, or that was exported with a different origin
   convention, would show up here as quadrants landing in the wrong
   region relative to this expectation.
4. Only after step 3 confirms a match (or documents a mismatch and how to
   correct for it) is it safe to reuse one dataset's position-consistency
   thresholds (e.g. UFBA-425's buffered 0.45/0.55 cutoffs in
   `anomaly_scan/scan_annotations.py`) on another, or to pool position-based
   statistics across datasets for anything else in this project (the
   coordinate baseline, the go/no-go comparison, etc.).

## Case log

### UFBA-425 - reference convention (established, not inherited from anywhere)

Checked in `cpu_repro/anomaly_scan/scan_annotations.py` /
`reference_quadrant_stats.csv`, from all 27,563 tooth instances in
`Dataset/yolo_train_dataset/`. This is the convention every other check in
this project is compared against:

| quadrant | median x | median y | region |
|---|---|---|---|
| 1 | 0.366 | 0.415 | image-left, image-upper |
| 2 | 0.656 | 0.420 | image-right, image-upper |
| 3 | 0.627 | 0.659 | image-right, image-lower |
| 4 | 0.390 | 0.655 | image-left, image-lower |

Standard mirrored radiographic layout (patient's right appears on the
viewer's left). This was itself verified empirically before being used
anywhere, not assumed from the outset - see
`cpu_repro/anomaly_scan/README.md`.

### Children's Dental Panoramic Radiographs dataset (Figshare 6317013) - not applicable

No convention check possible. Verified (`cpu_repro/anomaly_scan/
children_dataset_findings.md`) that none of this dataset's three
sub-datasets annotate per-tooth FDI identity at all - "tooth" is a single
undifferentiated category everywhere, and the disease-detection tier
labels diseases, not teeth. There is no quadrant field to compute
statistics over, so the protocol above has nothing to run against. Logged
here specifically so a future reader doesn't wonder whether this dataset
was checked and passed, or checked and failed - it was checked and found
to have no convention to check.

### DENTEX (arXiv 2305.19112) - checked, matches UFBA-425

Checked in `cpu_repro/anomaly_scan/scan_dentex.py` /
`dentex_findings.md` Section 5, from all 21,806 FDI-coded tooth instances
across DENTEX's `quadrant_enumeration` and `quadrant_enumeration_disease`
tiers:

| quadrant | median x | median y | region |
|---|---|---|---|
| 1 | 0.397 | 0.415 | image-left, image-upper |
| 2 | 0.609 | 0.416 | image-right, image-upper |
| 3 | 0.606 | 0.660 | image-right, image-lower |
| 4 | 0.403 | 0.661 | image-left, image-lower |

**Matches UFBA-425's convention exactly** - same mirrored layout, similar
(if anything slightly tighter) natural spread per quadrant. On this basis,
`scan_dentex.py` reuses UFBA-425's exact buffered position-consistency
thresholds (0.45/0.55) rather than deriving new ones - a checked decision,
not a copy-paste default. If a future dataset's own statistics *don't*
match this table, do not reuse these thresholds - derive new ones from
that dataset's own natural range, following the same protocol.

## Amendments

New datasets get a new entry appended above this line, following the same
format (per-quadrant median table + match/mismatch verdict), never a
silent edit to an existing entry's numbers.
