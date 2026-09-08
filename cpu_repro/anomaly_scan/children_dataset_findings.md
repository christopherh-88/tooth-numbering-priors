# Children's Dental Panoramic Radiographs Dataset - annotation scan applicability

**Verdict up front: this dataset does not contain usable anomaly cases for
the FDI-numbering scan in this directory. It has no per-tooth FDI numbering
at all - not permanent, not primary, not mixed. The scan literally has
nothing to check.**

## What was downloaded and inspected

Figshare collection 6317013 ("Children's Dental Panoramic Radiographs
Dataset for Caries Segmentation and Dental Disease Detection", Zhang et al.,
*Scientific Data* 2023, DOI 10.1038/s41597-023-02237-5). The collection's
main data article is 21621705; its single file (`Dental dataset.zip`, 1.65
GB, CC0) was downloaded from `https://ndownloader.figshare.com/files/38322366`
and inspected directly (not just read about) - every annotation schema
claim below is from the actual JSON files, not the paper's prose. The
download was deleted after inspection (1.7 GB, not something to keep in
this repo); this file is the record of what was found.

The archive contains three independent sub-datasets, none of which use FDI
numbering:

| sub-dataset | images | annotation format | categories (verified from the JSON itself) |
|---|---|---|---|
| Pediatric dental disease detection | 100 (70 train / 30 test) | LabelMe-format JSON, rectangles | 6 disease labels: 龋病 (caries, 314 boxes), 深窝沟 (deep sulcus, 35), 根尖周炎 (periapical infection, 34), 牙髓炎 (pulpitis, 29), 牙齿发育异常 (developmental abnormality, 24), 其他 (other, 12) |
| Children's dental caries segmentation | ~93+ (Train/Test/"Supplemental content-93") | COCO-format JSON | single category, "龋齿" / "龋齿分割" (caries / caries-segmentation) - a lesion class, not a tooth class |
| Adult tooth segmentation | 1417 (1224 train / 193 test) | COCO-format JSON | single category, `{"name": "tooth", "id": 1}` - every tooth is just "tooth," no identity |

None of the three has a field anywhere that plays the role `class_id` /
FDI code plays in `Dataset/yolo_train_dataset/` for UFBA-425. The "disease
detection" dataset's boxes are diseases, located on an image, with no tooth
identity attached at all (a caries box isn't linked to "which tooth"). The
two segmentation datasets go the other way: every tooth (or every caries
lesion) gets the *same* category id, so there's no way to tell tooth 14
from tooth 46 even though they're both segmented - the label carries no
identity information, only "this pixel region is a tooth" (or "is caries").

Side note, not related to the original ask: the "Adult tooth segmentation
dataset" images are named `cateN-NNNNN.jpg` (e.g. `cate8-00099.jpg`,
`cate10-00015.jpg`) - the exact same naming convention as this repo's
`Dataset/` (UFBA-UESC-derived). It's very likely sourced from the same
broader UFBA-UESC collection UFBA-425 is a curated subset of, just with a
different (and much coarser) annotation scheme applied and a different
image count (1417 vs. 425) - almost certainly not usable as a second FDI-
numbering dataset either, for the same reason: no tooth identity labels.

## The four things asked for

**1-4. Total tooth count / duplicate FDI codes / missing FDI codes /
position-vs-FDI-code consistency.** Not computable. All four checks in
`scan_annotations.py` are keyed on `class_id` being an FDI code
(`FDI_CODES[class_id]`). There is no such field in any of this dataset's
three annotation schemas - "duplicate FDI code" and "missing FDI code
relative to full dentition" are meaningless questions when no annotation
ever claims a box *is* a particular tooth. Total per-image box/instance
count is computable (see table above: caries dataset ~4.5 disease boxes/
image on average, min 1 max 16; adult segmentation ~36 tooth instances/
image), but "count" alone was never the anomaly signal here - it was
always paired with "count relative to what a full adult dentition should
have," which requires knowing *which* teeth are present, not just how many
boxes exist.

**Mixed dentition (both primary and permanent teeth present).** Cannot be
counted from the annotations, for the same reason - there is no per-tooth
label of any kind, so there is nothing to check "is this a primary or
permanent tooth" against. The paper states the cohort is ages 2-13, a
range that clinically spans exactly the mixed-dentition period (first
permanent molars erupt around age 6; the last primary teeth are typically
exfoliated by around age 12), so mixed dentition is very likely present in
a large fraction of these 100-193 images on clinical grounds - but that is
an inference from the stated age range in the paper, not something this
scan (or any script) can verify from the data itself. No per-image age
metadata is included in the archive (no README/CSV with per-patient ages
was found - checked explicitly), so even that inference can't be narrowed
down to a specific count or specific images.

**Does the annotation scheme support primary teeth (FDI 51-85)?** No -
and it's not "permanent only" either, which would at least be a
comparison point. It's neither. There is no FDI numbering scheme of any
kind in this dataset - not the 11-48 permanent range, not the 51-85
primary range. "Tooth" is a single undifferentiated category everywhere a
tooth is labeled at all.

## Bottom line

This is a real, useful, well-documented dataset - just for a different
task (disease/caries detection and generic tooth segmentation) than the
one `cpu_repro/anomaly_scan/scan_annotations.py` was built to screen
(FDI-numbering label-consistency). It is not a usable EXTEND target for
the coordinate-baseline-vs-YOLO numbering comparison either, for the same
root reason: there's no FDI-coded ground truth to train or evaluate a
numbering baseline against. Finding a second dataset for that purpose
means finding one with per-tooth FDI (or equivalent named tooth-identity)
labels - a segmentation/detection-only dataset like this one, however well
constructed, doesn't carry the signal this project's scans and baselines
are built around.
