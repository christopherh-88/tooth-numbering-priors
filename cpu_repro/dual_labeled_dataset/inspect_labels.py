"""Inspect the Dual-Labeled Dataset (Zhou et al., BMC Oral Health 2024,
Kaggle: zwbzwb12341234/a-dual-labeled-dataset) for label-91 (supernumerary
tooth) coverage. Not run against a copy of the data checked into this
repo - the dataset is third-party (license "unknown" per its own Kaggle
metadata) and ~550MB, so it isn't redistributed here.

To reproduce: `kaggle datasets download zwbzwb12341234/a-dual-labeled-dataset`,
unzip, then point LABELS_DIR/IMAGES_DIR below at the extracted
`labels/` and `images1/` folders. Results as run: `findings.md` in this
directory.
"""
import json
import os
from collections import Counter

LABELS_DIR = "extracted/labels"
IMAGES_DIR = "extracted/images1"

image_ids = set(os.path.splitext(f)[0] for f in os.listdir(IMAGES_DIR))
label_files = sorted(os.listdir(LABELS_DIR))

label_counter = Counter()
supernumerary_files = []
supernumerary_instance_count = 0
total_shapes = 0
files_with_matching_image = 0
files_without_matching_image = 0

for lf in label_files:
    stem = os.path.splitext(lf)[0]
    has_image = stem in image_ids
    if has_image:
        files_with_matching_image += 1
    else:
        files_without_matching_image += 1
    path = os.path.join(LABELS_DIR, lf)
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR reading {lf}: {e}")
        continue
    shapes = data.get("shapes", [])
    total_shapes += len(shapes)
    found_91_here = 0
    for s in shapes:
        label = str(s.get("label", ""))
        label_counter[label] += 1
        if label == "91":
            found_91_here += 1
    if found_91_here:
        supernumerary_instance_count += found_91_here
        supernumerary_files.append((stem, found_91_here, has_image))

print(f"Total label files: {len(label_files)}")
print(f"Total image files in images1/: {len(image_ids)}")
print(f"Label files with matching image in images1/: {files_with_matching_image}")
print(f"Label files WITHOUT matching image in images1/: {files_without_matching_image}")
print(f"Total shapes/instances across all label files: {total_shapes}")
print()
print(f"Distinct label values seen: {len(label_counter)}")
print("Top 40 label values by count:")
for lbl, cnt in label_counter.most_common(40):
    print(f"  {lbl}: {cnt}")
print()
print(f"Label '91' (supernumerary) total instance count: {supernumerary_instance_count}")
print(f"Number of DISTINCT image/label files containing at least one '91': {len(supernumerary_files)}")
print(f"Of those, how many have a matching image present in images1/: {sum(1 for _,_,h in supernumerary_files if h)}")
print(f"Of those, how many do NOT have a matching image in this download: {sum(1 for _,_,h in supernumerary_files if not h)}")
print()
print("Per-file breakdown (stem, count_of_91, has_image):")
for stem, cnt, has_img in supernumerary_files:
    print(f"  {stem}: {cnt} instances, has_image={has_img}")
