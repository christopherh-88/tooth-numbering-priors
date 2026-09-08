"""Perceptual-hash + filename overlap check: does the Dual-Labeled
Dataset's downloaded subset share any images with UFBA-425 (relevant
since the underlying paper draws partly on UFBA-UESC)? See
inspect_labels.py's docstring for how to obtain the data - not
redistributed in this repo (third-party, ~550MB, license "unknown").
Results as run: findings.md in this directory.
"""
import os
from PIL import Image
import imagehash

DUAL_DIR = "extracted/images1"
UFBA_DIR = "/Users/christopherhuang/Documents/GitHub/tooth-numbering-priors/Dataset/bb_u_net_dataset/panoramic_x_rays"

def hash_dir(d):
    out = {}
    for fn in sorted(os.listdir(d)):
        path = os.path.join(d, fn)
        try:
            with Image.open(path) as im:
                im = im.convert("L")
                h = imagehash.phash(im, hash_size=16)
            out[fn] = h
        except Exception as e:
            print(f"  skip {fn}: {e}")
    return out

print("hashing dual-labeled images1/ ...")
dual_hashes = hash_dir(DUAL_DIR)
print(f"  {len(dual_hashes)} hashed")

print("hashing UFBA-425 panoramic_x_rays/ ...")
ufba_hashes = hash_dir(UFBA_DIR)
print(f"  {len(ufba_hashes)} hashed")

# filename overlap
dual_stems = set(os.path.splitext(f)[0].lower() for f in dual_hashes)
ufba_stems = set(os.path.splitext(f)[0].lower() for f in ufba_hashes)
fn_overlap = dual_stems & ufba_stems
print(f"\nExact filename-stem overlap: {len(fn_overlap)}")
if fn_overlap:
    print(list(fn_overlap)[:20])

# perceptual hash near-duplicate check (Hamming distance threshold)
THRESH = 8
matches = []
ufba_items = list(ufba_hashes.items())
for dfn, dh in dual_hashes.items():
    for ufn, uh in ufba_items:
        dist = dh - uh
        if dist <= THRESH:
            matches.append((dfn, ufn, dist))

print(f"\nPerceptual-hash near-duplicate matches (Hamming distance <= {THRESH}, hash_size=16 -> max dist 256): {len(matches)}")
matches.sort(key=lambda x: x[2])
for dfn, ufn, dist in matches[:50]:
    print(f"  dual:{dfn}  <->  ufba:{ufn}   dist={dist}")

print("\ndone")
