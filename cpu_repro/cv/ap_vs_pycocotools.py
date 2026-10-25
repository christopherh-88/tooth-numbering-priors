"""Check ap_eval.py against pycocotools on synthetic detections (RESULTS.md Section 97).

    pip install pycocotools   # not in requirements.txt
    python ap_vs_pycocotools.py

For three seeds: 80 X-rays of boxes.csv, detections made from their labeled boxes (jitter,
8% missed, 10% misnumbered by one, 15% extra boxes, random confidence). Prints AP50 and AP50:95
from ap_eval.py and from pycocotools' COCOeval (class-aware, default settings).
"""
import contextlib
import io
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import ap_eval
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

boxes = pd.read_csv(HERE / "boxes.csv", dtype={"fdi": str})
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed)
    ids = sorted(boxes.image_id.unique())
    ids = list(rng.choice(ids, 80, replace=False))
    b = boxes[boxes.image_id.isin(ids)].reset_index(drop=True)
    W, H = 1000.0, 500.0
    rows = []
    for _, r in b.iterrows():
        x1, y1 = r.x_center - r.width / 2, r.y_center - r.height / 2
        x2, y2 = r.x_center + r.width / 2, r.y_center + r.height / 2
        u = rng.random()
        if u < 0.08:   # missed
            continue
        j = rng.normal(0, 0.12, 4) * np.array([r.width, r.height, r.width, r.height])
        cls = int(r.class_id)
        if rng.random() < 0.10:  # misnumbered
            cls = int(np.clip(cls + rng.choice([-1, 1]), 0, 31))
        rows.append((r.image_id, cls, float(np.clip(rng.beta(5, 1.5), 0.05, 1)), x1 + j[0], y1 + j[1], x2 + j[2], y2 + j[3]))
        if rng.random() < 0.15:  # duplicate / extra box
            rows.append((r.image_id, int(r.class_id), float(rng.uniform(0.05, 0.6)), x1 + 0.3 * (x2 - x1), y1, x2 + 0.3 * (x2 - x1), y2))
    d = pd.DataFrame(rows, columns=["image_id", "class_id", "conf", "x1", "y1", "x2", "y2"])
    per, n_img = ap_eval.collect(d, b)
    mine50, mine5095 = ap_eval.mean_ap(per, np.ones(n_img))
    # COCO
    img_id = {i: k + 1 for k, i in enumerate(sorted(b.image_id.unique()))}
    images = [dict(id=v, width=int(W), height=int(H)) for v in img_id.values()]
    anns = []
    for k, r in b.iterrows():
        x1, y1 = (r.x_center - r.width / 2) * W, (r.y_center - r.height / 2) * H
        w, h = r.width * W, r.height * H
        anns.append(dict(id=k + 1, image_id=img_id[r.image_id], category_id=int(r.class_id), bbox=[x1, y1, w, h], area=w * h, iscrowd=0))
    gt = dict(images=images, annotations=anns, categories=[dict(id=c) for c in range(32)])
    dts = [dict(image_id=img_id[r.image_id], category_id=int(r.class_id), bbox=[r.x1 * W, r.y1 * H, (r.x2 - r.x1) * W, (r.y2 - r.y1) * H], score=float(r.conf)) for r in d.itertuples()]
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(); cg.dataset = gt; cg.createIndex()
        cd = cg.loadRes(dts)
        E = COCOeval(cg, cd, "bbox"); E.evaluate(); E.accumulate(); E.summarize()
    print(f"seed {seed}: teeth {len(b)}, dets {len(d)} | AP50 mine {mine50:.3f} coco {100*E.stats[1]:.3f} | AP50:95 mine {mine5095:.3f} coco {100*E.stats[0]:.3f}")
