"""Score a tooth-numbering detector against the position-only baseline.

    python benchmark/score.py --pred my_detections.csv [--split 0|1] [--out report.csv]

my_detections.csv: one row per detection on the test X-rays of each fold
(each X-ray scored by the model that did not train on it), columns
    image_id, class_id, conf, x_min, y_min, x_max, y_max
with class_id 0..31 in the order of FDI 11..18, 21..28, 31..38, 41..48 and
box corners as fractions of image width and height.

Matching as in the paper: detections with conf >= 0.5, one-to-one greedy
matching to labeled boxes at IoU >= 0.5, highest confidence first,
class-agnostic. A labeled tooth with no match counts as wrong.

Reports, with 95% CIs from 10,000 bootstrap draws of X-rays:
  top1             share of labeled teeth matched and numbered right
  missed, misnumbered
  gap_pp           top1 minus the position-only model's top1 (paired)
  top1_next_to_gap teeth with an unlabeled arch neighbor
  top1_joint       the joint-failure teeth (all three reference detectors
                   and the position-only model wrong with one shared answer)
  repeats_shared   of the joint-failure teeth this model gets wrong, the
                   share where it gives the same shared wrong answer
  false_pos_per_xray  kept detections (conf >= 0.5) that are not a
                   correctly numbered match, per X-ray
  f1_numbered      2 TP / (2 TP + FP + FN), TP = correctly numbered teeth,
                   FP as above, FN = labeled teeth not correctly numbered
  top1_macro       mean of the 32 per-class top-1 values (no CI)
top1 ignores extra boxes; f1_numbered does not.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONF, MATCH_IOU, N_BOOT = 0.5, 0.5, 10_000


def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def match(gt, cls, conf, box):
    """Matched class per labeled box (nan if missed), and the number of kept detections."""
    pred = np.full(len(gt), np.nan)
    keep = conf >= CONF
    cls, conf, box = cls[keep], conf[keep], box[keep]
    if len(cls) == 0:
        return pred, 0
    m = iou(gt, box)
    used_g, used_d = set(), set()
    for _, _, i, j in sorted(((conf[j], m[i, j], i, j) for i, j in zip(*np.nonzero(m >= MATCH_IOU))), reverse=True):
        if i in used_g or j in used_d:
            continue
        used_g.add(i)
        used_d.add(j)
        pred[i] = cls[j]
    return pred, len(cls)


def boot(per_image, stat, rng):
    v = per_image.to_numpy(float)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    return stat(v.sum(0)), *np.nanpercentile(stat(v[idx].sum(1).T), [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--split", type=int, default=0, choices=[0, 1],
                    help="which CV split the predictions follow (the position-only model is matched to it)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    t = pd.read_csv(HERE / "teeth.csv", dtype={"fdi": str})
    d = pd.read_csv(a.pred)
    missing = set(t["image_id"]) - set(d["image_id"])
    if missing:
        print(f"note: {len(missing)} X-rays have no detections; their teeth count as missed")

    pred = np.full(len(t), np.nan)
    kept = pd.Series(0, index=sorted(t["image_id"].unique()))
    by_image = dict(tuple(d.groupby("image_id")))
    for image_id, g in t.groupby("image_id"):
        gt = np.stack([g.x_center - g.width / 2, g.y_center - g.height / 2,
                       g.x_center + g.width / 2, g.y_center + g.height / 2], 1)
        if image_id in by_image:
            x = by_image[image_id]
            pred[g.index], kept[image_id] = match(gt, x["class_id"].to_numpy(int), x["conf"].to_numpy(float),
                                  x[["x_min", "y_min", "x_max", "y_max"]].to_numpy(float))
    t["pred"] = pred
    ok = (t["pred"] == t["class_id"]).astype(int)
    pos = (t[f"position_only_pred_seed{a.split}"] == t["class_id"]).astype(int)
    w = pd.DataFrame(dict(image_id=t["image_id"], n=1, ok=ok, pos=pos, missed=t["pred"].isna().astype(int),
                          gap_n=t["next_to_gap"].astype(int), gap_ok=ok * t["next_to_gap"],
                          j_n=t["joint_failure"].astype(int), j_ok=ok * t["joint_failure"]))
    jw = t["joint_failure"] & (ok == 0) & t["pred"].notna()
    w["j_wrong"] = jw.astype(int)
    w["j_rep"] = (jw & (t["pred"] == t["shared_wrong_class"])).astype(int)
    s = w.groupby("image_id").sum()
    s["kept"] = kept  # detections at conf >= 0.5
    s["fp"] = s["kept"] - s["ok"]  # unmatched plus misnumbered
    c = {k: i for i, k in enumerate(s.columns)}
    r = lambda num, den: (lambda u: 100 * u[c[num]] / u[c[den]])
    stats = {
        "top1": r("ok", "n"),
        "missed": r("missed", "n"),
        "misnumbered": lambda u: 100 * (u[c["n"]] - u[c["ok"]] - u[c["missed"]]) / u[c["n"]],
        "position_only_top1": r("pos", "n"),
        "gap_pp": lambda u: 100 * (u[c["ok"]] - u[c["pos"]]) / u[c["n"]],
        "top1_next_to_gap": r("gap_ok", "gap_n"),
        "top1_joint": r("j_ok", "j_n"),
        "repeats_shared": r("j_rep", "j_wrong"),
        "false_pos_per_xray": lambda u: u[c["fp"]] / len(s),
        "f1_numbered": lambda u: 100 * 2 * u[c["ok"]] / (2 * u[c["ok"]] + u[c["fp"]] + u[c["n"]] - u[c["ok"]]),
    }
    rng = np.random.default_rng(0)
    rows = [dict(metric=k, value=v, ci_lo=lo, ci_hi=hi) for k, f in stats.items() for v, lo, hi in [boot(s, f, rng)]]
    rows += [dict(metric="top1_macro", value=100 * ok.groupby(t["class_id"]).mean().mean()),
             dict(metric="n_teeth", value=len(t)), dict(metric="n_joint", value=int(t["joint_failure"].sum())),
             dict(metric="n_next_to_gap", value=int(t["next_to_gap"].sum()))]
    out = pd.DataFrame(rows)
    if a.split == 1:
        print("note: the joint-failure set comes from split 0 reference detectors")
    print(out.round(2).to_string(index=False))
    if a.out:
        out.to_csv(a.out, index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
