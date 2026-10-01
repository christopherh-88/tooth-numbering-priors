"""Context masking (cpu_repro/cv/PHASE3_BATCH2_RULES.md item 2, frozen 2026-10-01).

For each sampled tooth and k in K: black out every pixel outside the
tooth's labeled box widened by k box widths on each side and k box heights
above and below, run the detector from the fold that tests the X-ray, and
match that tooth alone (confidence-first, as shift_test.match). Runs in a
Kaggle GPU kernel (see kaggle/make_phase3_kernels.py):

    python context_test.py --repo <repo checkout> --models <dir with *_cvseed0/> \
        --per-tooth <confidence-first per_tooth_predictions.csv> --out <dir>

Writes context_per_tooth.csv (one row per tooth, detector and k) and
context_summary.csv (control teeth: top-1 change from the full image with
an X-ray bootstrap CI and the rule; joint failures: shares still given the
shared wrong answer, correct, and missed).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import shift_test as st

DETS = ["yolov8x", "rtdetr_l", "fasterrcnn"]
K = [0.5, 1.0, 2.0]
N_CONTROL = 1000
N_BOOT = 10_000


def sample(t, rng):
    correct = np.logical_and.reduce([t[f"{d}_pred"] == t["class_id"] for d in DETS])
    joint = t["coord_pred"] != t["class_id"]
    for d in DETS:
        joint &= (t[f"{d}_pred"] != t["class_id"]) & t[f"{d}_pred"].notna()
    p = np.stack([t[f"{d}_pred"].to_numpy() for d in DETS])
    joint &= (p == p[0]).all(0)
    J = t[joint].assign(group="joint", shared_pred=p[0][joint])
    C = t[correct].sample(N_CONTROL, random_state=rng.integers(2**31)).assign(group="control", shared_pred=np.nan)
    return pd.concat([J, C]).reset_index(drop=True)


def mask_around(img, r, k):
    h, w = img.shape[:2]
    x0 = int(np.floor((r.x_center - r.width / 2 - k * r.width) * w))
    x1 = int(np.ceil((r.x_center + r.width / 2 + k * r.width) * w))
    y0 = int(np.floor((r.y_center - r.height / 2 - k * r.height) * h))
    y1 = int(np.ceil((r.y_center + r.height / 2 + k * r.height) * h))
    out = np.zeros_like(img)
    ys, xs = slice(max(y0, 0), min(y1, h)), slice(max(x0, 0), min(x1, w))
    out[ys, xs] = img[ys, xs]
    return out


def boot_change(df, rng):
    """Top-1 change (masked minus full, pp) and 95% CI, resampling X-rays."""
    s = df.assign(n=1).groupby("image_id")[["n", "ok_full", "ok"]].sum().to_numpy(float)
    stat = lambda u: 100 * (u[2] - u[1]) / u[0]
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    return stat(s.sum(0)), *np.percentile(stat(s[idx].sum(1).T), [2.5, 97.5])


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--per-tooth", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    st.MATCH_ORDER = "conf"
    rng = np.random.default_rng(0)

    s = sample(pd.read_csv(a.per_tooth), rng)
    print(s["group"].value_counts().to_dict(), flush=True)
    img_dir = a.repo / "Dataset/bb_u_net_dataset/panoramic_x_rays"

    rows = []
    for det in DETS:
        run = next(a.models.rglob(f"{det}_cvseed0"))
        for f in range(5):
            model = st.Detector(det, run / f"fold{f}")
            for image_id, g in s[s["fold"] == f].groupby("image_id"):
                img = cv2.imread(str(img_dir / f"{image_id}.jpg"))
                for r in g.itertuples():
                    gt = np.array([[r.x_center - r.width / 2, r.y_center - r.height / 2,
                                    r.x_center + r.width / 2, r.y_center + r.height / 2]])
                    for k in K:
                        pred = st.match(gt, *model(mask_around(img, r, k)))[0]
                        rows.append(dict(detector=det, k=k, group=r.group, image_id=image_id, fdi=r.fdi,
                                         class_id=r.class_id, shared_pred=r.shared_pred,
                                         full_pred=getattr(r, f"{det}_pred"), pred=pred))
            print(det, "fold", f, "done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(a.out / "context_per_tooth.csv", index=False)

    summary = []
    for (det, k), d in t.groupby(["detector", "k"]):
        c = d[d["group"] == "control"].assign(ok_full=lambda x: (x.full_pred == x.class_id).astype(int),
                                               ok=lambda x: (x.pred == x.class_id).astype(int))
        change, lo, hi = boot_change(c, rng)
        rule = 1 if change <= -20 else (2 if change > -5 else 3)
        j = d[d["group"] == "joint"]
        summary.append(dict(
            detector=det, k=k, n_control=len(c), control_top1_masked=100 * c["ok"].mean(),
            control_change_pp=change, control_change_lo=lo, control_change_hi=hi,
            control_missed_pct=100 * c["pred"].isna().mean(), rule_at_k1=rule if k == 1.0 else np.nan,
            n_joint=len(j), joint_still_shared_wrong_pct=100 * (j["pred"] == j["shared_pred"]).mean(),
            joint_now_correct_pct=100 * (j["pred"] == j["class_id"]).mean(),
            joint_other_wrong_pct=100 * (j["pred"].notna() & (j["pred"] != j["shared_pred"])
                                         & (j["pred"] != j["class_id"])).mean(),
            joint_missed_pct=100 * j["pred"].isna().mean()))
    out = pd.DataFrame(summary)
    out.to_csv(a.out / "context_summary.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
