"""Label-error check on the CV joint failures (cpu_repro/cv/LABEL_CHECK_RULES.md, frozen 2026-10-01).

    python cpu_repro/cv/label_check.py --per-tooth <per_tooth_predictions.csv> --out <dir>
        [--detectors yolov8x rtdetr_l fasterrcnn] [--roboflow Dataset/yolo_train_dataset]

Check A: for every labeled tooth, the FDI label the Roboflow box file gives
the same tooth, mapped back to the uncropped frame. Compares r_J (joint
failures where the Roboflow label equals the detectors' shared answer) with
r_C (control teeth where the Roboflow label differs from the mask label).
Check B (secondary): cleanlab on the position-only model's out-of-fold
probabilities, if cleanlab is installed.

Outputs: label_check_teeth.csv (one row per joint failure, with the
Roboflow label and flags) and label_check_summary.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES = ["x_center", "y_center", "width", "height", "area", "aspect_ratio"]
MATCH_IOU = 0.5
MAX_FIT_RESIDUAL = 0.01
OUTLIER = 0.02


def xyxy(df):
    x, y, w, h = (df[c].to_numpy() for c in ("x_center", "y_center", "width", "height"))
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)


def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def load_roboflow(root: Path):
    rows = []
    for p in sorted(root.glob("*/labels/*.txt")):
        image_id = p.stem.split("_jpg")[0]
        for line in p.read_text().split("\n"):
            if line.strip():
                c, x, y, w, h = line.split()[:5]
                rows.append(dict(image_id=image_id, copy=p.stem, class_id=int(c), x_center=float(x),
                                 y_center=float(y), width=float(w), height=float(h)))
    rf = pd.DataFrame(rows)
    rf["n_copies"] = rf.groupby("image_id")["copy"].transform("nunique")
    return rf


def fit_axis_map(mask, copy):
    """x_mask = a*x_rf + b and y_mask = c*y_rf + d (a crop plus a stretch),
    fitted on centers of teeth with the same label in both files, refitted
    once without pairs off by more than OUTLIER. Returns (params, median residual)."""
    pair = mask.merge(copy, on="class_id", suffixes=("_m", "_r"))
    if len(pair) < 4:
        return None, np.inf
    keep = np.ones(len(pair), bool)
    for _ in range(2):
        A = np.polyfit(pair["x_center_r"][keep], pair["x_center_m"][keep], 1)
        C = np.polyfit(pair["y_center_r"][keep], pair["y_center_m"][keep], 1)
        res = np.hypot(np.polyval(A, pair["x_center_r"]) - pair["x_center_m"],
                       np.polyval(C, pair["y_center_r"]) - pair["y_center_m"])
        keep = res <= OUTLIER
        if keep.sum() < 4:
            return None, np.inf
    return (A, C), float(np.median(res[keep]))


def roboflow_labels(teeth, rf):
    """Per labeled tooth: the Roboflow class of the best-overlapping box after
    mapping to the uncropped frame. Uses the single uncropped copy when one
    exists, else every copy that maps cleanly, and keeps the label only if
    those copies agree."""
    out = pd.Series(np.nan, index=teeth.index)
    status = {}
    for image_id, mask in teeth.groupby("image_id"):
        copies = rf[rf["image_id"] == image_id]
        if copies.empty:
            status[image_id] = "no_roboflow"
            continue
        votes = []
        for _, cp in copies.groupby("copy"):
            if cp["n_copies"].iloc[0] == 1:
                mapped = cp
            else:
                params, med = fit_axis_map(mask, cp)
                if params is None or med > MAX_FIT_RESIDUAL:
                    continue
                (a, b), (c, d) = params
                mapped = cp.assign(x_center=a * cp["x_center"] + b, y_center=c * cp["y_center"] + d,
                                   width=a * cp["width"], height=c * cp["height"])
            m = iou(xyxy(mask), xyxy(mapped))
            lab = np.full(len(mask), np.nan)
            j = m.argmax(1)
            ok = m[np.arange(len(mask)), j] >= MATCH_IOU
            lab[ok] = mapped["class_id"].to_numpy()[j[ok]]
            votes.append(lab)
        if not votes:
            status[image_id] = "no_clean_mapping"
            continue
        v = np.stack(votes)
        agree = np.array([len(set(col[~np.isnan(col)])) == 1 for col in v.T])
        first = np.array([col[~np.isnan(col)][0] if agree[i] else np.nan for i, col in enumerate(v.T)])
        out.loc[mask.index] = first
        status[image_id] = f"ok_{len(votes)}_copies"
    return out, status


def position_probs(teeth):
    t = teeth.assign(area=teeth["width"] * teeth["height"],
                     aspect_ratio=teeth["width"] / teeth["height"])
    probs = np.zeros((len(t), 32))
    for f in sorted(t["fold"].unique()):
        tr, te = t["fold"] != f, t["fold"] == f
        clf = HistGradientBoostingClassifier(random_state=0).fit(t.loc[tr, FEATURES], t.loc[tr, "class_id"])
        probs[np.where(te)[0][:, None], clf.classes_[None, :]] = clf.predict_proba(t.loc[te, FEATURES])
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tooth", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--detectors", nargs="+", default=["yolov8x", "rtdetr_l", "fasterrcnn"])
    ap.add_argument("--roboflow", type=Path,
                    default=Path(__file__).resolve().parents[2] / "Dataset" / "yolo_train_dataset")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(a.per_tooth)
    t["coord_correct"] = t["coord_pred"] == t["class_id"]
    dets = a.detectors
    correct = {d: t[f"{d}_pred"] == t["class_id"] for d in dets}
    joint = ~t["coord_correct"]
    for d in dets:
        joint &= ~correct[d] & t[f"{d}_pred"].notna()
    preds = np.stack([t[f"{d}_pred"].to_numpy() for d in dets])
    shared = np.where((preds == preds[0]).all(0), preds[0], np.nan)
    control = np.logical_and.reduce([correct[d] for d in dets])

    t["roboflow_label"], status = roboflow_labels(t, load_roboflow(a.roboflow))
    t["shared_pred"] = shared
    J = t[joint & ~np.isnan(shared)]
    C = t[control]
    jj, cc = J[J["roboflow_label"].notna()], C[C["roboflow_label"].notna()]
    r_J = float((jj["roboflow_label"] == jj["shared_pred"]).mean())
    r_C = float((cc["roboflow_label"] != cc["class_id"]).mean())
    rule = 1 if r_J >= 0.5 else (2 if r_J <= 2 * r_C else 3)

    summary = dict(detectors=" ".join(dets), n_teeth=len(t), n_joint_same_wrong=len(J),
                   n_joint_with_roboflow_label=len(jj), r_J_pct=100 * r_J,
                   n_J_roboflow_equals_mask=int((jj["roboflow_label"] == jj["class_id"]).sum()),
                   n_J_roboflow_other=int(((jj["roboflow_label"] != jj["class_id"])
                                           & (jj["roboflow_label"] != jj["shared_pred"])).sum()),
                   n_control=len(C), n_control_with_roboflow_label=len(cc), r_C_pct=100 * r_C,
                   rule=rule,
                   xrays_ok=sum(s.startswith("ok") for s in status.values()),
                   xrays_skipped=sum(not s.startswith("ok") for s in status.values()))

    # Joint-failure statistics recomputed without teeth whose Roboflow label sides with the models.
    rest = J[~(J["roboflow_label"] == J["shared_pred"])]
    summary.update(n_joint_after_excluding=len(rest),
                   position_match_pct_after_excluding=100 * float((rest["shared_pred"] == rest["coord_pred"]).mean())
                   if len(rest) else np.nan)

    try:
        from cleanlab.filter import find_label_issues
        probs = position_probs(t)
        issues = find_label_issues(t["class_id"].to_numpy(), probs)
        t["cleanlab_flag"] = issues
        summary.update(cleanlab_flag_pct_J=100 * float(issues[J.index].mean()),
                       cleanlab_flag_pct_C=100 * float(issues[C.index].mean()))
    except ImportError:
        t["cleanlab_flag"] = np.nan
        print("cleanlab not installed; check B skipped")

    cols = ["image_id", "category", "fold", "fdi", "class_id", "shared_pred", "coord_pred",
            "roboflow_label", "cleanlab_flag", "x_center", "y_center", "width", "height"]
    t.loc[J.index, cols].to_csv(a.out / "label_check_teeth.csv", index=False, float_format="%.6f")
    pd.DataFrame([summary]).to_csv(a.out / "label_check_summary.csv", index=False, float_format="%.4f")
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
