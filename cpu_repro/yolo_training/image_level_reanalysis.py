"""Phase 2 batch 1 (PHASE2_BATCH1_RULES.md, frozen 2026-10-01).

A. Gap CIs resampled by source X-ray (all copies, and one copy per X-ray),
   plus the Section 31 augmentation comparison resampled by source X-ray.
B. Missed vs. misnumbered, and the gap on matched teeth only.
C. Geirhos error-consistency kappa (and kappa_max) next to phi, for each
   detector vs. the position-only model and for each detector pair.

Each seed's UFBA-425 test set holds 85 source X-rays but about 200 image
files (up to 3 augmented copies of each X-ray). The source X-ray is the
independent unit, so every CI here resamples source X-rays: draw 85 with
replacement, keep every tooth on every copy of each drawn X-ray.

Reads saved per-tooth files only and writes new files only:
  eval_results/phase2_gap_by_xray.csv
  eval_results/phase2_missed_vs_misnumbered.csv
  eval_results/phase2_error_consistency.csv
  eval_results/phase2_augmentation_by_xray.csv
  eval_results/phase2_originals_vs_cropped.csv (exploratory audit, not in
    the frozen rules; see originals_vs_cropped())
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL = HERE / "eval_results"
N_BOOT = 10_000
RNG_SEED = 0

DETECTORS = {"yolov8x": ("multiseed", "yolo"),
             "rtdetr_l": ("rtdetr_multiseed", "rtdetr"),
             "fasterrcnn": ("fasterrcnn_multiseed", "fasterrcnn")}
GROUPS = {"T4 seeds 0-4": [0, 1, 2, 3, 4],
          "M3 seeds 5-9": [5, 6, 7, 8, 9],
          "T4 seeds 11-14": [11, 12, 13, 14]}
SEED_GROUP = {s: g for g, ss in GROUPS.items() for s in ss}
KEY = ["label_file", "line_idx"]


def source_id(label_file):
    return label_file.split("_jpg")[0]


def load_seed(seed):
    """One row per labeled test tooth, all three detectors joined."""
    merged = None
    for det, (folder, prefix) in DETECTORS.items():
        d = pd.read_csv(EVAL / folder / f"joined_seed{seed}.csv")
        d = d.rename(columns={f"{prefix}_pred": f"{det}_pred",
                              f"{prefix}_correct": f"{det}_correct"})
        cols = KEY + [f"{det}_pred", f"{det}_correct"]
        if merged is None:
            merged = d[["image_id"] + KEY + ["true_class", "coord_pred", "coord_correct"]
                       + cols[2:]]
        else:
            # Same test teeth for every detector, and the same position-only
            # predictions, or the comparison is not paired.
            assert len(d) == len(merged), (seed, det, len(d), len(merged))
            check = merged[KEY + ["coord_pred"]].merge(d[KEY + ["coord_pred"]], on=KEY,
                                                        suffixes=("", "_d"))
            assert len(check) == len(merged), (seed, det, "key mismatch")
            assert (check["coord_pred"] == check["coord_pred_d"]).all(), (seed, det)
            merged = merged.merge(d[cols], on=KEY, how="inner", validate="1:1")
    assert (merged["label_file"].map(source_id) == merged["image_id"]).all(), seed
    for det in DETECTORS:
        missed = merged[f"{det}_pred"].isna()
        assert not merged.loc[missed, f"{det}_correct"].any(), (seed, det)
        merged[f"{det}_missed"] = missed
    return merged


def one_copy(df):
    first = df.groupby("image_id")["label_file"].min()
    return df[df["label_file"].isin(set(first))]


def cluster_sums(df, cols):
    """Per-X-ray sums of 0/1 columns, as an (n_xrays, n_cols) array."""
    g = df.assign(n=1).groupby("image_id")[["n"] + cols].sum()
    return g.to_numpy(dtype=float)


def bootstrap(sums, stat, rng):
    """Point estimate and 95% percentile CI, resampling X-rays (rows)."""
    point = stat(sums.sum(axis=0))
    idx = rng.integers(0, len(sums), size=(N_BOOT, len(sums)))
    boots = sums[idx].sum(axis=1)  # (N_BOOT, n_cols)
    vals = stat(boots.T)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi


def consistency_stats(t):
    """t = totals [n, a_correct, b_correct, both_correct, both_wrong]."""
    n, a, b, both_c, both_w = t
    p1, p2 = a / n, b / n
    c_obs = (both_c + both_w) / n
    c_exp = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (c_obs - c_exp) / (1 - c_exp)
    kappa_max = ((1 - np.abs(p1 - p2)) - c_exp) / (1 - c_exp)
    n11, n00 = both_c, both_w
    n10, n01 = a - both_c, b - both_c
    phi = (n11 * n00 - n10 * n01) / np.sqrt(a * (n - a) * b * (n - b))
    return kappa, kappa_max, phi


def add_pair_cols(df, a_col, b_col):
    a = df[a_col].astype(int)
    b = df[b_col].astype(int)
    return df.assign(_a=a, _b=b, _bc=a & b, _bw=(1 - a) & (1 - b))


def main():
    rng = np.random.default_rng(RNG_SEED)
    gap_rows, mm_rows, ec_rows = [], [], []
    seeds = sorted(SEED_GROUP)
    summary_check = pd.read_csv(EVAL / "multiseed" / "multiseed_summary.csv").set_index("seed")

    for seed in seeds:
        full = load_seed(seed)
        group = SEED_GROUP[seed]
        n_xrays = full["image_id"].nunique()
        assert n_xrays == 85, (seed, n_xrays)

        for variant, df in [("all_copies", full), ("one_copy", one_copy(full))]:
            for det in DETECTORS:
                # A. paired gap, all labeled teeth, misses count as wrong.
                d = df.assign(_det=df[f"{det}_correct"].astype(int),
                              _crd=df["coord_correct"].astype(int))
                s = cluster_sums(d, ["_det", "_crd"])
                gap = lambda t: 100 * (t[1] - t[2]) / t[0]
                p, lo, hi = bootstrap(s, gap, rng)
                gap_rows.append(dict(seed=seed, group=group, variant=variant, detector=det,
                                     n_teeth=len(df), n_files=df["label_file"].nunique(),
                                     n_xrays=df["image_id"].nunique(),
                                     det_top1=100 * d["_det"].mean(),
                                     coord_top1=100 * d["_crd"].mean(),
                                     gap_pp=p, ci_lo=lo, ci_hi=hi))
                if seed in summary_check.index and variant == "all_copies" and det == "yolov8x":
                    ref = summary_check.loc[seed, "gap_pp"]
                    assert abs(p - ref) < 1e-9, (seed, p, ref)
                    assert abs(len(df) - summary_check.loc[seed, "n"]) == 0

                if variant != "all_copies":
                    continue

                # B. correct / misnumbered / missed, and the matched-only gap.
                m = ~df[f"{det}_missed"]
                d = df.assign(_m=m.astype(int),
                              _det=df[f"{det}_correct"].astype(int),
                              _crd_m=(df["coord_correct"] & m).astype(int),
                              _crd=df["coord_correct"].astype(int))
                s = cluster_sums(d, ["_m", "_det", "_crd_m", "_crd"])
                stats = {
                    "missed_pct": lambda t: 100 * (t[0] - t[1]) / t[0],
                    "misnumbered_pct": lambda t: 100 * (t[1] - t[2]) / t[0],
                    "correct_pct": lambda t: 100 * t[2] / t[0],
                    "matched_only_top1": lambda t: 100 * t[2] / t[1],
                    "coord_on_matched_top1": lambda t: 100 * t[3] / t[1],
                    "gap_all_gt_pp": lambda t: 100 * (t[2] - t[4]) / t[0],
                    "gap_matched_only_pp": lambda t: 100 * (t[2] / t[1] - t[3] / t[1]),
                    "matched_minus_all_gt_pp": lambda t: 100 * ((t[2] - t[3]) / t[1]
                                                               - (t[2] - t[4]) / t[0]),
                }
                row = dict(seed=seed, group=group, detector=det, n_teeth=len(df),
                           n_missed=int((~m).sum()),
                           n_misnumbered=int((m & ~df[f"{det}_correct"]).sum()))
                for name, fn in stats.items():
                    p, lo, hi = bootstrap(s, fn, rng)
                    row[name], row[name + "_lo"], row[name + "_hi"] = p, lo, hi
                mm_rows.append(row)

                # C. error consistency, detector vs. position-only.
                d = add_pair_cols(df, f"{det}_correct", "coord_correct")
                s = cluster_sums(d, ["_a", "_b", "_bc", "_bw"])
                ec_rows.append(consistency_row(seed, group, det, "position_only", s, rng))

            if variant != "all_copies":
                continue
            # C. error consistency between detector pairs (descriptive).
            dets = list(DETECTORS)
            for i in range(len(dets)):
                for j in range(i + 1, len(dets)):
                    d = add_pair_cols(df, f"{dets[i]}_correct", f"{dets[j]}_correct")
                    s = cluster_sums(d, ["_a", "_b", "_bc", "_bw"])
                    ec_rows.append(consistency_row(seed, group, dets[i], dets[j], s, rng))
        print(f"seed {seed:2d} done ({group})")

    pd.DataFrame(gap_rows).to_csv(EVAL / "phase2_gap_by_xray.csv", index=False)
    pd.DataFrame(mm_rows).to_csv(EVAL / "phase2_missed_vs_misnumbered.csv", index=False)
    pd.DataFrame(ec_rows).to_csv(EVAL / "phase2_error_consistency.csv", index=False)
    augmentation(rng)
    originals_vs_cropped()
    print("wrote phase2_*.csv")


def consistency_row(seed, group, a, b, s, rng):
    row = dict(seed=seed, group=group, model_a=a, model_b=b)
    for k, name in enumerate(["kappa", "kappa_max", "phi"]):
        p, lo, hi = bootstrap(s, lambda t, k=k: consistency_stats(t)[k], rng)
        row[name], row[name + "_lo"], row[name + "_hi"] = p, lo, hi
    return row


def augmentation(rng):
    """Section 31 rerun: zero-jitter minus jittered YOLOv8x, seed 0."""
    m = pd.read_csv(EVAL / "mitigation_paired_joined.csv")
    m["source_id"] = m["label_file"].map(source_id)
    m["jit_correct"] = m["jit_correct"].astype(bool)
    m["zero_correct"] = m["zero_correct"].astype(bool)

    # Reproduce the saved CI exactly (file-level, RandomState(0), 2000 draws),
    # so the comparison below changes only the resampling unit.
    ids = m["image_id"].dropna().unique()
    per = {i: sub for i, sub in m.groupby("image_id")}
    old_rng = np.random.RandomState(0)
    vals = []
    for _ in range(2000):
        rows = pd.concat([per[i] for i in old_rng.choice(ids, size=len(ids), replace=True)],
                         ignore_index=True)
        vals.append(rows["zero_correct"].mean() - rows["jit_correct"].mean())
    old_lo, old_hi = np.percentile(vals, [2.5, 97.5])
    saved = pd.read_csv(EVAL / "mitigation_bootstrap_ci.csv").set_index("metric")
    assert abs(old_lo - saved.loc["paired_delta_top1", "ci_lo"]) < 1e-12, old_lo
    assert abs(old_hi - saved.loc["paired_delta_top1", "ci_hi"]) < 1e-12, old_hi

    out = []
    delta = lambda t: 100 * (t[2] - t[1]) / t[0]
    for unit, col in [("image_file (saved, n_boot=2000)", None),
                      ("source_xray, all copies", "source_id"),
                      ("source_xray, one copy", "source_id")]:
        if col is None:
            point = 100 * (m["zero_correct"].mean() - m["jit_correct"].mean())
            out.append(dict(resample_unit=unit, n_units=len(ids), n_teeth=len(m),
                            delta_pp=point, ci_lo=100 * old_lo, ci_hi=100 * old_hi))
            continue
        df = m
        if "one copy" in unit:
            first = m.groupby("source_id")["label_file"].min()
            df = m[m["label_file"].isin(set(first))]
        d = df.assign(_j=df["jit_correct"].astype(int), _z=df["zero_correct"].astype(int))
        s = d.assign(n=1).groupby(col)[["n", "_j", "_z"]].sum().to_numpy(dtype=float)
        p, lo, hi = bootstrap(s, delta, rng)
        out.append(dict(resample_unit=unit, n_units=len(s), n_teeth=len(df),
                        delta_pp=p, ci_lo=lo, ci_hi=hi))
    pd.DataFrame(out).to_csv(EVAL / "phase2_augmentation_by_xray.csv", index=False)


def originals_vs_cropped():
    """Exploratory audit, added after the frozen analyses were run.

    The Roboflow export made 3 copies of each source X-ray in its own train
    split by randomly cropping 0-20% of the image (README.roboflow.txt), and
    kept its valid/test X-rays as single, uncropped originals. Cropping moves
    every tooth in the frame, which weakens the position cue. This splits
    each seed's test X-rays by kind and computes the gap on each. Uses its
    own random generator so the frozen outputs above are unchanged.
    """
    rng = np.random.default_rng(1)
    rows = []
    for seed in sorted(SEED_GROUP):
        d = load_seed(seed)
        n_copies = d.groupby("image_id")["label_file"].transform("nunique")
        assert set(n_copies) <= {1, 3}, (seed, set(n_copies))
        for kind, sub in [("original", d[n_copies == 1]), ("cropped_copies", d[n_copies == 3])]:
            for det in DETECTORS:
                x = sub.assign(_det=sub[f"{det}_correct"].astype(int),
                               _crd=sub["coord_correct"].astype(int))
                s = cluster_sums(x, ["_det", "_crd"])
                p, lo, hi = bootstrap(s, lambda t: 100 * (t[1] - t[2]) / t[0], rng)
                rows.append(dict(seed=seed, group=SEED_GROUP[seed], kind=kind, detector=det,
                                 n_xrays=sub["image_id"].nunique(), n_teeth=len(sub),
                                 det_top1=100 * x["_det"].mean(),
                                 coord_top1=100 * x["_crd"].mean(),
                                 gap_pp=p, ci_lo=lo, ci_hi=hi))
    pd.DataFrame(rows).to_csv(EVAL / "phase2_originals_vs_cropped.csv", index=False)


if __name__ == "__main__":
    main()
