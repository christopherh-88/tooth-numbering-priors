"""Cross-architecture error agreement (RESULTS.md Section 46).

Do YOLOv8x, Faster R-CNN, and RT-DETR-l converge on the *same* wrong FDI
class when they all fail on the same instance, and does that shared wrong
answer match what the coordinate-only (position-only) baseline predicts?

Joins each architecture's per-seed per-instance prediction table
(`cpu_repro/yolo_training/eval_results/{multiseed,fasterrcnn_multiseed,
rtdetr_multiseed}/joined_seed{N}.csv`, produced by Sections 33/40/45's own
evaluation runs) on (label_file, line_idx), restricts to instances where a
given pair/triple of architectures AND the coordinate-only baseline are all
simultaneously wrong (excluding null predictions, i.e. missed detections),
and measures how often they predict the identical wrong class. Significance
is assessed by permutation: shuffle one architecture's predicted-wrong-class
labels among the joint-wrong instance set (preserves each architecture's own
marginal wrong-class frequency, destroys instance-level pairing).

Run: python cpu_repro/yolo_training/cross_architecture_agreement.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "cpu_repro" / "yolo_training" / "eval_results"
SEEDS = [0, 1, 2, 3, 4]
N_PERM = 10_000
RNG_SEED = 0

sys.path.insert(0, str(REPO_ROOT / "cpu_repro" / "coord_baseline"))
from build_coord_baseline import FDI_CODES, is_neighbor_error  # noqa: E402

ARCHS = {
    "yolo": ("multiseed", "yolo_pred", "yolo_correct"),
    "fasterrcnn": ("fasterrcnn_multiseed", "fasterrcnn_pred", "fasterrcnn_correct"),
    "rtdetr": ("rtdetr_multiseed", "rtdetr_pred", "rtdetr_correct"),
}
DISPLAY = {"yolo": "YOLOv8x", "fasterrcnn": "Faster R-CNN", "rtdetr": "RT-DETR-l"}


def load_seed(seed: int) -> pd.DataFrame:
    tables = {}
    for arch, (subdir, pred_col, correct_col) in ARCHS.items():
        path = EVAL_ROOT / subdir / f"joined_seed{seed}.csv"
        df = pd.read_csv(path)
        df = df[["label_file", "line_idx", "true_class", "coord_pred", "coord_correct", pred_col, correct_col]]
        df = df.rename(columns={pred_col: f"{arch}_pred", correct_col: f"{arch}_correct"})
        tables[arch] = df

    merged = tables["yolo"]
    for arch in ("fasterrcnn", "rtdetr"):
        merged = merged.merge(
            tables[arch],
            on=["label_file", "line_idx"],
            suffixes=("", f"_{arch}dup"),
        )

    # sanity check: true_class/coord_pred/coord_correct must agree across all three source tables
    for arch in ("fasterrcnn", "rtdetr"):
        for col in ("true_class", "coord_pred", "coord_correct"):
            dup_col = f"{col}_{arch}dup"
            if not (merged[col] == merged[dup_col]).all():
                raise AssertionError(f"seed {seed}: {col} mismatch between yolo table and {arch} table")
            merged = merged.drop(columns=[dup_col])

    return merged


def permutation_p(n_joint: int, observed_agree: int, wrong_labels_a: np.ndarray, wrong_labels_b: np.ndarray, rng: np.random.RandomState):
    null_agree = np.empty(N_PERM)
    for i in range(N_PERM):
        shuffled = rng.permutation(wrong_labels_b)
        null_agree[i] = np.mean(wrong_labels_a == shuffled)
    return float(null_agree.mean()), float(np.mean(null_agree >= observed_agree / n_joint))


def pairwise_agreement(df: pd.DataFrame, arch_a: str, arch_b: str, rng: np.random.RandomState):
    both_wrong = df[
        (~df[f"{arch_a}_correct"].astype(bool))
        & (~df[f"{arch_b}_correct"].astype(bool))
        & (~df["coord_correct"].astype(bool))
        & df[f"{arch_a}_pred"].notna()
        & df[f"{arch_b}_pred"].notna()
    ].copy()
    n = len(both_wrong)
    a_pred = both_wrong[f"{arch_a}_pred"].astype(int).to_numpy()
    b_pred = both_wrong[f"{arch_b}_pred"].astype(int).to_numpy()
    same = a_pred == b_pred
    n_same = int(same.sum())
    null_mean, p = permutation_p(n, n_same, a_pred, b_pred, rng)

    adjacent = np.array([
        is_neighbor_error(t, p_) for t, p_ in zip(
            both_wrong.loc[same, "true_class"].astype(int).to_numpy(), a_pred[same]
        )
    ])
    return {
        "n_joint": n,
        "n_same": n_same,
        "rate": n_same / n if n else float("nan"),
        "null_mean": null_mean,
        "p": p,
        "n_adjacent": int(adjacent.sum()),
        "same_class_true": both_wrong.loc[same, "true_class"].astype(int).to_numpy(),
        "same_class_pred": a_pred[same],
    }


def triple_agreement(df: pd.DataFrame, rng: np.random.RandomState):
    all_wrong = df[
        (~df["yolo_correct"].astype(bool))
        & (~df["fasterrcnn_correct"].astype(bool))
        & (~df["rtdetr_correct"].astype(bool))
        & (~df["coord_correct"].astype(bool))
        & df["yolo_pred"].notna()
        & df["fasterrcnn_pred"].notna()
        & df["rtdetr_pred"].notna()
    ].copy()
    n = len(all_wrong)
    y = all_wrong["yolo_pred"].astype(int).to_numpy()
    f = all_wrong["fasterrcnn_pred"].astype(int).to_numpy()
    r = all_wrong["rtdetr_pred"].astype(int).to_numpy()
    same = (y == f) & (f == r)
    n_same = int(same.sum())

    coord_pred = all_wrong.loc[same, "coord_pred"].astype(int).to_numpy()
    shared_wrong = y[same]
    n_coord_match = int(np.sum(shared_wrong == coord_pred))
    null_mean, p_coord = permutation_p(n_same, n_coord_match, shared_wrong, coord_pred, rng)

    return {
        "n_joint": n,
        "n_triple_same": n_same,
        "rate": n_same / n if n else float("nan"),
        "n_coord_match": n_coord_match,
        "coord_match_rate": n_coord_match / n_same if n_same else float("nan"),
        "coord_null_mean": null_mean,
        "coord_p": p_coord,
    }


def main():
    per_seed_rows = []
    for seed in SEEDS:
        df = load_seed(seed)
        rng = np.random.RandomState(RNG_SEED)

        pairs = [("yolo", "fasterrcnn"), ("yolo", "rtdetr"), ("fasterrcnn", "rtdetr")]
        pair_results = {p: pairwise_agreement(df, p[0], p[1], rng) for p in pairs}
        triple = triple_agreement(df, rng)

        print(f"\n=== seed {seed} ===")
        for (a, b), res in pair_results.items():
            print(
                f"{DISPLAY[a]} vs. {DISPLAY[b]}: n={res['n_joint']} "
                f"same-wrong-class={res['n_same']}/{res['n_joint']} ({res['rate']:.1%}) "
                f"null_mean={res['null_mean']:.1%} p={res['p']:.4f} "
                f"adjacent={res['n_adjacent']}/{res['n_same']} ({res['n_adjacent']/res['n_same']:.1%})"
            )
        print(
            f"triple agreement: n={triple['n_joint']} "
            f"same={triple['n_triple_same']}/{triple['n_joint']} ({triple['rate']:.1%})"
        )
        print(
            f"coord-match (triple-agreement set): "
            f"{triple['n_coord_match']}/{triple['n_triple_same']} ({triple['coord_match_rate']:.1%}) "
            f"null_mean={triple['coord_null_mean']:.1%} p={triple['coord_p']:.4f}"
        )

        n_adj_pooled = sum(r["n_adjacent"] for r in pair_results.values())
        n_same_pooled = sum(r["n_same"] for r in pair_results.values())
        per_seed_rows.append({
            "seed": seed,
            "n_joint_triple": triple["n_joint"],
            "triple_rate": triple["rate"],
            "adjacent_share_pooled": n_adj_pooled / n_same_pooled if n_same_pooled else float("nan"),
            "n_triple_same": triple["n_triple_same"],
            "coord_match_rate": triple["coord_match_rate"],
        })

    summary = pd.DataFrame(per_seed_rows)
    print("\n=== 5-seed summary ===")
    print(summary.to_string(index=False))
    out_path = EVAL_ROOT / "cross_architecture_agreement_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
