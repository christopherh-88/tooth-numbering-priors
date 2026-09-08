"""Coordinate-only baseline: can FDI tooth class be predicted from bounding-box
geometry alone, with zero image data?

If accuracy is high, position (where the box sits on the jaw, its size/shape)
is doing most of the classification work that we'd otherwise credit to the
image features - i.e. the "hard" part of tooth numbering may just be spatial
reasoning that a coordinate-only model already captures.

Data: Dataset/yolo_train_dataset/{train,valid,test}/labels/*.txt (YOLO format:
"class_id x_center y_center width height", normalized 0-1). class_id indexes
into the FDI code list in Dataset/yolo_train_dataset/data.yaml.

IMPORTANT leakage note: Roboflow generated ~3 augmented crops per source
X-ray (e.g. "cate1-00002_jpg.rf.<hash-a>.txt", "...<hash-b>.txt", ...,
all derived from the same original "cate1-00002.jpg"). Splitting by label
*file* would put near-duplicate augmentations of the same X-ray in both
train and test. We split by the ORIGINAL image id (the part of the filename
before "_jpg.rf.<hash>"), not by label file, so every augmented copy of a
given X-ray stays entirely on one side of the split. We also ignore the
repo's existing train/valid/test folder boundaries and pool everything,
then do our own grouped random split per seed, since the question here is
about the coordinate signal in general, not about matching the YOLO
detector's original split.
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
TEST_FRACTION = 0.2
CONFUSION_MATRIX_SEED = SEEDS[0]  # which seed's run to save a confusion matrix for

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLO_ROOT = REPO_ROOT / "Dataset" / "yolo_train_dataset"
OUTPUT_DIR = Path(__file__).resolve().parent
# --------------------------------------------------------------------------

FDI_CODES = [
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
]
CODE_TO_IDX = {c: i for i, c in enumerate(FDI_CODES)}

AUG_SUFFIX_RE = re.compile(r"_jpg\.rf\.[0-9a-f]+$")


def base_image_id(label_stem: str) -> str:
    """Strip Roboflow's '_jpg.rf.<hash>' augmentation suffix so augmented
    copies of the same source X-ray share one id."""
    return AUG_SUFFIX_RE.sub("", label_stem)


def load_instances() -> pd.DataFrame:
    rows = []
    label_files = sorted(YOLO_ROOT.glob("*/labels/*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files found under {YOLO_ROOT}")

    for lf in label_files:
        img_id = base_image_id(lf.stem)
        for line in lf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            x, y, w, h = (float(v) for v in parts[1:5])
            rows.append(
                {
                    "image_id": img_id,
                    "label_file": lf.name,
                    "class_id": class_id,
                    "x_center": x,
                    "y_center": y,
                    "width": w,
                    "height": h,
                }
            )

    df = pd.DataFrame(rows)
    df["area"] = df["width"] * df["height"]
    df["aspect_ratio"] = df["width"] / df["height"].replace(0, np.nan)
    df["aspect_ratio"] = df["aspect_ratio"].fillna(df["aspect_ratio"].median())
    return df


FEATURE_COLS = ["x_center", "y_center", "width", "height", "area", "aspect_ratio"]


def quadrant_of(class_id: np.ndarray) -> np.ndarray:
    codes = np.array(FDI_CODES)[class_id]
    return np.array([c[0] for c in codes])


def tooth_type_of(class_id: np.ndarray) -> np.ndarray:
    codes = np.array(FDI_CODES)[class_id]
    return np.array([c[1] for c in codes])


MIRROR_QUADRANTS = {("1", "2"), ("2", "1"), ("3", "4"), ("4", "3")}


def is_mirror_quadrant_error(true_id: int, pred_id: int) -> bool:
    tc, pc = FDI_CODES[true_id], FDI_CODES[pred_id]
    return tc[1] == pc[1] and (tc[0], pc[0]) in MIRROR_QUADRANTS


def is_neighbor_error(true_id: int, pred_id: int) -> bool:
    tc, pc = FDI_CODES[true_id], FDI_CODES[pred_id]
    return tc[0] == pc[0] and abs(int(tc[1]) - int(pc[1])) == 1


def grouped_split(df: pd.DataFrame, seed: int, test_fraction: float):
    image_ids = np.array(df["image_id"].unique().tolist(), dtype=object)
    rng = np.random.RandomState(seed)
    shuffled = image_ids.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_fraction))
    test_ids = set(shuffled[:n_test])
    train_ids = set(shuffled[n_test:])
    train_df = df[df["image_id"].isin(train_ids)]
    test_df = df[df["image_id"].isin(test_ids)]
    return train_df, test_df


def evaluate(y_true, y_pred, train_y):
    top1 = float(np.mean(y_true == y_pred))
    quad_acc = float(np.mean(quadrant_of(y_true) == quadrant_of(y_pred)))
    type_acc = float(np.mean(tooth_type_of(y_true) == tooth_type_of(y_pred)))

    majority_class = np.bincount(train_y, minlength=32).argmax()
    majority_acc = float(np.mean(y_true == majority_class))

    wrong = y_true != y_pred
    n_wrong = int(wrong.sum())
    if n_wrong > 0:
        mirror_frac = float(
            np.mean([is_mirror_quadrant_error(t, p) for t, p in zip(y_true[wrong], y_pred[wrong])])
        )
        neighbor_frac = float(
            np.mean([is_neighbor_error(t, p) for t, p in zip(y_true[wrong], y_pred[wrong])])
        )
    else:
        mirror_frac = 0.0
        neighbor_frac = 0.0

    return {
        "top1_acc": top1,
        "quadrant_acc": quad_acc,
        "tooth_type_acc": type_acc,
        "majority_baseline_acc": majority_acc,
        "mirror_quadrant_error_frac": mirror_frac,
        "neighbor_error_frac": neighbor_frac,
        "n_test": len(y_true),
    }


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    if n < 2:
        return mean, 0.0
    sem = values.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean, t_crit * sem


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_instances()
    n_images = df["image_id"].nunique()
    print(f"Loaded {len(df)} tooth instances from {n_images} unique source X-rays "
          f"({df['label_file'].nunique()} label files, i.e. incl. augmented copies).")
    print(f"Class distribution: min={df['class_id'].value_counts().min()}, "
          f"max={df['class_id'].value_counts().max()}, "
          f"mean={df['class_id'].value_counts().mean():.1f}")
    print()

    classifiers = {
        "logistic_regression": lambda: LogisticRegression(max_iter=2000),
        "gradient_boosted_tree": lambda: HistGradientBoostingClassifier(random_state=0),
    }

    per_seed_results = {name: [] for name in classifiers}
    confusion_data = {}

    for seed in SEEDS:
        train_df, test_df = grouped_split(df, seed, TEST_FRACTION)
        x_train = train_df[FEATURE_COLS].to_numpy()
        y_train = train_df["class_id"].to_numpy()
        x_test = test_df[FEATURE_COLS].to_numpy()
        y_test = test_df["class_id"].to_numpy()

        scaler = StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        for name, make_clf in classifiers.items():
            clf = make_clf()
            if name == "logistic_regression":
                clf.fit(x_train_scaled, y_train)
                y_pred = clf.predict(x_test_scaled)
            else:
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)

            metrics = evaluate(y_test, y_pred, y_train)
            metrics["seed"] = seed
            per_seed_results[name].append(metrics)

            if seed == CONFUSION_MATRIX_SEED:
                confusion_data[name] = (y_test, y_pred)

        print(f"seed {seed}: train_images={train_df['image_id'].nunique()} "
              f"test_images={test_df['image_id'].nunique()} "
              f"train_instances={len(train_df)} test_instances={len(test_df)}")

    print()
    print("=" * 88)
    print("SUMMARY (mean +/- 95% CI over 5 seeds, image-level grouped split)")
    print("=" * 88)

    summary_rows = []
    metric_keys = [
        "top1_acc", "quadrant_acc", "tooth_type_acc",
        "majority_baseline_acc", "mirror_quadrant_error_frac", "neighbor_error_frac",
    ]
    for name, results in per_seed_results.items():
        print(f"\n{name}")
        row = {"classifier": name}
        for key in metric_keys:
            values = [r[key] for r in results]
            mean, ci = mean_ci95(values)
            print(f"  {key:28s} {mean:.4f} +/- {ci:.4f}")
            row[f"{key}_mean"] = mean
            row[f"{key}_ci95"] = ci
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    per_seed_rows = []
    for name, results in per_seed_results.items():
        for r in results:
            per_seed_rows.append({"classifier": name, **r})
    pd.DataFrame(per_seed_rows).to_csv(OUTPUT_DIR / "per_seed_results.csv", index=False)

    for name, (y_test, y_pred) in confusion_data.items():
        cm = confusion_matrix(y_test, y_pred, labels=list(range(32)))
        cm_df = pd.DataFrame(cm, index=FDI_CODES, columns=FDI_CODES)
        cm_df.to_csv(OUTPUT_DIR / f"confusion_matrix_{name}_seed{CONFUSION_MATRIX_SEED}.csv")

        fig, ax = plt.subplots(figsize=(11, 10))
        im = ax.imshow(cm, cmap="viridis")
        ax.set_xticks(range(32))
        ax.set_yticks(range(32))
        ax.set_xticklabels(FDI_CODES, rotation=90, fontsize=7)
        ax.set_yticklabels(FDI_CODES, fontsize=7)
        ax.set_xlabel("Predicted FDI code")
        ax.set_ylabel("True FDI code")
        ax.set_title(f"{name} - confusion matrix (seed {CONFUSION_MATRIX_SEED})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"confusion_matrix_{name}_seed{CONFUSION_MATRIX_SEED}.png", dpi=150)
        plt.close(fig)

    print(f"\nSaved summary.csv, per_seed_results.csv, and confusion matrices to {OUTPUT_DIR}")
    return summary_df


if __name__ == "__main__":
    main()
