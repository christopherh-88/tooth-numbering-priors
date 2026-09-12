"""Bootstrap CIs for Section 25's headline DenPAR numbers (top-1/quadrant
accuracy, real and shuffled-label control), resampled at the image level.
Reuses load_instances()/grouped_split() from build_coord_baseline_denpar.py
unchanged - refits the seed-0 GBT model once (deterministic, already done
for Section 25) and keeps per-instance identity for resampling, which the
original script's summary.csv/per_seed_results.csv output did not."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_coord_baseline_denpar import load_instances, grouped_split, FEATURE_COLS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "coord_baseline"))
from build_coord_baseline import quadrant_of

OUT_DIR = Path(__file__).resolve().parent
N_BOOTSTRAP = 2000
SEED = 0


def bootstrap_ci(image_ids, per_image_data, metric_fn, n_boot=N_BOOTSTRAP, seed=0):
    rng = np.random.RandomState(seed)
    n_images = len(image_ids)
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(image_ids, size=n_images, replace=True)
        rows = pd.concat([per_image_data[iid] for iid in sampled], ignore_index=True)
        values.append(metric_fn(rows))
    point = metric_fn(pd.concat([per_image_data[iid] for iid in image_ids], ignore_index=True))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return point, lo, hi


def main():
    df = load_instances()
    train_df, test_df = grouped_split(df, SEED, 0.2)
    x_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df["class_id"].to_numpy()
    x_test = test_df[FEATURE_COLS].to_numpy()
    y_test = test_df["class_id"].to_numpy()

    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    result = test_df[["image_id", "class_id"]].copy()
    result["pred"] = y_pred
    result = result.rename(columns={"class_id": "true_class"})

    rng = np.random.RandomState(0)
    y_train_shuffled = y_train.copy()
    rng.shuffle(y_train_shuffled)
    clf_shuffled = HistGradientBoostingClassifier(random_state=0)
    clf_shuffled.fit(x_train, y_train_shuffled)
    y_pred_shuffled = clf_shuffled.predict(x_test)
    result_shuffled = test_df[["image_id", "class_id"]].copy()
    result_shuffled["pred"] = y_pred_shuffled
    result_shuffled = result_shuffled.rename(columns={"class_id": "true_class"})

    def top1_metric(d):
        return float(np.mean(d["true_class"] == d["pred"]))

    def quadrant_metric(d):
        return float(np.mean(quadrant_of(d["true_class"].to_numpy())
                              == quadrant_of(d["pred"].to_numpy())))

    image_ids = result["image_id"].unique()
    per_image = {iid: sub for iid, sub in result.groupby("image_id")}
    per_image_shuffled = {iid: sub for iid, sub in result_shuffled.groupby("image_id")}

    print("=" * 70)
    print(f"DenPAR bootstrap CIs (image-level resampling, n={N_BOOTSTRAP})")
    print("=" * 70)

    out = {}
    for name, data, metric_fn in [
        ("top1_acc", per_image, top1_metric),
        ("quadrant_acc", per_image, quadrant_metric),
        ("top1_acc_shuffled_control", per_image_shuffled, top1_metric),
    ]:
        point, lo, hi = bootstrap_ci(image_ids, data, metric_fn)
        out[name] = (point, lo, hi)
        print(f"  {name:28s} {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    pd.DataFrame([{"metric": k, "point": v[0], "ci_lo": v[1], "ci_hi": v[2]} for k, v in out.items()]
                 ).to_csv(OUT_DIR / "bootstrap_ci_section25.csv", index=False)
    print(f"\nSaved bootstrap_ci_section25.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
