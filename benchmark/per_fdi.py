"""Per-tooth-number top-1 for the reference models (RESULTS.md Section 74).

    python benchmark/per_fdi.py

Reads benchmark/teeth.csv (split 0). Writes benchmark/per_fdi.csv: for each
FDI number, the number of labeled teeth, top-1 of each model with a 95% CI
from 10,000 bootstrap draws of X-rays, the joint failures, and the share of
teeth next to a gap. Also prints a summary by tooth type.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODELS = {"yolov8x": "yolov8x_pred", "rtdetr_l": "rtdetr_l_pred", "fasterrcnn": "fasterrcnn_pred",
          "position_only": "position_only_pred_seed0"}
TYPE = {"1": "incisor", "2": "incisor", "3": "canine", "4": "premolar", "5": "premolar",
        "6": "molar", "7": "molar", "8": "third molar"}
N_BOOT = 10_000


def boot_rate(ok, image_id, rng):
    s = pd.DataFrame(dict(image_id=image_id, n=1, ok=ok.astype(int))).groupby("image_id").sum().to_numpy(float)
    b = s[rng.integers(0, len(s), size=(N_BOOT, len(s)))].sum(1)
    return np.percentile(100 * b[:, 1] / b[:, 0], [2.5, 97.5])


def main():
    t = pd.read_csv(HERE / "teeth.csv", dtype={"fdi": str})
    rng = np.random.default_rng(0)
    rows = []
    for fdi, g in t.groupby("fdi"):
        r = dict(fdi=fdi, type=TYPE[fdi[1]], n=len(g), joint_failures=int(g["joint_failure"].sum()),
                 next_to_gap_pct=100 * g["next_to_gap"].mean())
        for name, col in MODELS.items():
            ok = g[col] == g["class_id"]
            r[f"{name}_top1"] = 100 * ok.mean()
            r[f"{name}_lo"], r[f"{name}_hi"] = boot_rate(ok, g["image_id"], rng)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "per_fdi.csv", index=False, float_format="%.2f")

    pd.set_option("display.width", 250)
    cols = ["fdi", "n", "joint_failures", "next_to_gap_pct"] + [f"{m}_top1" for m in MODELS]
    print(out[cols].round(1).to_string(index=False), "\n")
    by_type = t.assign(type=t["fdi"].str[1].map(TYPE)).groupby("type").apply(
        lambda g: pd.Series({"n": len(g), "joint_failures": int(g["joint_failure"].sum()),
                             **{m: 100 * (g[c] == g["class_id"]).mean() for m, c in MODELS.items()}}),
        include_groups=False)
    print(by_type.round(1).to_string())


if __name__ == "__main__":
    main()
