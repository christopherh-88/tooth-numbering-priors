"""Write the two Phase 3 batch 1 kernels (cpu_repro/cv/PHASE3_RULES.md).

    python cpu_repro/cv/kaggle/make_phase3_kernels.py --out <dir>

tooth-numbering-cv-shift-s0 (T4): shift_test.py on all three detectors'
fold models. tooth-numbering-cv-calib-s0 (CPU): calib_sweep.py.
tooth-numbering-cv-matchorder-s0 (CPU): the sweep under both matcher
orders (PHASE3_RULES.md note). tooth-numbering-cv-confmatch-s0 (CPU):
Section 60 scoring, calibration and label check redone with confidence-first
matching (RESULTS.md Section 64). tooth-numbering-cv-shiftctx-s0 (T4):
shift test v2 and context masking (PHASE3_BATCH2_RULES.md items 1 and 2).
tooth-numbering-cv-batch3-s0 (CPU): batch3.py (PHASE3_BATCH3_RULES.md).
tooth-numbering-cv-dentist-s0 (CPU): dentist_sheet.py (LABEL_CHECK_RULES.md check C).
tooth-numbering-cv-seed1-score (CPU): seed 0 and seed 1 scored with YOLOv8x and
RT-DETR-l, confidence-first (PHASE3_BATCH2_RULES.md item 3).
Each kernel
embeds the scripts it runs (written to disk at start, so the code that ran
is visible in the kernel itself), reads the training and scoring kernels'
outputs as kernel sources, and checks out only the data files it needs
from GitHub at commit 6ea25a5.
"""
import argparse
import json
from pathlib import Path

USER = "christopherhuang88"
COMMIT = "6ea25a58cb0a99b402cbb24898c10641fcb679a9"
REPO_URL = "https://github.com/christopherh-88/tooth-numbering-priors.git"
HERE = Path(__file__).resolve().parents[1]
TRAIN_KERNELS = [f"{USER}/tooth-numbering-cv-{d}-s0" for d in ("yolov8x", "rtdetr-l", "fasterrcnn")]

HEADER = '''import shutil, subprocess, sys
from pathlib import Path

FILES = {files!r}
for name, text in FILES.items():
    Path(name).write_text(text)
sys.path.insert(0, ".")
REPO = Path("/kaggle/working/repo")
subprocess.run(["git", "clone", "--quiet", "--no-checkout", "{repo_url}", str(REPO)], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "--quiet", "{commit}", "--", *{paths!r}], check=True)
'''

SHIFT_BODY = '''subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "ultralytics==8.4.143"], check=True)
import shift_test
sys.argv = ["shift_test.py", "--repo", str(REPO), "--models", "/kaggle/input", "--out", "/kaggle/working/shift"]
shift_test.main()
shutil.rmtree(REPO)
'''

CALIB_BODY = '''import calib_sweep
per_tooth = sorted(Path("/kaggle/input").rglob("per_tooth_predictions.csv"))
assert len(per_tooth) == 1, per_tooth
args = ["calib_sweep.py", "--per-tooth", str(per_tooth[0]), "--repo", str(REPO), "--out", "/kaggle/working/calib"]
for det in ["yolov8x", "rtdetr_l", "fasterrcnn"]:
    found = sorted(p for p in Path("/kaggle/input").rglob(f"{det}_cvseed0") if p.is_dir())
    assert len(found) == 1, (det, found)
    args += ["--det", f"{det}={found[0]}"]
sys.argv = args
calib_sweep.main()
shutil.rmtree(REPO)
'''


MATCH_BODY = CALIB_BODY.replace('"/kaggle/working/calib"]', '"/kaggle/working/match_order", "--match-order-only"]')


CONFMATCH_BODY = '''subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "cleanlab"], check=True)
import numpy as np, pandas as pd
import calib_sweep, label_check, score_cv
OUT = Path("/kaggle/working/confmatch")
old = sorted(Path("/kaggle/input").rglob("per_tooth_predictions.csv"))
assert len(old) == 1, old
args = ["score_cv.py", "--out", str(OUT / "score"), "--match-order", "conf",
        "--boxes", str(REPO / "cpu_repro/cv/boxes.csv"), "--folds", str(REPO / "cpu_repro/cv/folds.csv")]
for det in ["yolov8x", "rtdetr_l", "fasterrcnn"]:
    found = sorted(p for p in Path("/kaggle/input").rglob(f"{det}_cvseed0") if p.is_dir())
    assert len(found) == 1, (det, found)
    args += ["--det", f"{det}={found[0]}"]
sys.argv = args
score_cv.main()
new = OUT / "score" / "per_tooth_predictions.csv"

cal = calib_sweep.calibration(pd.read_csv(new), np.random.default_rng(0))
cal.to_csv(OUT / "calibration_joint_failures.csv", index=False, float_format="%.4f")
print(cal.round(3).to_string(index=False), flush=True)

sys.argv = ["label_check.py", "--per-tooth", str(new), "--out", str(OUT / "label_check"),
            "--roboflow", str(REPO / "Dataset/yolo_train_dataset")]
label_check.main()

def joint_set(path):
    t = pd.read_csv(path)
    j = t["coord_pred"] != t["class_id"]
    for d in calib_sweep.DETS:
        j &= (t[f"{d}_pred"] != t["class_id"]) & t[f"{d}_pred"].notna()
    p = np.stack([t[f"{d}_pred"].to_numpy() for d in calib_sweep.DETS])
    j &= (p == p[0]).all(0)
    return set(zip(t.loc[j, "image_id"], t.loc[j, "fdi"]))
a, b = joint_set(old[0]), joint_set(new)
print(f"joint same-wrong: IoU-first {len(a)}, conf-first {len(b)}, only IoU-first {len(a - b)}, "
      f"only conf-first {len(b - a)}, symmetric difference {len(a ^ b)}")
pd.DataFrame([dict(image_id=i, fdi=f, in_iou_first=(i, f) in a, in_conf_first=(i, f) in b)
              for i, f in sorted(a ^ b)]).to_csv(OUT / "joint_set_diff.csv", index=False)
shutil.rmtree(REPO)
'''


SHIFTCTX_BODY = '''subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "ultralytics==8.4.143"], check=True)
import context_test, shift_test
per_tooth = sorted(Path("/kaggle/input").rglob("per_tooth_predictions.csv"))
assert len(per_tooth) == 1, per_tooth
sys.argv = ["shift_test.py", "--repo", str(REPO), "--models", "/kaggle/input", "--out", "/kaggle/working/shift_v2",
            "--match-order", "conf", "--save-raw"]
shift_test.main()
sys.argv = ["context_test.py", "--repo", str(REPO), "--models", "/kaggle/input", "--per-tooth", str(per_tooth[0]),
            "--out", "/kaggle/working/context"]
context_test.main()
shutil.rmtree(REPO)
'''


BATCH3_BODY = '''import batch3
args = ["batch3.py", "--repo", str(REPO)]
for det in ["yolov8x", "rtdetr_l", "fasterrcnn"]:
    found = sorted(p for p in Path("/kaggle/input").rglob(f"{det}_cvseed0") if p.is_dir())
    assert len(found) == 1, (det, found)
    args += ["--det", f"{det}={found[0]}"]
sys.argv = args + ["--out", "/kaggle/working/batch3_smoke", "--images", "20"]
batch3.main()
print("smoke run done; full run follows", flush=True)
sys.argv = args + ["--out", "/kaggle/working/batch3"]
batch3.main()
shutil.rmtree(REPO)
'''


DENTIST_BODY = '''import dentist_sheet
per_tooth = sorted(Path("/kaggle/input").rglob("per_tooth_predictions.csv"))
assert len(per_tooth) == 1, per_tooth
sys.argv = ["dentist_sheet.py", "--per-tooth", str(per_tooth[0]),
            "--images", str(REPO / "Dataset/bb_u_net_dataset/panoramic_x_rays"), "--out", "/kaggle/working/dentist"]
dentist_sheet.main()
shutil.rmtree(REPO)
'''


SEED1_BODY = '''import score_cv
subprocess.run(["git", "-C", str(REPO), "fetch", "--quiet", "origin", "f12fa7bb0f34828ec967261d263ef1a985f5fa47"], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "--quiet", "f12fa7bb0f34828ec967261d263ef1a985f5fa47", "--",
                "cpu_repro/cv/folds_seed1.csv"], check=True)
folds = {0: REPO / "cpu_repro/cv/folds.csv", 1: REPO / "cpu_repro/cv/folds_seed1.csv"}
for seed in (0, 1):
    args = ["score_cv.py", "--out", f"/kaggle/working/seed{seed}_2det", "--match-order", "conf",
            "--boxes", str(REPO / "cpu_repro/cv/boxes.csv"), "--folds", str(folds[seed])]
    for det in ["yolov8x", "rtdetr_l"]:
        found = sorted(p for p in Path("/kaggle/input").rglob(f"{det}_cvseed{seed}") if p.is_dir())
        assert len(found) == 1, (det, seed, found)
        args += ["--det", f"{det}={found[0]}"]
    print(f"===== seed {seed}", flush=True)
    sys.argv = args
    score_cv.main()
shutil.rmtree(REPO)
'''


def write(out, name, files, paths, body, gpu, sources):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    code = HEADER.format(files={f: (HERE / f).read_text() for f in files}, repo_url=REPO_URL,
                         commit=COMMIT, paths=paths) + body
    (d / f"{name}.py").write_text(code)
    meta = {"id": f"{USER}/{name}", "title": name, "code_file": f"{name}.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "kernel_sources": sources,
            "competition_sources": [], "model_sources": []}
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote", d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    csvs = ["cpu_repro/cv/boxes.csv", "cpu_repro/cv/folds.csv"]
    write(a.out, "tooth-numbering-cv-shift-s0", ["shift_test.py"],
          csvs + ["Dataset/bb_u_net_dataset/panoramic_x_rays"], SHIFT_BODY, True, TRAIN_KERNELS)
    write(a.out, "tooth-numbering-cv-calib-s0", ["score_cv.py", "calib_sweep.py"], csvs, CALIB_BODY,
          False, TRAIN_KERNELS + [f"{USER}/tooth-numbering-cv-score-s0"])
    write(a.out, "tooth-numbering-cv-matchorder-s0", ["score_cv.py", "calib_sweep.py"], csvs, MATCH_BODY,
          False, TRAIN_KERNELS + [f"{USER}/tooth-numbering-cv-score-s0"])
    write(a.out, "tooth-numbering-cv-confmatch-s0", ["score_cv.py", "calib_sweep.py", "label_check.py"],
          csvs + ["Dataset/yolo_train_dataset"], CONFMATCH_BODY, False,
          TRAIN_KERNELS + [f"{USER}/tooth-numbering-cv-score-s0"])
    write(a.out, "tooth-numbering-cv-shiftctx-s0", ["shift_test.py", "context_test.py"],
          csvs + ["Dataset/bb_u_net_dataset/panoramic_x_rays"], SHIFTCTX_BODY, True,
          TRAIN_KERNELS + [f"{USER}/tooth-numbering-cv-confmatch-s0"])
    seed1_sources = [f"{USER}/tooth-numbering-cv-{d}-s{s}" for d in ("yolov8x", "rtdetr-l") for s in (0, 1)]
    write(a.out, "tooth-numbering-cv-seed1-score", ["score_cv.py"], csvs, SEED1_BODY, False, seed1_sources)
    write(a.out, "tooth-numbering-cv-dentist-s0", ["dentist_sheet.py"], ["Dataset/bb_u_net_dataset/panoramic_x_rays"],
          DENTIST_BODY, False, [f"{USER}/tooth-numbering-cv-confmatch-s0"])
    write(a.out, "tooth-numbering-cv-batch3-s0", ["score_cv.py", "batch3.py"], csvs, BATCH3_BODY, False,
          TRAIN_KERNELS)


if __name__ == "__main__":
    main()
