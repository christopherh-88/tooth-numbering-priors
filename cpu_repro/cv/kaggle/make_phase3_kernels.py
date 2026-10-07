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
tooth-numbering-cv-gapcheck-s0 (CPU): gap_check.py (PHASE3_BATCH3_RULES.md note).
tooth-numbering-cv-dentex-indomain (CPU): dentex_external.py --saved on the
models trained within DENTEX (DENTEX_INDOMAIN_RULES.md), after checking that
the --saved path reproduces Section 85 from its own detections.
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


GAP_BODY = '''import gap_check
per_tooth = sorted(Path("/kaggle/input").rglob("per_tooth_predictions.csv"))
assert len(per_tooth) == 1, per_tooth
sys.argv = ["gap_check.py", "--per-tooth", str(per_tooth[0]), "--out", "/kaggle/working/gap_check"]
gap_check.main()
shutil.rmtree(REPO)
'''


GAPINT_BODY = '''subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "ultralytics==8.4.143"], check=True)
import gap_intervention
per_tooth = sorted(Path("/kaggle/input").rglob("per_tooth_predictions.csv"))
assert len(per_tooth) == 1, per_tooth
args = ["gap_intervention.py", "--repo", str(REPO), "--models", "/kaggle/input", "--per-tooth", str(per_tooth[0])]
sys.argv = args + ["--out", "/kaggle/working/gap_intervention_smoke", "--images", "10"]
gap_intervention.main()
sys.argv = args + ["--out", "/kaggle/working/gap_intervention"]
gap_intervention.main()
shutil.rmtree(REPO)
'''


DRIFT_BODY = '''import drift_check
per_tooth = sorted(Path("/kaggle/input").rglob("per_tooth_predictions.csv"))
assert len(per_tooth) == 1, per_tooth
args = ["drift_check.py", "--repo", str(REPO), "--per-tooth", str(per_tooth[0])]
sys.argv = args + ["--out", "/kaggle/working/drift_smoke", "--images", "40"]
drift_check.main()
sys.argv = args + ["--out", "/kaggle/working/drift"]
drift_check.main()
shutil.rmtree(REPO)
'''


F1_BODY = '''import pandas as pd
subprocess.run(["git", "-C", str(REPO), "checkout", "--quiet", "a78cfa6", "--", "benchmark"], check=True)
OUT = Path("/kaggle/working/f1")
OUT.mkdir(parents=True, exist_ok=True)
for det in ["yolov8x", "rtdetr_l", "fasterrcnn"]:
    run = next(Path("/kaggle/input").rglob(f"{det}_cvseed0"))
    files = sorted(run.glob("fold*/test_detections.csv"))
    assert len(files) == 5, (det, files)
    d = pd.concat([pd.read_csv(f) for f in files]).rename(
        columns={"x1": "x_min", "y1": "y_min", "x2": "x_max", "y2": "y_max"})
    d.to_csv(f"/kaggle/temp_{det}.csv", index=False)
    print("=====", det, len(d), "detections", flush=True)
    subprocess.run([sys.executable, str(REPO / "benchmark/score.py"), "--pred", f"/kaggle/temp_{det}.csv",
                    "--out", str(OUT / f"{det}_score.csv")], check=True)
shutil.rmtree(REPO)
'''


SEED1_3DET_BODY = SEED1_BODY.replace('["yolov8x", "rtdetr_l"]', '["yolov8x", "rtdetr_l", "fasterrcnn"]').replace(
    "_2det", "_3det")


CLOSURE_BODY = GAPINT_BODY.replace("gap_intervention", "closure_intervention").replace('"10"]', '"10"]')


AUGGRADE_BODY = '''import gap_augment_grade
# Training kernels leave a repo copy in their output; use only the real result folders.
per_tooth = sorted(p for p in Path("/kaggle/input").rglob("per_tooth_predictions.csv")
                   if "tooth-numbering-cv-confmatch-s0" in str(p) and "/repo/" not in str(p))
assert len(per_tooth) == 1, per_tooth
run = {r: sorted(p for p in Path("/kaggle/input").rglob(f"yolov8x_cvseed0{s}") if p.is_dir() and "/cv_out/" in str(p))
       for r, s in (("original", ""), ("control", "_tseed1"), ("augmented", "_aug"))}
assert all(len(v) == 1 for v in run.values()), run
sys.argv = ["gap_augment_grade.py", "--per-tooth", str(per_tooth[0]), "--out", "/kaggle/working/augment"]
for r, v in run.items():
    sys.argv += [f"--{r}", str(v[0])]
gap_augment_grade.main()
shutil.rmtree(REPO)
'''


SEED2_BODY = (SEED1_3DET_BODY.replace("if p.is_dir())", "if p.is_dir() and \"/cv_out/\" in str(p))").replace("f12fa7bb0f34828ec967261d263ef1a985f5fa47", "01cfa88b4a73d07cc7b84b6c96242026cdb6faa2")
              .replace("folds_seed1.csv", "folds_seed2.csv").replace("1: REPO", "2: REPO")
              .replace("for seed in (0, 1):", "for seed in (0, 2):"))


CLOSURE_S1_BODY = '''subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "ultralytics==8.4.143"], check=True)
import closure_intervention
subprocess.run(["git", "-C", str(REPO), "fetch", "--quiet", "origin", "01cfa88b4a73d07cc7b84b6c96242026cdb6faa2"], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "--quiet", "01cfa88b4a73d07cc7b84b6c96242026cdb6faa2", "--",
                "cpu_repro/cv/folds_seed1.csv"], check=True)
per_tooth = sorted(p for p in Path("/kaggle/input").rglob("per_tooth_predictions.csv")
                   if "seed1_3det" in str(p) and "/repo/" not in str(p))
assert len(per_tooth) == 1, per_tooth
args = ["closure_intervention.py", "--repo", str(REPO), "--models", "/kaggle/input", "--per-tooth", str(per_tooth[0]),
        "--cv-seed", "1", "--folds", str(REPO / "cpu_repro/cv/folds_seed1.csv")]
sys.argv = args + ["--out", "/kaggle/working/closure_s1_smoke", "--images", "10"]
closure_intervention.main()
sys.argv = args + ["--out", "/kaggle/working/closure_s1"]
closure_intervention.main()
shutil.rmtree(REPO)
'''


TIP_BODY = CLOSURE_BODY.replace("str(per_tooth[0])]", "str(per_tooth[0]), \"--mode\", \"tip\"]").replace(
    "closure_intervention_smoke", "closure_tip_smoke").replace('"/kaggle/working/closure_intervention"]',
                                                               '"/kaggle/working/closure_tip"]')
CLOSURE_S2_BODY = (CLOSURE_S1_BODY.replace("folds_seed1.csv", "folds_seed2.csv").replace("seed1_3det", "seed2_3det")
                   .replace('"--cv-seed", "1"', '"--cv-seed", "2"').replace("closure_s1", "closure_s2"))
TSEED_BODY = '''import score_cv
for tag, suffix in (("seed0_3det", ""), ("tseed1_3det", "_tseed1")):
    args = ["score_cv.py", "--out", f"/kaggle/working/{tag}", "--match-order", "conf",
            "--boxes", str(REPO / "cpu_repro/cv/boxes.csv"), "--folds", str(REPO / "cpu_repro/cv/folds.csv")]
    for det in ["yolov8x", "rtdetr_l", "fasterrcnn"]:
        found = sorted(p for p in Path("/kaggle/input").rglob(f"{det}_cvseed0{suffix}")
                       if p.is_dir() and "/cv_out/" in str(p))
        assert len(found) == 1, (det, suffix, found)
        args += ["--det", f"{det}={found[0]}"]
    print(f"===== {tag}", flush=True)
    sys.argv = args
    score_cv.main()
shutil.rmtree(REPO)
'''


TIP_S1_BODY = CLOSURE_S1_BODY.replace('"--cv-seed", "1",', '"--cv-seed", "1", "--mode", "tip",').replace(
    "closure_s1", "closure_tip_s1")
TIP_S2_BODY = CLOSURE_S2_BODY.replace('"--cv-seed", "2",', '"--cv-seed", "2", "--mode", "tip",').replace(
    "closure_s2", "closure_tip_s2")
GAP_FLAG_BODY = '''import gap_flag
subprocess.run(["git", "-C", str(REPO), "fetch", "--quiet", "origin", "01cfa88b4a73d07cc7b84b6c96242026cdb6faa2"], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "--quiet", "01cfa88b4a73d07cc7b84b6c96242026cdb6faa2", "--",
                "cpu_repro/cv/folds_seed1.csv", "cpu_repro/cv/folds_seed2.csv"], check=True)
for seed, folds in ((0, "folds.csv"), (1, "folds_seed1.csv"), (2, "folds_seed2.csv")):
    args = ["gap_flag.py", "--repo", str(REPO), "--folds", str(REPO / "cpu_repro/cv" / folds), "--seed", str(seed)]
    for det in ["yolov8x", "rtdetr_l", "fasterrcnn"]:
        found = sorted(p for p in Path("/kaggle/input").rglob(f"{det}_cvseed{seed}")
                       if p.is_dir() and "/repo/" not in str(p))
        assert len(found) == 1, (det, seed, found)
        args += ["--det", f"{det}={found[0]}"]
    if seed == 0:
        sys.argv = args + ["--out", "/kaggle/working/gap_flag_smoke", "--images", "20"]
        gap_flag.main()
        print("smoke run done; full runs follow", flush=True)
    sys.argv = args + ["--out", "/kaggle/working/gap_flag"]
    gap_flag.main()
shutil.rmtree(REPO)
'''
DENTEX_BODY = '''subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "ultralytics==8.4.143"], check=True)
import dentex_external
xrays = sorted(p for p in Path("/kaggle/input").rglob("xrays")
               if p.is_dir() and p.parent.name == "quadrant_enumeration" and "/repo/" not in str(p))
assert len(xrays) == 1, xrays
args = ["dentex_external.py", "--labels", str(REPO / "cpu_repro/anomaly_scan/dentex_raw/train_quadrant_enumeration.json"),
        "--images", str(xrays[0]), "--models", "/kaggle/input", "--ufba-boxes", str(REPO / "cpu_repro/cv/boxes.csv"),
        "--ufba-folds", str(REPO / "cpu_repro/cv/folds.csv")]
sys.argv = args + ["--out", "/kaggle/working/dentex_smoke", "--images-limit", "10"]
dentex_external.main()
print("smoke run done; full run follows", flush=True)
sys.argv = args + ["--out", "/kaggle/working/dentex"]
dentex_external.main()
shutil.rmtree(REPO)
'''

DENTEX_INDOMAIN_BODY = '''subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "ultralytics==8.4.143"], check=True)
import pandas as pd
import dentex_external
ext = sorted(p for p in Path("/kaggle/input").rglob("dentex_detections.csv") if p.parent.name == "dentex")
assert len(ext) == 1, ext
args = ["dentex_external.py", "--labels", str(REPO / "cpu_repro/anomaly_scan/dentex_raw/train_quadrant_enumeration.json"),
        "--ufba-boxes", str(REPO / "cpu_repro/cv/boxes.csv"), "--ufba-folds", str(REPO / "cpu_repro/cv/folds.csv")]
# Check: the --saved path, fed the Section 85 detections split by fold, reproduces Section 85.
teeth = pd.read_csv(ext[0].parent / "dentex_teeth.csv")
fold_of = teeth.drop_duplicates("image_id").set_index("image_id")["fold"]
d85 = pd.read_csv(ext[0])
chk = Path("/kaggle/working/check85")
for (d, f), g in d85.groupby(["detector", d85["image_id"].map(fold_of)]):
    o = chk / f"{d}_cvseed0_dentex" / f"fold{f}"
    o.mkdir(parents=True, exist_ok=True)
    g.drop(columns="detector").to_csv(o / "test_detections.csv", index=False)
sys.argv = args + ["--saved", str(chk), "--out", "/kaggle/working/check85_out"]
dentex_external.main()
a = pd.read_csv("/kaggle/working/check85_out/dentex_summary.csv").set_index("metric")[["point", "lo", "hi"]]
b = pd.read_csv(ext[0].parent / "dentex_summary.csv").set_index("metric")[["point", "lo", "hi"]]
assert ((a - b).abs() < 1e-9).all().all(), (a - b)
shutil.rmtree(chk)
print("check passed: --saved reproduces Section 85", flush=True)
sys.argv = args + ["--saved", "/kaggle/input", "--out", "/kaggle/working/dentex_indomain"]
dentex_external.main()
shutil.rmtree(REPO)
'''

DENTEX_TSEED1_BODY = '''subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "ultralytics==8.4.143"], check=True)
import dentex_external
sys.argv = ["dentex_external.py", "--labels", str(REPO / "cpu_repro/anomaly_scan/dentex_raw/train_quadrant_enumeration.json"),
            "--ufba-boxes", str(REPO / "cpu_repro/cv/boxes.csv"), "--ufba-folds", str(REPO / "cpu_repro/cv/folds.csv"),
            "--saved", "/kaggle/input", "--saved-tag", "_dentex_tseed1", "--out", "/kaggle/working/dentex_tseed1"]
dentex_external.main()
shutil.rmtree(REPO)
'''


def write(out, name, files, paths, body, gpu, sources, datasets=()):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    code = HEADER.format(files={f: (HERE / f).read_text() for f in files}, repo_url=REPO_URL,
                         commit=COMMIT, paths=paths) + body
    (d / f"{name}.py").write_text(code)
    meta = {"id": f"{USER}/{name}", "title": name, "code_file": f"{name}.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": list(datasets), "kernel_sources": sources,
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
    seed1_3det_sources = [f"{USER}/tooth-numbering-cv-{d}-s{s}" for d in ("yolov8x", "rtdetr-l", "fasterrcnn")
                          for s in (0, 1)]
    write(a.out, "tooth-numbering-cv-seed1-score3", ["score_cv.py"], csvs, SEED1_3DET_BODY, False, seed1_3det_sources)
    write(a.out, "tooth-numbering-cv-gapcheck-s0", ["gap_check.py"], csvs, GAP_BODY, False,
          [f"{USER}/tooth-numbering-cv-confmatch-s0"])
    write(a.out, "tooth-numbering-cv-dentist-s0", ["dentist_sheet.py"], ["Dataset/bb_u_net_dataset/panoramic_x_rays"],
          DENTIST_BODY, False, [f"{USER}/tooth-numbering-cv-confmatch-s0"])
    write(a.out, "tooth-numbering-cv-gapint-s0", ["shift_test.py", "gap_check.py", "gap_intervention.py"],
          csvs + ["Dataset/bb_u_net_dataset/panoramic_x_rays", "Dataset/bb_u_net_dataset/labels"], GAPINT_BODY, True,
          TRAIN_KERNELS + [f"{USER}/tooth-numbering-cv-confmatch-s0"])
    write(a.out, "tooth-numbering-cv-drift-s0", ["gap_check.py", "drift_check.py"],
          csvs + ["Dataset/bb_u_net_dataset/labels"], DRIFT_BODY, False,
          [f"{USER}/tooth-numbering-cv-confmatch-s0"])
    write(a.out, "tooth-numbering-cv-f1-s0", [], ["cpu_repro/cv/folds.csv"], F1_BODY, False, TRAIN_KERNELS)
    write(a.out, "tooth-numbering-cv-closure-s0",
          ["shift_test.py", "gap_check.py", "gap_intervention.py", "closure_intervention.py"],
          csvs + ["Dataset/bb_u_net_dataset/panoramic_x_rays", "Dataset/bb_u_net_dataset/labels"], CLOSURE_BODY,
          True, TRAIN_KERNELS + [f"{USER}/tooth-numbering-cv-confmatch-s0"])
    write(a.out, "tooth-numbering-cv-augment-grade", ["score_cv.py", "gap_check.py", "gap_augment_grade.py"], csvs,
          AUGGRADE_BODY, False, [f"{USER}/tooth-numbering-cv-yolov8x-s0", f"{USER}/tooth-numbering-cv-yolov8x-s0-tseed1",
                                 f"{USER}/tooth-numbering-cv-yolov8x-s0-aug", f"{USER}/tooth-numbering-cv-confmatch-s0"])
    write(a.out, "tooth-numbering-cv-seed2-score3", ["score_cv.py"], csvs, SEED2_BODY, False,
          [f"{USER}/tooth-numbering-cv-{d}-s{s}" for d in ("yolov8x", "rtdetr-l", "fasterrcnn") for s in (0, 2)])
    write(a.out, "tooth-numbering-cv-closure-s1",
          ["shift_test.py", "gap_check.py", "gap_intervention.py", "closure_intervention.py"],
          csvs + ["Dataset/bb_u_net_dataset/panoramic_x_rays", "Dataset/bb_u_net_dataset/labels"], CLOSURE_S1_BODY,
          True, [f"{USER}/tooth-numbering-cv-{d}-s1" for d in ("yolov8x", "rtdetr-l", "fasterrcnn")]
          + [f"{USER}/tooth-numbering-cv-seed1-score3"])
    closure_files = ["shift_test.py", "gap_check.py", "gap_intervention.py", "closure_intervention.py"]
    closure_paths = csvs + ["Dataset/bb_u_net_dataset/panoramic_x_rays", "Dataset/bb_u_net_dataset/labels"]
    write(a.out, "tooth-numbering-cv-tip-s0", closure_files, closure_paths, TIP_BODY, True,
          TRAIN_KERNELS + [f"{USER}/tooth-numbering-cv-confmatch-s0"])
    write(a.out, "tooth-numbering-cv-closure-s2", closure_files, closure_paths, CLOSURE_S2_BODY, True,
          [f"{USER}/tooth-numbering-cv-{d}-s2" for d in ("yolov8x", "rtdetr-l", "fasterrcnn")]
          + [f"{USER}/tooth-numbering-cv-seed2-score3"])
    write(a.out, "tooth-numbering-cv-tseed-score3", ["score_cv.py"], csvs, TSEED_BODY, False,
          TRAIN_KERNELS + [f"{USER}/tooth-numbering-cv-{d}-s0-tseed1" for d in ("yolov8x", "rtdetr-l", "fasterrcnn")])
    write(a.out, "tooth-numbering-cv-tip-s1", closure_files, closure_paths, TIP_S1_BODY, True,
          [f"{USER}/tooth-numbering-cv-{d}-s1" for d in ("yolov8x", "rtdetr-l", "fasterrcnn")]
          + [f"{USER}/tooth-numbering-cv-seed1-score3"])
    write(a.out, "tooth-numbering-cv-tip-s2", closure_files, closure_paths, TIP_S2_BODY, True,
          [f"{USER}/tooth-numbering-cv-{d}-s2" for d in ("yolov8x", "rtdetr-l", "fasterrcnn")]
          + [f"{USER}/tooth-numbering-cv-seed2-score3"])
    write(a.out, "tooth-numbering-cv-gapflag", ["score_cv.py", "batch3.py", "gap_flag.py"], csvs, GAP_FLAG_BODY,
          False, [f"{USER}/tooth-numbering-cv-{d}-s{s}" for s in (0, 1, 2) for d in ("yolov8x", "rtdetr-l", "fasterrcnn")])
    write(a.out, "tooth-numbering-cv-dentex-external",
          ["score_cv.py", "batch3.py", "gap_flag.py", "shift_test.py", "gap_check.py", "gap_intervention.py",
           "closure_intervention.py", "dentex_external.py"],
          csvs + ["cpu_repro/anomaly_scan/dentex_raw/train_quadrant_enumeration.json"], DENTEX_BODY, True,
          TRAIN_KERNELS, ["truthisneverlinear/dentex-challenge-2023"])
    write(a.out, "tooth-numbering-cv-batch3-s0", ["score_cv.py", "batch3.py"], csvs, BATCH3_BODY, False,
          TRAIN_KERNELS)
    write(a.out, "tooth-numbering-cv-dentex-indomain",
          ["score_cv.py", "batch3.py", "gap_flag.py", "shift_test.py", "gap_check.py", "gap_intervention.py",
           "closure_intervention.py", "dentex_external.py"],
          csvs + ["cpu_repro/anomaly_scan/dentex_raw/train_quadrant_enumeration.json"], DENTEX_INDOMAIN_BODY, False,
          [f"{USER}/tooth-numbering-cv-{d}-dentex" for d in ("yolov8x", "rtdetr-l", "fasterrcnn")]
          + [f"{USER}/tooth-numbering-cv-dentex-external"])
    write(a.out, "tooth-numbering-cv-dentex-tseed1",
          ["score_cv.py", "batch3.py", "gap_flag.py", "shift_test.py", "gap_check.py", "gap_intervention.py",
           "closure_intervention.py", "dentex_external.py"],
          csvs + ["cpu_repro/anomaly_scan/dentex_raw/train_quadrant_enumeration.json"], DENTEX_TSEED1_BODY, False,
          [f"{USER}/tooth-numbering-cv-{d}-dentex-tseed1" for d in ("yolov8x", "rtdetr-l", "fasterrcnn")])


if __name__ == "__main__":
    main()
