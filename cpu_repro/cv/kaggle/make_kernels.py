"""Write one Kaggle kernel per detector for the Phase 2 CV runs.

    python cpu_repro/cv/kaggle/make_kernels.py --commit <git sha> --out <dir> [--folds 0 1 2 3 4] [--cv-seed 1]

Each kernel clones this repo from GitHub at the given commit (code and the
uncropped X-rays and masks, which git tracks in Dataset/bb_u_net_dataset),
checks for a usable GPU, and runs cpu_repro/cv/train_cv.py for its detector under a
watchdog, so finished folds are saved before Kaggle's 12 h session limit.
Push a generated directory with: kaggle kernels push -p <dir>/<kernel>
"""
import argparse
import json
from pathlib import Path

USER = "christopherhuang88"
REPO_URL = "https://github.com/christopherh-88/tooth-numbering-priors.git"
ULTRALYTICS = "ultralytics==8.4.143"  # same version as cpu_repro/requirements.txt
WATCHDOG_SECONDS = 11 * 3600

SCRIPT = '''"""Phase 2 CV kernel: {detector}, CV seed {cv_seed}, folds {folds}, repo commit {commit}."""
import shutil, subprocess, sys, time
from pathlib import Path

COMMIT = "{commit}"
DETECTOR = "{detector}"
FOLDS = {folds}
REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/cv_out")


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


import torch
if not torch.cuda.is_available():
    sys.exit("FATAL: no GPU")
a = torch.randn(1024, 1024, device="cuda")
(a @ a).sum().item()
print("GPU:", torch.cuda.get_device_name(0), flush=True)

sh([sys.executable, "-m", "pip", "install", "--quiet", "{ultralytics}"])
if REPO.exists():
    shutil.rmtree(REPO)
sh(["git", "clone", "--quiet", "{repo_url}", str(REPO)])
sh(["git", "-C", str(REPO), "checkout", "--quiet", COMMIT])
n = len(list((REPO / "Dataset/bb_u_net_dataset/panoramic_x_rays").glob("*.jpg")))
assert n == 425, n

import ultralytics, torchvision
print("versions: torch", torch.__version__, "torchvision", torchvision.__version__,
      "ultralytics", ultralytics.__version__, flush=True)

t0 = time.time()
cmd = [sys.executable, str(REPO / "cpu_repro/cv/train_cv.py"), "--detector", DETECTOR,
       "--out-root", str(OUT), "--work", "/kaggle/working/prepared", "--device", "0",
       "--folds", *map(str, FOLDS), "--cv-seed", "{cv_seed}"]
try:
    sh(cmd, cwd=str(REPO), timeout={watchdog})
    status = "finished"
except subprocess.TimeoutExpired:
    status = "WATCHDOG: stopped after {watchdog} s; folds with test_detections.csv are complete"
except subprocess.CalledProcessError as e:
    status = f"FAILED with exit code {{e.returncode}}"
print(status, f"({{(time.time() - t0) / 3600:.2f}} h)", flush=True)
(OUT / "STATUS.txt").parent.mkdir(parents=True, exist_ok=True)
(OUT / "STATUS.txt").write_text(f"{{status}}\\ncommit {{COMMIT}}\\n")
# Keep the outputs small: drop Ultralytics' per-epoch extras, keep best.pt.
for p in OUT.rglob("weights/epoch*.pt"):
    p.unlink()
for p in OUT.rglob("weights/last.pt"):
    p.unlink()
for p in OUT.rglob("last.pt"):
    p.unlink()
if status != "finished":
    sys.exit(1)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--detectors", nargs="+", default=["yolov8x", "rtdetr_l", "fasterrcnn"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--cv-seed", type=int, default=0)
    a = ap.parse_args()
    for det in a.detectors:
        name = f"tooth-numbering-cv-{det.replace('_', '-')}-s{a.cv_seed}{a.suffix}"
        d = a.out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.py").write_text(SCRIPT.format(
            detector=det, folds=a.folds, commit=a.commit, cv_seed=a.cv_seed, ultralytics=ULTRALYTICS,
            repo_url=REPO_URL, watchdog=WATCHDOG_SECONDS))
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{USER}/{name}", "title": name, "code_file": f"{name}.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
            "model_sources": [], "machine_shape": "NvidiaTeslaT4"}, indent=2))
        print("wrote", d)


if __name__ == "__main__":
    main()
