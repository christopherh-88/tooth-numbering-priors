# Environment setup

For anyone setting this up cold on a different machine (this was written
so Kavish doesn't have to re-derive it). Two things about this project's
dependencies are non-obvious enough to cause real confusion if you hit them
without warning: it needs **two deep learning frameworks at once**, and
there's a **numpy version conflict** between them that isn't visible until
you try to install everything together.

## Two frameworks, on purpose, not a mistake

This project trains/runs both:
- **TensorFlow/Keras**: for the U-Net segmentation model
  (`notebooks/Unet/`, `cpu_repro/train_unet_cpu.py`).
- **PyTorch, via Ultralytics**: for YOLOv8 detection/numbering
  (`notebooks/yolov8/`, `cpu_repro/yolo_training/`).

If you only look at `yolov8_train.ipynb` you'd reasonably assume this is a
PyTorch project and be confused when `unet_training.ipynb` imports
`tensorflow`. It's both, in the same environment, at the same time. This
was verified to actually work (see below), not something to "fix" by
picking one framework.

## The numpy conflict

TensorFlow 2.16.2 (the version this project uses) requires `numpy<2.0`.
Recent releases of `opencv-python`/`opencv-python-headless`, `tifffile`,
and `imagecodecs` all require `numpy>=2` in their *latest* versions. If you
install packages one at a time without pinning numpy first (which is the
natural thing to do), pip will happily install a numpy 2.x release when it
gets to `opencv-python-headless` or `tifffile`, silently breaking
TensorFlow, and the failure won't show up until the next time you `import
tensorflow` (a confusing `AttributeError` deep in a scipy/keras import
chain, not an "incompatible numpy" message).

The fix is: pin numpy **and** older, numpy-1.x-compatible releases of
`opencv-python`/`opencv-python-headless`, `tifffile`, and `imagecodecs` all together in one
`requirements.txt`, so pip's resolver sees every constraint at once instead
of hitting them one at a time. This is already done in
`cpu_repro/requirements.txt`, installing that file in one `pip install -r`
call resolves cleanly. **Verified from a completely fresh venv, one shot,
no manual fixes, no `pip check` warnings**. See "What was actually
tested" below.

## Setup (verified working sequence)

Requires **Python 3.11 or 3.12**, not 3.13 or later. TensorFlow does not
yet support Python 3.13+; a newer system Python will fail confusingly deep
into the TensorFlow install, not with a clear "unsupported Python version"
message. Check what you have available:

```bash
python3.12 --version   # or python3.11
# on macOS, if you don't have one: brew install python@3.12
```

Then, from the repo root:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate        # Windows: .venv312\Scripts\activate
python -m pip install --upgrade pip

# 1. torch/torchvision FIRST, from PyTorch's own CPU-only wheel index, not
#    plain PyPI (on Linux, PyPI's torch is the CUDA build: enormous and
#    pointless on a CPU-only machine). `-c` holds numpy/pillow and the
#    other dependencies to the same pins as step 2.
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0 -c cpu_repro/requirements.txt

# 2. Everything else, in one shot. The torch/torchvision pins in this file
#    are already satisfied by step 1, so pip keeps the CPU builds. Every
#    version is pinned in this one file and pip's resolver sees all the
#    constraints together. Don't install these packages one at a time from
#    memory. Use this file.
pip install -r cpu_repro/requirements.txt
```

Verify it worked:

```bash
python -c "
import numpy, tensorflow as tf, torch, ultralytics, cv2
print('numpy', numpy.__version__)      # expect 1.26.4
print('tensorflow', tf.__version__)    # expect 2.16.2
print('torch', torch.__version__)      # expect 2.14.0 (2.14.0+cpu on Linux)
print('cuda built', torch.backends.cuda.is_built())  # expect False
print('ultralytics', ultralytics.__version__)        # expect 8.4.143
print('cv2', cv2.__version__)          # expect 4.10.0
"
pip check   # expect: "No broken requirements found." (see note below)
```

## What was actually tested

The two-command sequence above was run from a **brand-new venv** on
2026-09-27 and produced zero dependency-resolver warnings, a clean
`pip check`, `torch.backends.cuda.is_built() == False` and cv2 4.10.0.
Tested on macOS 14, Apple Silicon (arm64), Python 3.12.13. This is not
the original messy incremental install used while building this project
(which hit the numpy conflict live and fixed it step by step; that
trial-and-error is NOT what's documented here).

**Linux x86_64 (Ubuntu 24.04, Python 3.12.14)** is tested on every change
to the install files by `.github/workflows/linux-repro.yml` on a
GitHub-hosted runner: the same two commands, `pip check`, the versions
above (`2.14.0+cpu` / `0.29.0+cpu` builds), no CUDA packages, and a rerun
of the core analyses against the committed outputs. The first run passed
the install checks; the only output difference was last-digit drift in
the logistic-regression baseline (RESULTS.md Section 57). If you hit
something different on Linux, update this file rather than working
around it silently.

The order was reversed on 2026-09-27 (RESULTS.md Section 55). The earlier
order (this file first, then torch from the CPU index; tested on
2026-09-07) pulled torch from plain PyPI in the first step. That is
harmless on macOS but gets the CUDA build on Linux, and the CPU-index
step afterwards would not replace it, because the installed torch
already satisfies `torch==2.14.0`.

One OpenCV note: Ultralytics' package metadata requires `opencv-python`,
so a fresh install gets it alongside `opencv-python-headless`. Both
provide the same `cv2` module, so `requirements.txt` pins both to
4.10.0.84. An older venv built with only the headless package (like the
original development one) will show `ultralytics 8.4.143 requires
opencv-python, which is not installed` in `pip check`; that one is
harmless. Anything else in `pip check` is a real problem.

## GPU / Kaggle note

Everything above is the CPU-only setup used for local development. The
YOLOv8 training run in `cpu_repro/yolo_training/` is meant to run on a
Kaggle GPU session, `torch`/`ultralytics` will use the GPU automatically
there without any environment changes beyond what Kaggle provides; you do
not need a different requirements file for that, just don't reinstall the
CPU-only torch wheel above on a GPU machine (install plain `torch` there,
or let Kaggle's preinstalled torch stand. Check
`cpu_repro/yolo_training/README.md` before touching the environment on
Kaggle, since Kaggle notebooks usually come with torch/CUDA preinstalled
and reinstalling can be slower and riskier than just using what's there).

## If you hit a dependency error anyway

Don't hand-fix it package by package. That's exactly how the original
numpy conflict happened during development. Instead: check whether
`cpu_repro/requirements.txt` is out of date (a newer TensorFlow/Ultralytics
release may have shifted the numpy compatibility window again), fix the
pins there, verify with a fresh venv the way this file was verified, and
update the "What was actually tested" section above with the new date and
versions, not just silently move on.
