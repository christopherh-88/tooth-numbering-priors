# CPU repro of unet_training.ipynb

Verified working on macOS 14, arm64, Python 3.12.13, CPU only. 3 epochs on
24 images at 256x256 took ~4.5 minutes total (~85s/epoch) on an M-series
Mac. Scale `SUBSET_SIZE`/`EPOCHS`/`IMG_SIZE` in `train_unet_cpu.py` up once
you have GPU quota again.

## Setup

```bash
cd cpu_repro
python3.12 -m venv .venv        # needs Python <=3.12; TensorFlow doesn't support 3.13/3.14 yet
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0
```

torch is installed separately from the CPU wheel index so you don't
accidentally pull a CUDA build (which is enormous and useless on a CPU-only
machine).

## Run

```bash
python train_unet_cpu.py
```

Edit the CONFIG block at the top of `train_unet_cpu.py` to scale up
(`SUBSET_SIZE`, `EPOCHS`, `IMG_SIZE`, `BATCH_SIZE`).

## Data

The dataset is already present in this repo at `Dataset/bb_u_net_dataset/`
(panoramic X-rays + per-tooth label export tiffs) and
`Dataset/yolo_train_dataset/` (YOLO-format detection labels) - nothing to
download for this script. If you ever need to re-download UFBA-425 from
scratch, get it from Figshare (linked in the repo's top-level README.md)
and match the existing `Dataset/` layout.

## Known deviations from the original notebook (see train_unet_cpu.py docstring for why)

- Bounding-box prior channels are zero-filled (they're unused by the plain
  U-Net's forward pass anyway).
- Ground-truth masks are built directly from the per-tooth export tiffs
  instead of requiring you to first run `notebooks/Data_gen/2ddatagen.ipynb`
  to produce combined per-image tiffs (the repo doesn't ship those).
- `IMG_SIZE` defaults to 256, not 512, for CPU epoch time.

None of this is expected to reproduce the README's reported Dice scores -
that needs the full 425-image dataset and real YOLO-generated bbox priors
(for OralBBNet) or many more epochs (for plain U-Net), on a GPU.
