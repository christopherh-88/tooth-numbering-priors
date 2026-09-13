# YOLOv8 training - run, results in RESULTS.md Sections 21/26/31/33

**Status (2026-09-12): this has been run for real, 7 times total** - seed
0 (`train_yolo.py`, `RESULTS.md` Section 21), seeds 1-4
(`train_yolo_seed{1,2,3,4}.py`, Section 33), and a zero-jitter mitigation
variant (`train_yolo_zerojitter.py`, Sections 30/31) - all on Kaggle GPU
(Tesla T4 x2; see "Hardware and environment actually used" below). The
rest of this file was originally written before any GPU run happened and
is kept below because the mechanics (split handling, resume, evaluation
protocol) are still accurate - only the "not yet run" framing was stale.

It reads the coordinate baseline's persisted image-level split, resumes
cleanly across Kaggle's 12-hour session limit, and evaluates with the
exact same metrics as `cpu_repro/coord_baseline` so the two are directly
comparable.

## Reproducing every YOLO-derived table/figure, in order

1. `cd cpu_repro/coord_baseline && python export_split.py <seed>` for
   each seed in `{0,1,2,3,4}` (seed 0 also has no-argument backward
   compatibility) - produces `image_split_seed{N}.csv`/`.meta.json`.
   Already committed in this repo; only rerun if you need to regenerate
   from scratch.
2. On a CUDA GPU (a real one - see the hardware note below, CPU-only
   will not finish in a reasonable time): run `train_yolo.py` (seed 0,
   jittered - the base config) and `train_yolo_seed{1,2,3,4}.py` (seeds
   1-4, same jitter config) to reproduce Sections 21/33. Run
   `train_yolo_zerojitter.py` (translate=0, scale=0) to reproduce
   Sections 30/31's mitigation arm. Each writes its own
   `eval_results/<run>/summary.csv` and
   `runs/<run_name>/weights/best.pt`.
3. CPU-only analysis scripts, run after the weights above exist:
   - `python case_study_yolo_vs_coord.py` /
     `error_correlation_analysis.py` /
     `robustness_analysis.py` - seed-0-only breakdowns (Sections 22,
     26, 27).
   - `python mitigation_analysis.py` - paired jittered-vs-zerojitter
     comparison (Section 31). Requires both `train_yolo.py`'s and
     `train_yolo_zerojitter.py`'s weights to exist first.
   - `python multiseed_analysis.py` - 5-seed replication summary
     (Section 33's main table); verifies its own seed-0 output against
     the already-published Section 21/26/31 numbers before trusting
     seeds 1-4, and exits non-zero if that check fails. Requires all 5
     seeds' weights to exist first.
   - `python -c "from multiseed_analysis import run_breakdowns; run_breakdowns()"`
     (there is no CLI entry point for this - `__main__` only calls
     `main()`) - 5-seed error-taxonomy and per-class breakdowns
     (Section 33's "deferred per-seed breakdowns" update). Uses
     `build_merged()`'s on-disk cache
     (`eval_results/multiseed/joined_seed{0-4}.csv`) so it does not
     re-run CPU inference if that cache already exists from the
     previous step (i.e. run `main()` first, or `run_breakdowns()` will
     build the cache itself on first use).

## Hardware and environment actually used

All GPU training ran on **Kaggle notebooks, GPU T4 x2 accelerator**
(selected manually in the Kaggle web UI Settings panel - the Kaggle API's
`enable_gpu: true` flag repeatedly and unreliably assigned a Tesla P100
instead, which this project's torch build cannot use - see `RESULTS.md`
Sections 21/31/33 for the repeated fail-fast/retry log). Package versions
on the Kaggle side were Kaggle's own current GPU-notebook image at the
time each kernel ran (not independently pinned by this repo) - **this is
a real reproducibility gap**: the exact `torch`/`ultralytics` build
Kaggle provisions can drift over time, and this repo does not pin or
snapshot it. `cpu_repro/requirements.txt` pins the CPU-only environment
used for every analysis/evaluation script above, but explicitly does
*not* cover the GPU training environment. If exact reproduction matters,
pin `torch`/`ultralytics`/`opencv` versions in a Kaggle
`kernel-metadata.json` before re-running, and record the resulting
versions in `RESULTS.md` alongside the run.

A small `ensure_gpu()` fail-fast helper (a real CUDA matmul check, not
just `torch.cuda.is_available()`) was used as a wrapper around each
Kaggle kernel invocation during this project, to abort in ~10 seconds
rather than burn GPU-hours if Kaggle assigned an incompatible GPU. That
wrapper was Kaggle-kernel-launch scaffolding, not part of the analysis
pipeline itself, and was not committed to this repo - it is not needed
to run `train_yolo.py` directly on a correctly-configured GPU machine,
only to guard against Kaggle's specific flaky-accelerator-assignment
failure mode.

What has been verified without a GPU (see "What was and wasn't tested"
below): the split file is generated and correct, the IoU-matching/scoring
logic is unit-tested with synthetic data, and `prepare_yolo_dataset()` has
been run for real against the actual dataset (820 train / 201 val images
correctly assigned - see below for the one expected excluded image).

## Same split as the coordinate baseline - not regenerated

`cpu_repro/coord_baseline/export_split.py` was run once to persist the
coordinate baseline's own `grouped_split(df, seed=0, TEST_FRACTION=0.2)` to
`cpu_repro/coord_baseline/image_split_seed0.csv` (image_id -> train/test,
340 train images / 85 test images - matches build_coord_baseline.py's own
seed-0 printout exactly). Seed 0 is also the seed the coordinate baseline
uses for its saved confusion matrices, so it's the natural one to compare a
real trained model against.

`train_yolo.py` **reads that file** (`load_split()`) rather than
recomputing anything - it never calls `grouped_split` or touches
`coord_baseline`'s randomness. If you ever need a different split, rerun
`export_split.py` with a different seed and re-point `SPLIT_FILE`, or edit
`SEEDS[0]` in `build_coord_baseline.py`, but don't hand-roll a second split
generator - there should be exactly one source of truth for "the split."

`prepare_yolo_dataset()` turns that file into what YOLO actually needs:
`prepared/train.txt` / `prepared/val.txt` (absolute paths into the existing
`Dataset/yolo_train_dataset/*/images/`, nothing copied) and `prepared/
data.yaml` (`nc: 32`, names in the exact `FDI_CODES` order imported from
`build_coord_baseline.py`, matching the dataset's own `data.yaml`). This
regenerates every run (cheap, deterministic from the split file) so it's
always in sync - the previous run's `prepared/` directory doesn't need to be
carried over between machines.

One image is expected to be skipped: `cate5-00013` has exactly one label
file on disk (`Dataset/yolo_train_dataset/test/labels/cate5-00013_jpg.rf.
....txt`) and it is completely empty (0 annotated teeth) - verified by hand.
It never appears in the coordinate baseline's data either (its
`load_instances()` only counts label files that contribute at least one
row), so it correctly has no entry in the split file and gets a one-line
`WARNING: 1 label files had an image_id not present in the split file`
during prep. This is expected, not a bug - there's nothing to train or
evaluate on in an empty annotation file.

## Checkpoint-and-resume for the 12-hour Kaggle limit

Ultralytics saves `weights/last.pt` (and `weights/best.pt`) after every
epoch by default; `SAVE_PERIOD = 5` additionally keeps numbered snapshots
(`epoch5.pt`, `epoch10.pt`, ...) as a safety net in case `last.pt` is ever
left mid-write by a hard kill. There is no reliable way to intercept a
Kaggle VM being cut off mid-session from inside the notebook, so no signal
handling is attempted - the recovery story is simply:

```bash
python train_yolo.py
```

run again, unchanged. `get_or_create_model()` checks for
`runs/yolov8_seed0split/weights/last.pt`; if it's there, it loads it and
calls `model.train(resume=True)`, which reloads the original epoch count,
batch size, etc. from that run's own `args.yaml` and continues from the
next epoch. If `epochs` in `train_yolo.py` was already reached before the
session ended, `train()` still calls `evaluate()` immediately using the
existing `best.pt` - it isn't gated on a fresh training call succeeding.

Don't change the hyperparameter constants (`EPOCHS`, `BATCH`, `IMGSZ`, etc.)
between a partial run and its resume - ultralytics resumes using the
*original* run's saved args, not whatever is currently in this file, so
edits made mid-run won't take effect until you start a new `RUN_NAME`.

## Evaluation, in the same format as the coordinate baseline

After training, `evaluate(best.pt)`:

1. Runs `model.predict()` on every image in the val split (`conf=0.5,
   iou=0.7`, matching the `conf`/`iou` values already used for inference in
   `notebooks/yolov8/yolo_test.ipynb` and the `yolov8+unet` notebooks).
2. Greedily IoU-matches predicted boxes to ground-truth boxes per image
   (`MATCH_IOU_THRESHOLD = 0.5`) - this is a different threshold/purpose
   than the NMS `iou=0.7` above: NMS iou controls how ultralytics
   de-duplicates its own overlapping predictions before we ever see them;
   match IoU controls whether a surviving prediction counts as "the same
   tooth" as a ground-truth box.
3. Calls `build_coord_baseline.evaluate(y_true, y_pred, train_y)` - **the
   exact same function** the coordinate baseline uses - on the matched
   pairs, so `top1_acc`, `quadrant_acc`, `tooth_type_acc`,
   `majority_baseline_acc`, `mirror_quadrant_error_frac`, and
   `neighbor_error_frac` are computed identically and land in the same
   columns as `cpu_repro/coord_baseline/per_seed_results.csv` (with
   `classifier="yolov8"`, `seed=0`). You can concatenate the two CSVs
   directly to compare.
4. Also saves a 32x32 confusion matrix (`confusion_matrix_yolov8_seed0.csv`
   / `.png`) in the same format/labeling as the coordinate baseline's.

### The one thing that is NOT like-for-like, and why

The coordinate baseline is only ever asked "given this ground-truth box,
what tooth is it" - it's never wrong about *where* the box is, because the
box is given. YOLOv8 has to find the boxes too. So `top1_acc` etc. here are
computed **only over matched detections** - unmatched ground truth (missed
detections) and unmatched predictions (spurious detections) are excluded
from the accuracy numbers and reported separately as `detection_recall`,
`n_unmatched_gt_missed_detections`, `n_unmatched_pred_spurious_detections`.
Read `top1_acc` here as "conditional on YOLO finding the tooth at all, is
the number right" - directly comparable to the coordinate baseline's
question - and read `detection_recall` alongside it as the (separate)
answer to "how often does it find the tooth in the first place." A model
that's very accurate on matched detections but has low recall is not
strictly better than the coordinate baseline at the overall task, even
though its `top1_acc` number alone would suggest so.

## What was and wasn't tested (no GPU available)

Verified without a GPU or a trained model:
- `export_split.py` was run for real; `image_split_seed0.csv` matches
  `build_coord_baseline.py`'s own seed-0 numbers exactly (340/85 images,
  22072/5491 instances).
- `prepare_yolo_dataset()` was run for real against the actual dataset:
  correctly produced 820 train / 201 val image paths and a valid
  `data.yaml`, with the one expected empty-label exclusion above and no
  others.
- `iou_xyxy()` and `match_boxes()` (the core detection-matching logic) are
  unit-tested with synthetic boxes: exact overlap, partial overlap, zero
  overlap, a wrong-class-but-correct-box case, and a completely missed/
  spurious detection case - all produced the expected matches/scores.
- `xywhn_to_xyxyn()` and `base_image_id()` are unit-tested against known
  inputs/outputs.
- Ultralytics API surface used here (`train()`'s kwargs incl. `dropout`,
  `cos_lr`, `close_mosaic`, `single_cls`, `warmup_epochs`, `lrf`, `resume`,
  `save_period`, `project`, `name`, `exist_ok`; `Boxes.xyxyn`/`.cls`/
  `.conf`) was confirmed to exist in the installed ultralytics 8.4.143 by
  introspecting `ultralytics.cfg.get_cfg()` and
  `ultralytics.engine.results.Boxes` directly.

NOT tested (needs a GPU, or at minimum a real training run to produce a
checkpoint): the actual `model.train()` call end-to-end, the
resume-from-checkpoint path with a real `last.pt`, and `evaluate()`'s
`model.predict()` call against real detections. Do a short smoke test
before trusting a long run - e.g. temporarily set `BASE_WEIGHTS =
"yolov8n.pt"`, `EPOCHS = 1`, `DEVICE = 0` (or `"cpu"` if desperate) and run
once to confirm the full pipeline executes on your actual Kaggle
environment, then restore the real config and start the full run.

## Config

Everything you're likely to change lives at the top of `train_yolo.py`:
`BASE_WEIGHTS`, `EPOCHS`, `BATCH`, `IMGSZ`, `DEVICE`, `DROPOUT`,
`CLOSE_MOSAIC`, `COS_LR`, `WARMUP_EPOCHS`, `LRF`, `SINGLE_CLS`,
`SAVE_PERIOD`, `EVAL_CONF`, `EVAL_NMS_IOU`, `MATCH_IOU_THRESHOLD`. The
training hyperparameters default to matching
`notebooks/yolov8/yolov8_train.ipynb`'s CLI call as closely as possible
(`yolov8x.pt`, 30 epochs, batch 10, imgsz 640, dropout 0.6, cos_lr,
warmup_epochs 10, lrf 0.005) so this run is comparable to the repo's
original YOLOv8 training, just on the fixed, coordinate-baseline-matching
split instead of the original Roboflow train/valid/test boundary.

## Run (on Kaggle, once GPU quota is available)

```bash
cd cpu_repro/yolo_training
python train_yolo.py
```

If the session gets killed partway through, just run the same command
again - see "Checkpoint-and-resume" above.

## Files

- `train_yolo.py` - the script (prepare data -> train with resume -> evaluate).
- `prepared/` - generated each run: `train.txt`, `val.txt`, `data.yaml`.
  Not committed (paths are machine-specific).
- `runs/yolov8_seed0split/weights/{last,best}.pt` - checkpoints. Not
  committed (large binaries).
- `eval_results/summary.csv`, `confusion_matrix_yolov8_seed0.csv`/`.png` -
  written after training completes, in the same format as
  `cpu_repro/coord_baseline`'s outputs.
