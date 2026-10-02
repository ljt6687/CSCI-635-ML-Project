# Squirrel detection · RF-DETR Medium

The complete workflow is in [squirrel_detection_rfdetr_medium.ipynb](squirrel_detection_rfdetr_medium.ipynb): Roboflow v6 download, dataset audit and EDA, ClearML tracking, memory probe, training, validation/test reports, and verified best-weight export.

## Multiclass v2 dataset audit and training

The 11-class v2 workflow uses ten named species plus `generic_squirrel`. Its
source dataset is `data/squirrel-v2-clean`; that directory is never changed by
the audit or exporter. Complete the audit and review its evidence before
creating the versioned training export:

```bash
# Review the completed audit and its evidence first.
cat output/v2_annotation/audit_v2/report.md
uv run python scripts/export_squirrel_v2_audited_coco.py
uv run jupyter lab squirrel_detection_rfdetr_medium_v2.ipynb
```

The exporter stops unless the reviewed `decisions.json` has passing QA and no
unresolved cases. To start a fresh audit, use a new output directory, such as
`uv run python scripts/audit_squirrel_v2_purity.py --embeddings --output
output/v2_annotation/audit_v2_refresh`. A full audit refuses to overwrite an
existing decision file. Continuation flags such as `--embeddings-only` update
specific analyses while preserving reviewed decisions. Point the exporter at
the reviewed new decision file with `--decisions` and use a new `--destination`
for another immutable export.

The exporter requires `decisions.json` to match the source annotation
fingerprint, contain no unresolved cases, and record a passing quality audit.
It excludes unusable images and evidence-backed bad annotations, removes exact duplicate images, keeps observation
and pHash Hamming-distance-4 perceptual-risk groups within one split, and writes resized,
orientation-corrected RGB images and COCO labels to
`data/squirrel-v2-audited-coco/`. Review findings remain under
`output/v2_annotation/audit_v2/`; the export records every removal and move in
`decisions_applied.json` and its counts in `export_summary.json`.

The v2 notebook requires that audited export and checks its manifest and
annotations before training. Its train loader uses moderate class-aware
sampling; validation and test retain their natural distributions. The notebook
does not build the dataset implicitly. Running the notebook's training cell
starts the full 50-epoch experiment, so run its audit and memory-probe cells
first when validating a new export.

## Setup

```bash
uv sync --locked
uv run jupyter lab squirrel_detection_rfdetr_medium.ipynb
```

Use the project `.venv` kernel. Python 3.13 and an NVIDIA CUDA GPU are required. The checked-in lockfile pins the tested environment. Fill the following variables privately in `.env` (never put values in the notebook):

- `ROBOFLOW_KEY` (or `ROBOFLOW_API_KEY`)
- `CLEARML_API_ACCESS_KEY`
- `CLEARML_API_SECRET_KEY`
- Optional: `CLEARML_API_HOST`, `CLEARML_WEB_HOST`, `CLEARML_FILES_HOST` (hosted ClearML defaults)

Run cells in order. The first execution downloads the specified COCO dataset. The audit stops training on confirmed cross-split duplicates or unresolved perceptual matches; inspect the generated review gallery and record false-positive reasons in the configuration cell. Confirmed overlap cannot be waived. Dataset source images and splits are never silently modified.

## Experiment

- One `SQUIRREL` class, bounding boxes and confidence scores; no species/individual identification.
- RF-DETR Medium, 576 pixels, mixed precision, gradient checkpointing, seed 42.
- Up to 50 epochs; batch × accumulation probes: `8×2`, `4×4`, `4×2`.
- Best checkpoint selected by validation mAP@0.50:0.95; confidence threshold selected by validation F1, then frozen for test.
- Detection AP/AR, confusion matrix, precision/recall/F1 and PR curves; image-presence ROC/AUC only when both positive and negative images exist.

Generated files are ignored by Git:

- `data/squirrel-v6-coco/`: original export and completion manifest.
- `output/runs/<run>/`: config, logs, resumable `last.ckpt`, best weights and reports.
- `output/runs/<run>/reports/report.html`: EDA/final overview; `valid/` and `test/` contain separate evaluation reports.
- `output/model_weights/best_squirrel_rfdetr_medium.pth`: verified best export, accompanied by metadata and model card.

For recovery, set `RESUME_RUN` in the first notebook cell to the original run folder. Keep its configuration and batch selection unchanged; the full `last.ckpt` restores training state within the 50-epoch ceiling. A `.pth` best checkpoint is intended for inference, not full-state recovery.

The notebook disables automatic ClearML source/environment/console capture and sends only explicit artifacts and sanitized metrics/logs. It retains local logs and pending upload events if tracking fails. Do not commit `.env`, raw outputs, credentials, or signed export URLs.

Dataset: [Root and Nut, Squirrel-Re-ID-Training-V1 v6](https://universe.roboflow.com/root-and-nut/squirrel-re-id-training-v1-fzpbr/dataset/6), CC BY 4.0. Results must include the split/leakage limitations; this workflow does not promise a predetermined accuracy.

## YOLO multiclass baselines and comparison

Use the same immutable audited v2 COCO export with the two medium detection
models. These are bounding-box detectors with eleven species classes, rather
than image-only classifiers:

- [YOLO26 Medium](squirrel_detection_yolo26_medium_v2.ipynb): pretrained
  `yolo26m.pt`, with separate accuracy-oriented and NMS-free head reports.
- [Original YOLOv5 Medium](squirrel_detection_yolov5_medium_v2.ipynb): pretrained
  anchor-based `yolov5m.pt` using a pinned original repository revision, rather
  than the newer YOLOv5u architecture.
- [Three-model comparison](squirrel_detection_model_comparison_v2.ipynb): run
  after completing training and evaluation in both YOLO notebooks.

```bash
uv sync --locked --all-extras
uv run --all-extras jupyter lab squirrel_detection_yolo26_medium_v2.ipynb
```

Select the project `.venv` kernel. Run preparation and GPU-probe cells first;
**the training cell starts up to 200 epochs with early-stopping patience 30**.
Both baselines use 640 pixels, seed 42, AMP, standard YOLO sampling, and their
native optimizer/augmentation recipes. YOLO26's native checkpoint criterion
is validation mAP50–95; original YOLOv5 uses `0.1 * AP50 + 0.9 * mAP50–95`.
A small batch probe selects 16, 8, or 4 without carrying its weights into the
full run. Resolved native settings and optimizer choices are saved.

The shared exporter converts COCO category IDs `1..11` to YOLO labels `0..10`
without changing images, classes, annotations, or split membership. It copies
image bytes into `data/squirrel-v2-audited-yolo/<fingerprint>/`, validates boxes,
and records image/label hashes. Reuse verifies those hashes and the current
source images. Invalid boxes, stale annotations, or exact cross-split duplicate
images stop preparation. The source COCO dataset is never modified.

Each model owns its timestamped artifacts:

- `output_yolo26/runs/<run>/` and `output_yolov5/runs/<run>/`: configuration,
  audit manifest, environment, sanitized logs, GPU-probe profiler traces,
  native training outputs, and epoch/resource history.
- `native/weights/`: native best/latest, periodic checkpoints, and the
  unstripped `last_resumable.pt`. Set `RESUME_RUN` to this run folder to recover
  an interrupted experiment; configuration and dataset identity must match.
- `reports/valid/` and `reports/test/`: common COCO AP/AR, per-class
  precision/recall/F1 and AP, PR/confidence curves, background-aware raw and
  normalized confusion matrices, and annotated examples/error galleries.
- `reports/end2end/`: YOLO26's secondary head reports, using an independently
  loaded copy of the same checkpoint and its own validation-selected threshold.
- `benchmarks/<head>/`: local CPU/GPU batch-one FP32 latency, individual image
  timings, hardware, and timing protocol. Image decoding is excluded;
  preprocessing, inference, and postprocessing are included.
- `weights/`: verified native best weights, model card, class mapping, hashes,
  and frozen confidence thresholds. `reports.zip` packages the evaluation
  reports. Native plots and TensorBoard events also remain in the run folder.

Threshold selection uses validation micro F1 at matching IoU 0.50, ties choose
highest confidence, and the threshold/checkpoint are frozen for held-out test.
Final comparisons use pycocotools with maxDets 100 and prediction floor 0.0001.
Native training curves can use different AP implementations and loss scales;
final metrics use the shared evaluator. Optional ClearML uploads are explicit
and sanitized; tracking failure leaves local artifacts and a pending-event file.

The comparison defaults to RF-DETR run `20260926T153802Z-v2-f35e9e` and the
newest compatible completed YOLO runs. Explicit run paths can be set in its
configuration cell. It verifies current pixels against RF-DETR's saved image inventory, reproduces
its archived accuracy metrics, measures
fresh latency for all models, and saves tables/figures under
`output_model_comparison/runs/<comparison>/`. The archived RF-DETR predictions
include low-confidence output index 11 outside the eleven-class taxonomy.
These are preserved in an audit file and excluded from expanded reports only
when below its frozen threshold; original COCO and operating scores must match.

Interpret this as a comparison of model recipes: RF-DETR used 576 pixels,
class-aware sampling, and up to 50 epochs, while YOLO uses 640 pixels, natural
sampling, and up to 200 epochs. Pretraining, model capacity, and postprocessing
also differ. Report accuracy and latency leaders separately. Local CPU timings
do not establish performance on a particular edge device.

To regenerate clean notebooks or run bounded verification:

```bash
uv run --all-extras python scripts/create_baseline_notebooks.py
uv run --all-extras python -m unittest discover -s tests -p 'test_yolo_baselines.py' -v
uv run --all-extras python scripts/smoke_yolo_baselines.py --check-resume
```

The smoke workflow uses only 22 training and 11 validation/test images, two
training epochs, and short benchmarks. Smoke runs are separate from production
baselines and cannot enter the default comparison. Preparation downloads the
pinned original YOLOv5 source and pretrained assets when absent; it does not
run the full experiments or modify the existing RF-DETR notebook.


## Dataset download and setup

To skip running the full audit and export pipeline from scratch, download the pre-built, audited multiclass v2 COCO export archive:

- **Google Drive link:** [`squirrel-v2-audited-coco.zip` (Google Drive)](https://drive.google.com/file/d/1WWIY89soGJR2hDTC6zQyS0NiTU_e5AgV/view?usp=sharing)
- **Archive size:** ~2.9 GB compressed (~3.0 GB uncompressed)
- **Format:** COCO detection format (RGB images + JSON annotations)
- **Classes (11):** 10 squirrel species (`Callosciurus erythraeus`, `Sciurus aureogaster`, `Sciurus carolinensis`, `Sciurus granatensis`, `Sciurus griseus`, `Sciurus lis`, `Sciurus niger`, `Sciurus vulgaris`, `Tamiasciurus douglasii`, `Tamiasciurus hudsonicus`) plus `generic_squirrel`.
- **Splits:** 8,554 images and 9,109 bounding-box annotations across `train` (5,988), `valid` (1,282), and `test` (1,284).
- **Leakage mitigation:** Identical observations and pHash perceptual risk groups (Hamming distance $\le 4$) are grouped into single splits; false positives (e.g., tracks, domestic cats, raccoons, stationary branches) and exact pixel duplicates have been removed. Includes the `.complete.json` manifest required by the training notebooks.

### Guided installation

#### Option 1: Command line (`gdown`)

You can download and extract directly using `gdown` via `uv`:

```bash
mkdir -p data
uv run --with gdown gdown "https://drive.google.com/uc?id=1WWIY89soGJR2hDTC6zQyS0NiTU_e5AgV" -O data/squirrel-v2-audited-coco.zip
unzip -q data/squirrel-v2-audited-coco.zip -d data/
```

#### Option 2: Manual download

1. Download [`squirrel-v2-audited-coco.zip`](https://drive.google.com/file/d/1WWIY89soGJR2hDTC6zQyS0NiTU_e5AgV/view?usp=sharing) from Google Drive.
2. Move the downloaded file into the `data/` folder in the project root (`data/squirrel-v2-audited-coco.zip`).
3. Extract the archive into `data/`:

```bash
mkdir -p data
unzip -q data/squirrel-v2-audited-coco.zip -d data/
```

### Verification

Ensure the extracted directory structure matches the layout expected by the notebooks:

```text
data/squirrel-v2-audited-coco/
├── .complete.json
├── decisions_applied.json
├── export_summary.json
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg
```

Run a quick check:

```bash
# Verify the presence of splits and the completion manifest
ls -la data/squirrel-v2-audited-coco/
```

Once extracted, you can directly run [`squirrel_detection_rfdetr_medium_v2.ipynb`](squirrel_detection_rfdetr_medium_v2.ipynb), [`squirrel_detection_yolo26_medium_v2.ipynb`](squirrel_detection_yolo26_medium_v2.ipynb), or [`squirrel_detection_yolov5_medium_v2.ipynb`](squirrel_detection_yolov5_medium_v2.ipynb).
