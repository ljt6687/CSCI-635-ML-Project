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
