"""Reproducibly generate clean runnable YOLO and model-comparison notebooks."""
from pathlib import Path
import hashlib
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip(), id=hashlib.sha256(text.encode()).hexdigest()[:12])


def code(text):
    return nbf.v4.new_code_cell(text.strip(), id=hashlib.sha256(text.encode()).hexdigest()[:12])


def write(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "CSCI-635-ML-Project (3.13.x)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.15"},
    })
    nbf.validate(notebook)
    for cell in cells:
        if cell.cell_type == "code": compile(cell.source, name, "exec")
    nbf.write(notebook, ROOT / name)


for model, title in [("yolo26", "YOLO26 Medium"), ("yolov5", "Original YOLOv5 Medium")]:
    cells = [markdown(f'''# Squirrel multiclass detection — {title} (v2)

Complete workflow: audited dataset → immutable YOLO conversion → model/GPU verification → memory probe → training → validation/test reports → CPU/GPU benchmarks → model package.

**Task:** eleven-class bounding-box detection, with ten species plus `generic_squirrel`.
**Recipe:** pretrained medium model, 640 pixels, standard native YOLO sampling/augmentations, seed 42, AMP, **at most 200 epochs and early-stopping patience 30**.

Run `uv sync --locked --all-extras` before selecting the project `.venv` kernel. Run cells in order. The training cell starts the full experiment; preparation and probe cells do not. Generated files stay under `output_{model}/runs/<run-id>/`.

Recover an interrupted experiment by setting `RESUME_RUN` below and keeping its configuration unchanged. Complete checkpoints remain frozen for validation/test. Source COCO images and splits are never modified.

YOLO26 reports both its accuracy-oriented and NMS-free heads from independent loads of the same checkpoint. Original YOLOv5 uses `yolov5m.pt`, not the anchor-free YOLOv5u variant.
'''), code(f'''# 1. Nonsecret configuration and independent run folder.
from pathlib import Path
import json
from IPython.display import display, HTML, IFrame
from csci_635_ml_project.baseline import (
    default_config, create_run, prepare_run, probe_batches, train_run,
    predict_run, benchmark_run, package_run,
)
from csci_635_ml_project.baseline_reports import dataset_report, evaluate_run

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
DATA = ROOT / "data/squirrel-v2-audited-coco"
CFG = default_config("{model}")  # epochs=200, patience=30, imgsz=640, seed=42
RESUME_RUN = None  # ROOT / "output_{model}/runs/<run-id>"
RUN = create_run(ROOT, CFG, resume_run=RESUME_RUN, data=DATA)
print("Run:", RUN)
display(CFG)
'''), markdown('''## 2. Dataset verification, EDA, and backend setup

Require the passing audited v3 manifest and matching annotation hashes. Export the exact splits and class mapping to a fingerprinted YOLO dataset; every subsequent use verifies its image and label hashes. Native caches use the converted copy. Backend setup pins the model/source version, records pretrained weights and environment hashes, and executes a CUDA kernel.
'''), code('''DATA_YAML = prepare_run(RUN)
class_counts = dataset_report(RUN, DATA)
display(class_counts)
print("YOLO dataset:", DATA_YAML)
display(IFrame(src=str((RUN / "reports/report.html").relative_to(ROOT)), width="100%", height=700))
'''), markdown('''## 3. GPU memory probe

Try batches 16, 8, then 4 on a small seeded train/validation sample. Only CUDA OOM triggers a smaller-batch retry. Native nominal batch size is 64; resolved settings are saved. Probe weights are discarded for full training. Probe folders include batch-loss traces and a short Torch profiler trace where supported.
'''), code('''selected = probe_batches(RUN)
display(selected)
'''), markdown('''## 4. Full training — this cell starts the experiment

Use the native optimizer/augmentation recipe with up to 200 epochs and 30-epoch early-stopping patience. YOLO26 selects its checkpoint by validation mAP50–95; original YOLOv5 uses 0.1 × AP50 + 0.9 × mAP50–95. Keep best/latest, periodic, and unstripped recovery checkpoints. Training logs, batch losses, epoch metrics, timing, VRAM, and loss curves stay in this run.
'''), code('''if (RUN / "training_complete.json").exists():
    print("Training already completed; using the frozen best checkpoint.")
else:
    history = train_run(RUN)
    display(history.tail())
'''), markdown('''## 5. Common validation and held-out test evaluation

Predict at confidence floor 0.0001 with maximum 100 detections. Select each head's threshold using validation micro F1 at matching IoU 0.50; ties choose the highest threshold. Freeze its checkpoint/hash and threshold before test scoring. Final AP/AR uses the same pycocotools evaluator as RF-DETR, rather than comparing framework-native AP implementations.

Reports include per-class AP/precision/recall/F1, aggregate metrics, PR/confidence curves, raw and normalized confusion matrices including background, and annotated successes/errors. Wrong species contributes one false positive and one false negative. Loss scales are model-specific.
'''), code('''predict_run(RUN)
metrics = evaluate_run(RUN, DATA, heads=CFG["heads"])
for head, splits in metrics.items():
    print(head, {split: result["coco"]["mAP_50_95"] for split, result in splits.items()})
print("Reports:", RUN / "reports")
'''), markdown('''## 6. Local GPU and CPU latency

Benchmark identical seeded validation images, batch one, FP32, 20 warm-up calls, and one CPU thread. Include preprocessing/inference/postprocessing; exclude decoding. GPU timing synchronizes CUDA. No compilation or TF32. Save every image timing, mean/median/p95, FPS, hardware, checkpoint, threshold, and timing protocol. These measure local PyTorch performance; deployment-device timing requires a separate experiment.
'''), code('''benchmarks = benchmark_run(RUN)
import pandas as pd
display(pd.DataFrame([{k: row[k] for k in ("head", "device", "mean_ms", "median_ms", "p95_ms", "throughput_fps")} for row in benchmarks]))
'''), markdown('''## 7. Package and optional explicit tracking

Export verified native best weights, class order, frozen thresholds, checkpoint/dataset hashes, model card, and report archive. Automatic framework tracking is disabled. If ClearML credentials exist privately in `.env`, this cell uploads explicit reports and metrics. Failures preserve local reports and a retry queue; no credentials are written into notebook outputs.
'''), code('''best_weights = package_run(RUN)
print("Verified best weights:", best_weights)
print("Recovery checkpoint:", RUN / "native/weights/last_resumable.pt")
print("Report archive:", RUN / "reports.zip")
'''), markdown('''## References

- [YOLO26 models and inference heads](https://docs.ultralytics.com/models/yolo26/)
- [Original YOLOv5 source](https://github.com/ultralytics/yolov5)
- [YOLOv5 versus YOLOv5u compatibility](https://docs.ultralytics.com/models/yolov5/)
- [Ultralytics training settings](https://docs.ultralytics.com/modes/train/)
''')]
    write(f"squirrel_detection_{model}_medium_v2.ipynb", cells)

write("squirrel_detection_model_comparison_v2.ipynb", [markdown('''# Squirrel v2 — RF-DETR, YOLOv5, and YOLO26 comparison

Run after completing both YOLO notebooks, including their evaluation cells. The main table compares RF-DETR Medium, original YOLOv5 Medium, and YOLO26 Medium's accuracy-oriented head. YOLO26's NMS-free head is a secondary variant from the same training run.

Final accuracy uses a common COCO evaluator and frozen validation thresholds. Latency is freshly measured using a shared local CPU/GPU protocol. RF-DETR's existing reports remain unchanged. All comparison artifacts live under `output_model_comparison/runs/<comparison-id>/`.

This compares trained model recipes: YOLO uses 640 pixels, standard sampling, and up to 200 epochs; RF-DETR used 576 pixels, class-aware sampling, and up to 50 epochs. Different pretraining, capacity, losses, and postprocessing also affect results. One seed does not establish a universal architecture winner.
'''), code('''from pathlib import Path
from IPython.display import display, HTML, IFrame
from csci_635_ml_project.baseline_comparison import resolve_runs, build_comparison

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
RFDETR_RUN = ROOT / "output/runs/20260926T153802Z-v2-f35e9e"
YOLO26_RUN = None  # Explicit run path, or newest completed YOLO26 run.
YOLOV5_RUN = None  # Explicit run path, or newest completed original YOLOv5 run.
REFRESH_BENCHMARKS = True  # Fresh common CPU/GPU measurements for all models.
try:
    runs = resolve_runs(ROOT, RFDETR_RUN, YOLO26_RUN, YOLOV5_RUN)
    display({model: str(path) for model, path in runs.items()})
except FileNotFoundError as exc:
    runs = None
    print(exc)
'''), markdown('''## Verify compatibility and build the comparison

Refuse incompatible dataset fingerprints, split counts, class order, checkpoint hashes, frozen thresholds, or latency protocols. Reproduce RF-DETR's saved COCO scores before adding per-class and confusion reports. Re-measure RF-DETR latency rather than mixing its older timings with new YOLO timings.

The benchmark cells can take several minutes, especially on CPU. Accuracy and latency leaders are reported separately. Native loss curves use separate panels; their magnitudes cannot be ranked across models.
'''), code('''if runs is None:
    print("Finish both training/evaluation notebooks, then rerun the configuration cell.")
else:
    COMPARISON, table = build_comparison(ROOT, runs, refresh_benchmarks=REFRESH_BENCHMARKS)
    display(table)
    print("Reports:", COMPARISON)
    display(IFrame(src=str((COMPARISON / "report.html").relative_to(ROOT)), width="100%", height=700))
'''), markdown('''## Saved evidence

`input_runs.json` records the exact experiments. `model_comparison.csv/json` contains validation/test accuracy, per-class aggregate metrics, epochs, duration, checkpoint size, parameters, and CPU/GPU latency. `per_class_comparison.csv`, benchmark records, accuracy/latency plots, species AP plots, and separate loss panels support inspection. Expanded RF-DETR reports are copied under this comparison folder. `findings.json` records leaders and interpretation limits; `reports.zip` packages the report.
''')])
