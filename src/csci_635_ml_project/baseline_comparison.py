"""Read completed baseline runs and generate a common three-model comparison."""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baseline import benchmark_run, default_config
from .baseline_data import dataset_fingerprint, records_for_split, save_json, sha256, verify_dataset
from .baseline_reports import evaluate_report, figure, html_report, load_predictions

DEFAULT_RFDETR_RUN = "20260926T153802Z-v2-f35e9e"
LABELS = {"rfdetr": "RF-DETR Medium", "yolov5": "YOLOv5 Medium", "yolo26": "YOLO26 Medium"}


def resolve_runs(root, rfdetr_run=None, yolo26_run=None, yolov5_run=None) -> dict:
    root = Path(root).resolve()
    result = {"rfdetr": Path(rfdetr_run or root / "output/runs" / DEFAULT_RFDETR_RUN).resolve()}
    for model, override in [("yolo26", yolo26_run), ("yolov5", yolov5_run)]:
        if override:
            result[model] = Path(override).resolve()
        else:
            manifest, _ = verify_dataset(root / "data/squirrel-v2-audited-coco")
            candidates = [p.parent for p in (root / f"output_{model}/runs").glob("*/training_complete.json")
                          if dataset_fingerprint(json.loads((p.parent / "dataset_manifest.json").read_text())) == dataset_fingerprint(manifest)]
            if not candidates:
                raise FileNotFoundError(f"No completed {model} run; finish its training/evaluation notebook first")
            result[model] = max(candidates, key=lambda p: p.name)
    return result


def validate_runs(runs, manifest):
    expected = dataset_fingerprint(manifest)
    for model, run in runs.items():
        run = Path(run)
        saved = json.loads((run / "dataset_manifest.json").read_text())
        if dataset_fingerprint(saved) != expected:
            raise ValueError(f"{model}: incompatible dataset, split counts, or class order")
        completion = json.loads((run / "training_complete.json").read_text())
        if sha256(run / completion["best_checkpoint"]) != completion["sha256"]:
            raise ValueError(f"{model}: best checkpoint hash differs")
        for split in ("valid", "test"):
            if not (run / "reports" / split / "predictions.json").exists():
                raise FileNotFoundError(f"{model}: {split} predictions are missing")
        if model != "rfdetr":
            cfg = json.loads((run / "config.json").read_text())
            if cfg["model"] != model:
                raise ValueError("Comparison model and run configuration disagree")
            for head in cfg["heads"]:
                base = run / "reports" if head == "accuracy" else run / "reports" / head
                decision = json.loads((run / ("decision_threshold.json" if head == "accuracy" else f"decision_threshold_{head}.json")).read_text())
                if decision["checkpoint_sha256"] != completion["sha256"] or decision["dataset_fingerprint"] != expected or decision["inference_head"] != head:
                    raise ValueError(f"{model}/{head}: frozen decision belongs to different checkpoint/data/head")
                for split in ("valid", "test"):
                    metric = json.loads((base / split / "metrics.json").read_text())
                    metadata = metric["metadata"]
                    if metadata["dataset_fingerprint"] != expected or metadata["checkpoint_sha256"] != completion["sha256"]:
                        raise ValueError(f"{model}/{head}: stale evaluation")
                    if metric["confidence_threshold"] != decision["threshold"] or metadata["inference_head"] != head:
                        raise ValueError(f"{model}/{head}: test threshold/head differs from validation")


def validate_benchmarks(benchmarks):
    if not benchmarks:
        raise ValueError("No shared benchmark records")
    reference = benchmarks[0]["protocol"]
    for benchmark in benchmarks:
        if benchmark["protocol"] != reference:
            raise ValueError("Latency records use different hardware, image IDs, precision, threads, or timing scopes; refresh benchmarks")


def prepare_rfdetr_copy(original, output, root, data):
    original, output = Path(original), Path(output)
    output.mkdir(parents=True)
    for name in ("dataset_manifest.json", "environment.json", "decision_threshold.json"):
        shutil.copyfile(original / name, output / name)
    cfg = default_config("yolo26")
    previous = json.loads((original / "config.json").read_text())
    cfg.update(model="rfdetr", imgsz=previous["resolution"], heads=["accuracy"], epochs=previous["epochs"])
    cfg.update({k: previous[k] for k in ("benchmark_images", "benchmark_warmup", "cpu_threads") if k in previous})
    save_json(output / "config.json", cfg)
    completion = json.loads((original / "training_complete.json").read_text())
    completion["best_checkpoint"] = str((original / completion["best_checkpoint"]).resolve())
    save_json(output / "training_complete.json", completion)
    manifest, docs = verify_dataset(data)
    save_json(output / "image_verification.json", verify_reference_pixels(original, data, docs))
    save_json(output / "run_manifest.json", dict(root=str(root), data=str(data), model="rfdetr",
              dataset_fingerprint=dataset_fingerprint(manifest), source_run=str(original)))
    checkpoint_hash = completion["sha256"]
    decision = json.loads((original / "decision_threshold.json").read_text())
    if decision["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("RF-DETR threshold was selected for another checkpoint")
    threshold = decision["threshold"]
    for split in ("valid", "test"):
        directory = output / "reports" / split
        directory.mkdir(parents=True)
        source = original / "reports" / split / "predictions.json"
        raw = json.loads(source.read_text())
        names = {c["id"]: c["name"] for c in docs[split]["categories"]}
        excluded = []
        normalized = {}
        for image_id, predictions in raw.items():
            normalized[image_id] = []
            for prediction in predictions:
                if prediction["category_id"] not in names:
                    if (prediction.get("label"), prediction["category_id"], prediction["class_name"]) != (11, 12, "cls_11") or prediction["score"] >= threshold:
                        raise ValueError("RF-DETR has an unknown label affecting its frozen operating point")
                    excluded.append(dict(image_id=int(image_id), **prediction))
                else:
                    normalized[image_id].append(prediction)
        shutil.copyfile(source, directory / "predictions.original.json")
        save_json(directory / "predictions.json", normalized)
        save_json(directory / "unmapped_predictions.json", dict(
            source_sha256=sha256(source), count=len(excluded), predictions=excluded,
            policy="Archived output index 11 is outside the 11-class taxonomy and below the frozen threshold; excluded from common reports. COCO AP and frozen operating metrics must reproduce the source report."))
        predictions = load_predictions(directory / "predictions.json", docs[split])
        metrics = evaluate_report(docs[split], records_for_split(data, split, docs[split]), predictions,
                                  threshold, directory, dict(split=split, checkpoint_sha256=checkpoint_hash,
                                  dataset_fingerprint=dataset_fingerprint(manifest), inference_head="accuracy"))
        saved = json.loads((original / "reports" / split / "metrics.json").read_text())
        for key, value in saved["coco"].items():
            current = metrics["coco"][key]
            if value is not None and (current is None or abs(value - current) > 1e-9):
                raise ValueError(f"RF-DETR cached-prediction metric parity failed: {split}/{key}")
        for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
            if abs(saved["operating"][key] - metrics["operating"][key]) > 1e-9:
                raise ValueError(f"RF-DETR frozen operating metric parity failed: {split}/{key}")
    shutil.copyfile(original / "reports/training_history.csv", output / "reports/training_history.csv")
    return output


def build_comparison(root, runs=None, refresh_benchmarks=True):
    root = Path(root).resolve(); data = root / "data/squirrel-v2-audited-coco"
    runs = runs or resolve_runs(root)
    manifest, _ = verify_dataset(data)
    validate_runs(runs, manifest)
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    output = root / "output_model_comparison/runs" / identifier
    output.mkdir(parents=True)
    save_json(output / "input_runs.json", {model: str(Path(run).resolve()) for model, run in runs.items()})
    save_json(output / "dataset_manifest.json", manifest)
    rf = prepare_rfdetr_copy(runs["rfdetr"], output / "rfdetr", root, data)
    evaluation_runs = {**runs, "rfdetr": rf}
    # RF-DETR's old latency lacks the common protocol and is never substituted.
    benchmark_run(rf)
    if refresh_benchmarks:
        for model in ("yolov5", "yolo26"):
            benchmark_run(evaluation_runs[model])
    records, benchmark_records, class_records, confusion_records = [], [], [], []
    for model, run in evaluation_runs.items():
        run = Path(run); config = json.loads((run / "config.json").read_text())
        for head in config["heads"]:
            base = run / "reports" if head == "accuracy" else run / "reports" / head
            test = json.loads((base / "test/metrics.json").read_text())
            valid = json.loads((base / "valid/metrics.json").read_text())
            label = LABELS[model] + (" (NMS-free)" if head == "end2end" else "")
            completion = json.loads((run / "training_complete.json").read_text())
            checkpoint = run / completion["best_checkpoint"]
            values = dict(model=label, role="secondary head" if head == "end2end" else "primary",
                          input_size=config["imgsz"], validation_mAP_50_95=valid["coco"]["mAP_50_95"],
                          test_mAP_50_95=test["coco"]["mAP_50_95"], test_AP50=test["coco"]["AP50"],
                          test_precision=test["operating"]["precision"], test_recall=test["operating"]["recall"],
                          test_micro_f1=test["operating"]["f1"], test_macro_f1=test["macro"]["f1"],
                          threshold=test["confidence_threshold"], checkpoint_mib=checkpoint.stat().st_size / 2**20)
            history = pd.read_csv(run / "reports/training_history.csv")
            values["epochs_completed"] = int(history.epoch.max())
            values["epoch_seconds_total"] = float(history.seconds.sum()) if "seconds" in history else None
            for device in ("cuda", "cpu"):
                benchmark = json.loads((run / "benchmarks" / head / f"{device}.json").read_text())
                if benchmark["checkpoint_sha256"] != completion["sha256"] or benchmark["confidence_threshold"] != test["confidence_threshold"]:
                    raise ValueError(f"{label}: stale benchmark checkpoint/threshold")
                benchmark_records.append(benchmark)
                values[f"{device}_mean_ms"] = benchmark["mean_ms"]
                values[f"{device}_p95_ms"] = benchmark["p95_ms"]
                values[f"{device}_fps"] = benchmark["throughput_fps"]
                values["inference_parameters"] = benchmark["parameters"]
            records.append(values)
            confusion_records.append((label, pd.read_csv(base / "test/confusion_normalized.csv", index_col=0)))
            class_records.extend(dict(model=label, **row) for row in test["per_class"])
    validate_benchmarks(benchmark_records)
    table = pd.DataFrame(records)
    table.to_csv(output / "model_comparison.csv", index=False)
    save_json(output / "model_comparison.json", records)
    save_json(output / "benchmark_records.json", benchmark_records)
    classes = pd.DataFrame(class_records); classes.to_csv(output / "per_class_comparison.csv", index=False)
    for device in ("cuda", "cpu"):
        fig, ax = plt.subplots(figsize=(9, 6))
        for row in records:
            ax.scatter(row[f"{device}_mean_ms"], row["test_mAP_50_95"], s=70)
            ax.annotate(row["model"], (row[f"{device}_mean_ms"], row["test_mAP_50_95"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set(xlabel=f"{device.upper()} mean batch-one latency (ms)", ylabel="Test COCO mAP50–95", title="Accuracy and latency")
        ax.grid(alpha=.2); figure(fig, output / f"accuracy_latency_{device}.png")
    pivot = classes.pivot(index="class_name", columns="model", values="mAP_50_95")
    fig, ax = plt.subplots(figsize=(15, 7)); pivot.plot.bar(ax=ax)
    ax.set(ylabel="Test COCO AP50–95", xlabel="Species", ylim=(0, 1)); ax.tick_params(axis="x", labelsize=8)
    figure(fig, output / "per_class_ap.png")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, model in zip(axes, ("rfdetr", "yolov5", "yolo26")):
        history = pd.read_csv(Path(evaluation_runs[model]) / "reports/training_history.csv")
        loss_columns = [c for c in history if "loss" in c and not any(t in c for t in ("_aux", "_enc", "_0", "_1", "_2"))]
        history.plot(x="epoch", y=loss_columns, ax=ax, legend=True)
        ax.set_title(LABELS[model] + " native loss scales"); ax.grid(alpha=.2)
    figure(fig, output / "model_loss_panels.png")
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    for ax, (label, matrix) in zip(axes.flat, confusion_records):
        ax.imshow(matrix.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=65, ha="right", fontsize=7)
        ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=7)
        ax.set(title=label, xlabel="Predicted", ylabel="Ground truth")
        for i in range(len(matrix)):
            for j in range(len(matrix.columns)):
                ax.text(j, i, f"{matrix.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if matrix.iloc[i,j] > .5 else "black")
    figure(fig, output / "confusion_panels.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    fields = {"rfdetr": "val/mAP_50_95", "yolov5": "metrics/mAP_0.5:0.95", "yolo26": "metrics/mAP50-95(B)"}
    for model in fields:
        history = pd.read_csv(Path(evaluation_runs[model]) / "reports/training_history.csv")
        if fields[model] in history:
            ax.plot(history.epoch, history[fields[model]], label=LABELS[model])
    ax.set(xlabel="Epoch", ylabel="Native validation mAP50–95", title="Training progress (native AP definitions differ)")
    ax.legend(); ax.grid(alpha=.2); figure(fig, output / "native_validation_ap.png")
    primary = [r for r in records if r["role"] == "primary"]
    findings = dict(accuracy_leader=max(primary, key=lambda r: r["test_mAP_50_95"])["model"],
                    gpu_latency_leader=min(primary, key=lambda r: r["cuda_mean_ms"])["model"],
                    cpu_latency_leader=min(primary, key=lambda r: r["cpu_mean_ms"])["model"],
                    limitations=["One seed and one dataset; recipe-level comparison, not an isolated architecture ablation.",
                        "YOLO: 640 pixels, standard sampling, max 200 epochs with early stopping; RF-DETR: 576 pixels, class-aware sampling, max 50 epochs.",
                        "Loss functions and native training AP implementations differ; final AP uses the common COCO evaluator.",
                        "Pretraining datasets and inference postprocessing differ; medium model sizes do not imply equal capacity.",
                        "CPU results measure local PyTorch inference and do not establish latency on a particular edge device.",
                        "Archived RF-DETR output index 11 is outside the taxonomy; its low-confidence predictions are audited and excluded, with original AP and frozen operating scores checked for parity."])
    save_json(output / "findings.json", findings)
    html_report(output, "Squirrel v2 — RF-DETR and YOLO comparison", findings)
    import zipfile
    with zipfile.ZipFile(output / "reports.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path != output / "reports.zip":
                archive.write(path, path.relative_to(output))
    return output, table


def verify_reference_pixels(original, data, docs):
    """Bind the historical RF-DETR baseline to current decoded image content."""
    import csv
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from .baseline_data import image_path
    path = Path(original) / "reports/image_inventory.csv"
    with path.open() as f:
        inventory = {(r["split"], int(r["image_id"])): r for r in csv.DictReader(f)}
    expected = {(split, im["id"]): im for split, doc in docs.items() for im in doc["images"]}
    if set(inventory) != set(expected):
        raise ValueError("RF-DETR image inventory differs from current split/image IDs")
    def check(item):
        (split, image_id), im = item
        saved = inventory[(split, image_id)]
        if saved["file_name"] != im["file_name"]:
            raise ValueError("RF-DETR filename identity changed")
        with Image.open(image_path(Path(data), split, im)) as opened:
            rgb = opened.convert("RGB")
        w, h = rgb.size
        digest = hashlib.sha256(f"{w}x{h}".encode() + rgb.tobytes()).hexdigest()
        if digest != saved["pixel_hash"]:
            raise ValueError(f"RF-DETR baseline image pixels changed: {split}/{image_id}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check, expected.items()))
    return dict(images_verified=len(expected), source_inventory_sha256=sha256(path), status="passed")
