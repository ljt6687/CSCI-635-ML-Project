"""Notebook orchestration. Model libraries run in separate worker processes."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .baseline_data import (ULTRALYTICS_VERSION, YOLOV5_COMMIT, dataset_fingerprint,
                            export_yolo, save_json, sha256, verify_dataset)


def default_config(model: str) -> dict:
    if model not in {"yolo26", "yolov5"}:
        raise ValueError("Unknown baseline model")
    return dict(model=model, epochs=200, patience=30, imgsz=640, seed=42, batch_candidates=[16, 8, 4],
                workers=4, amp=True, save_period=10, optimizer="auto" if model == "yolo26" else "SGD",
                prediction_floor=0.0001, max_detections=100, matching_iou=.5,
                nms_iou=.7 if model == "yolo26" else .45,
                heads=["accuracy", "end2end"] if model == "yolo26" else ["accuracy"],
                ultralytics_version=ULTRALYTICS_VERSION, yolov5_commit=YOLOV5_COMMIT,
                benchmark_images=200, benchmark_warmup=20, cpu_threads=1)


def redact(text: str) -> str:
    for key, value in os.environ.items():
        if value and any(word in key.upper() for word in ("SECRET", "TOKEN", "PASSWORD", "API_KEY", "ACCESS_KEY")):
            text = text.replace(value, "[REDACTED]")
    return re.sub(r"(?i)([?&](?:token|signature|api_key|key|secret|x-amz-signature)=)[^&\s]+", r"\1[REDACTED]", text)


def create_run(root, config, resume_run=None, data=None) -> Path:
    root = Path(root).resolve()
    data = Path(data or root / "data/squirrel-v2-audited-coco").resolve()
    manifest, _ = verify_dataset(data)
    if not 1 <= config["epochs"] <= 200 or config["patience"] <= 0:
        raise ValueError("Epoch ceiling must be at most 200 and patience positive")
    if resume_run:
        run = Path(resume_run).resolve()
        if json.loads((run / "config.json").read_text()) != config:
            raise ValueError("Resume requires the original configuration")
        previous = json.loads((run / "dataset_manifest.json").read_text())
        if dataset_fingerprint(previous) != dataset_fingerprint(manifest):
            raise ValueError("Resume dataset changed")
        return run
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    run = root / f"output_{config['model']}" / ("smoke" if config.get("smoke") else "runs") / run_id
    run.mkdir(parents=True)
    save_json(run / "config.json", config)
    save_json(run / "dataset_manifest.json", manifest)
    save_json(run / "run_manifest.json", dict(schema_version=1, root=str(root), data=str(data),
                                              dataset_fingerprint=dataset_fingerprint(manifest), model=config["model"]))
    return run


def prepare_backend(root) -> Path:
    """Fetch the exact maintained legacy source, outside tracked project files."""
    root = Path(root).resolve()
    target = root / "output_yolov5/backend" / YOLOV5_COMMIT
    if target.exists():
        saved = json.loads((target / "baseline_source.json").read_text())
        if saved["commit"] != YOLOV5_COMMIT:
            raise ValueError("YOLOv5 source revision mismatch")
        for name, digest in saved["files"].items():
            if sha256(target / name) != digest:
                raise ValueError(f"Pinned YOLOv5 source changed: {name}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.parent / f"{YOLOV5_COMMIT}.tar.gz"
    url = f"https://codeload.github.com/ultralytics/yolov5/tar.gz/{YOLOV5_COMMIT}"
    urllib.request.urlretrieve(url, archive)
    temporary = target.with_name(target.name + ".extracting")
    temporary.mkdir()
    try:
        with tarfile.open(archive) as tar:
            tar.extractall(temporary, filter="data")
        folders = list(temporary.iterdir())
        if len(folders) != 1 or not (folders[0] / "train.py").is_file():
            raise ValueError("Unexpected backend archive layout")
        folders[0].rename(target)
        save_json(target / "baseline_source.json", dict(commit=YOLOV5_COMMIT, url=url, archive_sha256=sha256(archive),
                  files={str(p.relative_to(target)): sha256(p) for p in target.rglob("*") if p.is_file()}))
    finally:
        shutil.rmtree(temporary)
    return target


def run_job(run, action, **kwargs):
    from filelock import FileLock, Timeout
    run = Path(run).resolve()
    context = json.loads((run / "run_manifest.json").read_text())
    lock_path = Path(context["root"]) / "output_model_comparison/.runtime.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=1)
    except Timeout:
        print("Waiting for another model worker to finish; local training and benchmarks run serially.", flush=True)
        lock.acquire()
    try:
        return _run_job(run, action, **kwargs)
    finally:
        lock.release()


def _run_job(run, action, **kwargs):
    run = Path(run).resolve()
    config = json.loads((run / "config.json").read_text())
    context = json.loads((run / "run_manifest.json").read_text())
    jobs = run / "jobs"; jobs.mkdir(exist_ok=True)
    job = jobs / f"{action}-{uuid.uuid4().hex[:8]}.json"
    result = job.with_suffix(".result.json")
    value = dict(action=action, run=str(run), config=config, context=context, result=str(result), **kwargs)
    save_json(job, value)
    (run / ".ultralytics").mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(YOLO_CONFIG_DIR=str(run / ".ultralytics"), YOLO_AUTOINSTALL="false", WANDB_MODE="disabled",
               COMET_MODE="DISABLED", CLEARML_OFFLINE_MODE="1", MPLBACKEND="Agg")
    # Native integrations cannot use credentials; optional explicit tracking is parent-only.
    for key in list(env):
        if key.startswith(("CLEARML_API_", "WANDB_API_", "COMET_API_")):
            env.pop(key)
    log = run / "logs" / f"{job.stem}.log"; log.parent.mkdir(exist_ok=True)
    with log.open("w") as output:
        process = subprocess.Popen([sys.executable, "-u", "-m", "csci_635_ml_project.baseline_worker", str(job)],
                                   cwd=context["root"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                safe = redact(line)
                output.write(safe); output.flush()
                print(safe, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate(); process.wait(timeout=30)
            raise
    response = json.loads(result.read_text()) if result.exists() else {"status": "failed", "error": "Worker exited without a result"}
    if code or response["status"] != "ok":
        message = response.get("error", f"Worker exited with code {code}")
        if response.get("oom"):
            raise MemoryError(message)
        raise RuntimeError(f"{action} failed: {message}; see {log}")
    return response


def prepare_run(run) -> Path:
    run = Path(run)
    context = json.loads((run / "run_manifest.json").read_text())
    config = json.loads((run / "config.json").read_text())
    yaml = export_yolo(context["data"])
    updates = dict(yolo_yaml=str(yaml))
    if config["model"] == "yolov5":
        updates["backend"] = str(prepare_backend(context["root"]))
    context.update(updates)
    save_json(run / "run_manifest.json", context)
    run_job(run, "setup")
    return yaml


def make_probe_yaml(run, data_yaml):
    import random
    import yaml
    run = Path(run)
    root = Path(data_yaml).parent
    manifest = json.loads((root / "export_manifest.json").read_text())
    config = json.loads((run / "config.json").read_text())
    rng = random.Random(config["seed"])
    output = run / "probe_data"; output.mkdir(exist_ok=True)
    settings = yaml.safe_load(Path(data_yaml).read_text())
    for split, count, key in [("train", 128, "train"), ("valid", 32, "val")]:
        rows = [r for r in manifest["images"] if r["split"] == split]
        chosen = rng.sample(rows, min(count, len(rows)))
        path = output / f"{split}.txt"
        path.write_text("\n".join(str(root / r["image"]) for r in chosen) + "\n")
        settings[key] = str(path)
    path = output / "data.yaml"; path.write_text(yaml.safe_dump(settings, sort_keys=False))
    return path


def probe_batches(run):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    selected = run / "selected_batch.json"
    if selected.exists():
        return json.loads(selected.read_text())
    context = json.loads((run / "run_manifest.json").read_text())
    yaml = make_probe_yaml(run, context["yolo_yaml"])
    attempts = []
    for batch in config["batch_candidates"]:
        try:
            result = run_job(run, "train", batch=batch, probe=True, data_yaml=str(yaml),
                             destination=str(run / "probes" / f"batch-{batch}-{uuid.uuid4().hex[:6]}"))
            attempts.append(dict(batch=batch, status="passed"))
            selection = dict(batch_size=batch, nominal_batch=64, attempts=attempts, peak_vram_gib=result["peak_vram_gib"])
            save_json(selected, selection)
            return selection
        except MemoryError:
            attempts.append(dict(batch=batch, status="CUDA OOM"))
            save_json(run / "probe_failures.json", attempts)
    raise RuntimeError("All GPU batch probes failed")


def train_run(run):
    run = Path(run)
    if (run / "training_complete.json").exists():
        raise ValueError("Training already complete; continue with reports")
    config = json.loads((run / "config.json").read_text())
    context = json.loads((run / "run_manifest.json").read_text())
    selected = json.loads((run / "selected_batch.json").read_text())
    export_yolo(context["data"], Path(context["yolo_yaml"]).parent)
    manifest, _ = verify_dataset(context["data"])
    if dataset_fingerprint(manifest) != context["dataset_fingerprint"]:
        raise ValueError("Dataset changed before training")
    destination = run / "native"
    resume = destination.exists()
    checkpoint = run / "native/weights/last_resumable.pt"
    if resume and not checkpoint.is_file():
        raise ValueError("Native run exists without a resumable checkpoint; start a fresh run")
    result = None
    # A parent interruption after a successful worker must not restart finished training.
    for job_path in sorted((run / "jobs").glob("train-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if job_path.name.endswith(".result.json"):
            continue
        previous = json.loads(job_path.read_text())
        result_path = job_path.with_suffix(".result.json")
        if previous.get("probe") or previous.get("config") != config or not result_path.exists():
            continue
        completed = json.loads(result_path.read_text())
        if completed.get("status") == "ok" and completed.get("best_checkpoint_sha256") == sha256(destination / "weights/best.pt"):
            result = completed
            break
    if result is None:
        result = run_job(run, "train", batch=selected["batch_size"], probe=False,
                         data_yaml=context["yolo_yaml"], destination=str(destination),
                         resume=str(checkpoint) if resume else None)
    best = run / "native/weights/best.pt"
    save_json(run / "training_complete.json", dict(best_checkpoint=str(best.relative_to(run)), sha256=sha256(best),
              max_epochs=config["epochs"], epochs_completed=result["epochs_completed"],
              checkpoint_selection="validation mAP50-95" if config["model"] == "yolo26" else "0.1 * validation AP50 + 0.9 * validation mAP50-95"))
    from .baseline_reports import training_report
    return training_report(run)


def predict_run(run):
    run = Path(run)
    if not (run / "training_complete.json").exists():
        raise ValueError("Complete training first")
    context = json.loads((run / "run_manifest.json").read_text())
    export_yolo(context["data"], Path(context["yolo_yaml"]).parent)
    return run_job(run, "predict")


def benchmark_run(run):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    return [run_job(run, "benchmark", head=head, device=device)
            for head in config["heads"] for device in ("cuda", "cpu")]


def package_run(run):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    completion = json.loads((run / "training_complete.json").read_text())
    weights = run / "weights"; weights.mkdir(exist_ok=True)
    best = run / completion["best_checkpoint"]
    if sha256(best) != completion["sha256"]:
        raise ValueError("Best checkpoint changed")
    shutil.copyfile(best, weights / "best.pt")
    manifest = json.loads((run / "dataset_manifest.json").read_text())
    metadata = dict(config=config, classes=manifest["categories"], dataset_fingerprint=dataset_fingerprint(manifest),
                    checkpoint_sha256=completion["sha256"], decisions={h: json.loads((run / (
                        "decision_threshold.json" if h == "accuracy" else f"decision_threshold_{h}.json")).read_text()) for h in config["heads"]})
    save_json(weights / "model_metadata.json", metadata)
    (weights / "MODEL_CARD.md").write_text(
        f"# {config['model']} Medium — squirrel v2\n\n11-class object detector, fine-tuned at {config['imgsz']} pixels.\n"
        "Class order, frozen confidence thresholds, dataset identity, and checkpoint hash are in model_metadata.json.\n"
        "Validation/test reports are in ../reports; CPU/GPU timings are in ../benchmarks.\n"
        "Original YOLOv5 checkpoints require the pinned original repository, not the YOLOv5u loader.\n"
        "The two YOLO26 heads require independent checkpoint loads before inference fusion.\n")
    shutil.make_archive(str(run / "reports"), "zip", run / "reports")
    track_run(run)
    return weights / "best.pt"


def track_run(run):
    """Explicit sanitized uploads only; retry locally queued events if tracking fails."""
    from dotenv import load_dotenv
    run = Path(run)
    context = json.loads((run / "run_manifest.json").read_text())
    load_dotenv(Path(context["root"]) / ".env")
    if not all(os.getenv(k) for k in ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY")):
        save_json(run / "tracking_status.json", dict(status="disabled", reason="credentials unavailable"))
        return
    events = [dict(type="artifact", path=str(p.relative_to(run))) for p in run.rglob("*")
              if p.is_file() and (p.suffix in {".png", ".csv", ".html"} or p.name in {"reports.zip", "model_metadata.json", "config.json"})]
    pending = run / "tracking_pending.json"
    save_json(pending, events)
    task = None
    try:
        from clearml import Task
        task = Task.init(project_name="CSCI-635-Squirrel-Multiclass-v2", task_name=f"{context['model']}-{run.name}",
                         reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
                         auto_connect_streams=False, auto_resource_monitoring=False)
        task.connect(json.loads((run / "config.json").read_text()), name="Training config")
        history = run / "reports/training_history.csv"
        if history.exists():
            import pandas as pd
            for row in pd.read_csv(history).to_dict("records"):
                for name, value in row.items():
                    if name != "epoch" and pd.notna(value):
                        task.get_logger().report_scalar("epoch metrics", name, float(value), int(row["epoch"]))
        remaining = events.copy()
        for event in events:
            task.upload_artifact(name=event["path"], artifact_object=str(run / event["path"]))
            remaining.remove(event); save_json(pending, remaining)
        task.close()
        save_json(run / "tracking_status.json", dict(status="uploaded", task_id=task.id))
    except Exception as exc:
        save_json(run / "tracking_status.json", dict(status="pending", error=redact(str(exc))))
        if task:
            with __import__('contextlib').suppress(Exception):
                task.close()
