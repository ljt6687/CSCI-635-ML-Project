"""Isolated pinned model runtimes; launched by baseline.run_job."""
from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
import urllib.request
from pathlib import Path

from .baseline_data import ULTRALYTICS_VERSION, dataset_fingerprint, records_for_split, save_json, sha256, verify_dataset
from .baseline import prepare_backend, redact


class SafeWriter:
    def __init__(self, stream): self.stream = stream
    def write(self, value): return self.stream.write(redact(value))
    def flush(self): return self.stream.flush()
    def isatty(self): return False
    @property
    def encoding(self): return self.stream.encoding


def runtime(job):
    import torch
    import ultralytics
    if importlib.metadata.version("ultralytics") != ULTRALYTICS_VERSION:
        raise RuntimeError(f"Install the locked YOLO extra (ultralytics=={ULTRALYTICS_VERSION})")
    disabled = {k: False for k in ("sync", "clearml", "wandb", "comet", "mlflow", "neptune", "hub") if k in ultralytics.settings}
    disabled["weights_dir"] = str(Path(job["context"]["root"]) / f"output_{job['config']['model']}/pretrained")
    ultralytics.settings.update(disabled)
    if job["action"] in {"setup", "train", "predict"} and not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA GPU is required for this training workflow")
    random.seed(job["config"]["seed"])
    import numpy as np
    np.random.seed(job["config"]["seed"]); torch.manual_seed(job["config"]["seed"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(job["config"]["seed"])
    return torch


def weights_path(job):
    model = job["config"]["model"]
    path = Path(job["context"]["root"]) / f"output_{model}/pretrained" / ("yolo26m.pt" if model == "yolo26" else "yolov5m.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if model == "yolo26":
            from ultralytics.utils.downloads import attempt_download_asset
            attempt_download_asset(str(path))
        else:
            temporary = path.with_suffix(".part")
            urllib.request.urlretrieve("https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt", temporary)
            temporary.replace(path)
    if not path.is_file(): raise RuntimeError(f"Pretrained weights unavailable at {path}")
    return path


def legacy_imports(job):
    backend = prepare_backend(job["context"]["root"])
    sys.path.insert(0, str(backend))
    return backend


def model_diagram(model, directory):
    import matplotlib.pyplot as plt
    from .baseline_reports import figure
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    (directory / "model_summary.txt").write_text(str(model) + "\n")
    layers = list(model.model) if hasattr(model, "model") else list(model.children())
    n = len(layers); columns = 4
    fig, ax = plt.subplots(figsize=(14, max(5, ((n + columns - 1) // columns) * 1.4)))
    positions = {i: (i % columns, -(i // columns)) for i in range(n)}
    for i, layer in enumerate(layers):
        x, y = positions[i]
        ax.text(x, y, f"{i}: {type(layer).__name__}", ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=.5", facecolor="#e7efff", edgecolor="#496b9d"))
        sources = getattr(layer, "f", -1)
        sources = sources if isinstance(sources, list) else [sources]
        for source in sources:
            source = i + source if source < 0 else source
            if source in positions and source != i:
                ax.annotate("", xy=(x, y + .13), xytext=positions[source],
                            arrowprops=dict(arrowstyle="->", alpha=.25, color="#496b9d"))
    ax.set(xlim=(-.5, columns - .5), ylim=(-(n // columns) - .5, .5), title="Model modules and source connections")
    ax.axis("off"); figure(fig, directory / "architecture.png")
    return dict(parameters=sum(p.numel() for p in model.parameters()), modules=n)


def setup(job):
    torch = runtime(job)
    path = weights_path(job); run = Path(job["run"])
    if job["config"]["model"] == "yolo26":
        from ultralytics import YOLO
        model = YOLO(str(path)).model
    else:
        legacy_imports(job)
        from models.experimental import attempt_load
        model = attempt_load(str(path), device=torch.device("cpu"), fuse=False)
    summary = model_diagram(model, run / "reports/pretrained")
    save_json(run / "pretrained.json", dict(path=str(path), sha256=sha256(path), **summary))
    # Kernel execution, not only a driver inventory.
    assert torch.zeros(1, device="cuda").add(1).item() == 1
    environment = dict(python=platform.python_version(), packages={k: importlib.metadata.version(k) for k in (
        "torch", "torchvision", "ultralytics", "rfdetr", "numpy", "pycocotools")},
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        total_vram_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
        cpu=cpu_name(), yolov5_commit=job["config"]["yolov5_commit"])
    save_json(run / "environment.json", environment)
    return dict(status="ok", **environment)


def cpu_name():
    path = Path("/proc/cpuinfo")
    if path.exists():
        return next((s.split(":", 1)[1].strip() for s in path.read_text().splitlines() if s.startswith("model name")), platform.processor())
    return platform.processor()


def flatten_losses(losses, prefix="loss"):
    if isinstance(losses, dict):
        result = {}
        for name, values in sorted(losses.items()):
            result.update(flatten_losses(values, f"{prefix}/{name}"))
        return result
    if hasattr(losses, "detach"):
        losses = losses.detach().cpu().reshape(-1).tolist()
    if isinstance(losses, (list, tuple)):
        return {f"{prefix}/{i}": float(value) for i, value in enumerate(losses)}
    return {prefix: float(losses)}


def train(job):
    torch = runtime(job)
    config = job["config"]; run = Path(job["run"]); out = Path(job["destination"])
    probe = job["probe"]; epochs = 1 if probe else config["epochs"]
    out.parent.mkdir(parents=True, exist_ok=True)
    started = [time.perf_counter()]; peak = [0.0]; completed = [0]; profiler = [None]; step = [0]; active_epoch = [None]
    trace_dir = out / "traces"; trace_dir.mkdir(parents=True, exist_ok=True)
    history_path = out / "history.jsonl" if probe else run / "history.jsonl"
    trace_file = trace_dir / "batch_losses.jsonl"

    def epoch_start(*args, **kwargs):
        torch.cuda.reset_peak_memory_stats(); started[0] = time.perf_counter(); step[0] = 0
        active_epoch[0] = args[0].epoch if args and hasattr(args[0], "epoch") else None

    def batch_trace(losses, step):
        values = flatten_losses(losses)
        if not all(__import__('math').isfinite(v) for v in values.values()):
            raise FloatingPointError("Nonfinite training loss; inspect the saved batch trace")
        with trace_file.open("a") as f:
            f.write(json.dumps(dict(step=int(step), losses=values)) + "\n")
        if profiler[0] is not None:
            profiler[0].step()

    def fit_end(epoch):
        if config["model"] == "yolo26" and epoch != active_epoch[0]:
            return  # final best-checkpoint validation is not another training epoch
        torch.cuda.synchronize()
        memory = torch.cuda.max_memory_allocated() / 2**30
        peak[0] = max(peak[0], memory); completed[0] = int(epoch) + 1
        with history_path.open("a") as f:
            f.write(json.dumps(dict(epoch=completed[0], seconds=time.perf_counter() - started[0], peak_vram_gib=memory)) + "\n")

    def save_resume(path, epoch=None):
        if not probe:
            fit_end(epoch)
            destination = out / "weights/last_resumable.pt"
            temporary = destination.with_suffix(".tmp")
            shutil.copyfile(path, temporary); temporary.replace(destination)
            if job.get("interrupt_after") and epoch is not None and epoch + 1 >= job["interrupt_after"]:
                raise RuntimeError("Intentional smoke interruption after checkpoint save")

    if probe:
        try:
            profiler[0] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                                 record_shapes=True, profile_memory=True)
            profiler[0].start()
        except Exception as exc:
            save_json(trace_dir / "profiler_status.json", dict(status="unavailable", error=redact(str(exc))))
            profiler[0] = None
    try:
        if config["model"] == "yolo26":
            from ultralytics import YOLO
            model = YOLO(job.get("resume") or str(weights_path(job)))
            model.add_callback("on_train_epoch_start", epoch_start)
            def on_batch(trainer):
                batch_trace(trainer.loss_items, trainer.epoch * len(trainer.train_loader) + step[0]); step[0] += 1
            model.add_callback("on_train_batch_end", on_batch)
            model.add_callback("on_fit_epoch_end", lambda trainer: fit_end(trainer.epoch))
            model.add_callback("on_model_save", lambda trainer: save_resume(trainer.last, trainer.epoch))
            # Match the full run's auto-selected optimizer during the memory probe.
            optimizer = "MuSGD" if probe and config["optimizer"] == "auto" else config["optimizer"]
            model.train(data=job["data_yaml"], epochs=epochs, patience=config["patience"], batch=job["batch"],
                        imgsz=config["imgsz"], device=0, workers=config["workers"], amp=config["amp"],
                        optimizer=optimizer, seed=config["seed"], deterministic=True, cache=False, val=True,
                        save=True, save_period=-1 if probe else config["save_period"],
                        project=str(out.parent), name=out.name, exist_ok=True, plots=True,
                        resume=bool(job.get("resume")))
            if not probe:
                fresh = YOLO(str(out / "weights/best.pt"))
                metadata = model_diagram(fresh.model, run / "reports")
        else:
            backend = legacy_imports(job)
            import train as native
            from functools import partial
            from utils.callbacks import Callbacks
            from utils.loggers import Loggers
            native.Loggers = partial(Loggers, include=("csv", "tb"))
            callbacks = Callbacks()
            callbacks.register_action("on_train_epoch_start", callback=epoch_start)
            callbacks.register_action("on_train_batch_end", callback=lambda model, ni, imgs, targets, paths, losses: batch_trace(losses, ni))
            callbacks.register_action("on_fit_epoch_end", callback=lambda values, epoch, best, fitness: fit_end(epoch))
            callbacks.register_action("on_model_save", callback=lambda last, epoch, *args: save_resume(last, epoch))
            opt = native.parse_opt(known=True)
            for key, value in dict(weights=str(weights_path(job)), data=job["data_yaml"], epochs=epochs,
                    patience=config["patience"], batch_size=job["batch"], imgsz=config["imgsz"], device="0",
                    workers=config["workers"], seed=config["seed"], optimizer=config["optimizer"],
                    project=str(out.parent), name=out.name, exist_ok=True, save_period=-1 if probe else config["save_period"],
                    hyp=str(backend / "data/hyps/hyp.scratch-low.yaml"), resume=job.get("resume") or False).items():
                setattr(opt, key, value)
            native.main(opt, callbacks)
            if not probe:
                from models.experimental import attempt_load
                fresh = attempt_load(str(out / "weights/best.pt"), device=torch.device("cpu"), fuse=False)
                metadata = model_diagram(fresh, run / "reports")
        if not probe:
            save_json(run / "trained_model.json", dict(**metadata, checkpoint_bytes=(out / "weights/best.pt").stat().st_size))
        import pandas as pd
        history = pd.read_csv(out / "results.csv")
        history.columns = [c.strip() for c in history.columns]
        actual_epochs = int(history.epoch.max()) + (1 if history.epoch.iloc[0] == 0 else 0)
        return dict(status="ok", peak_vram_gib=peak[0], epochs_completed=actual_epochs,
                    best_checkpoint_sha256=sha256(out / "weights/best.pt"))
    finally:
        if profiler[0] is not None:
            try:
                profiler[0].stop()
                profiler[0].export_chrome_trace(str(trace_dir / "probe_torch_trace.json"))
            except Exception as exc:
                save_json(trace_dir / "profiler_status.json", dict(status="unavailable", error=redact(str(exc))))


def load_adapter(job, head, device, threshold):
    """Decoded RGB PIL -> original-image xyxy predictions; no image decoding inside."""
    import torch
    import numpy as np
    run = Path(job["run"]); config = job["config"]
    completion = json.loads((run / "training_complete.json").read_text())
    checkpoint = run / completion["best_checkpoint"]
    if sha256(checkpoint) != completion["sha256"]:
        raise ValueError("Frozen checkpoint changed")
    names = json.loads((run / "dataset_manifest.json").read_text())["categories"]
    if config["model"] == "rfdetr":
        from rfdetr import RFDETRMedium
        model = RFDETRMedium(pretrain_weights=str(checkpoint), device=device, resolution=config["imgsz"], num_classes=len(names))
        model.model.model.float()
        def predict(image):
            result = model.predict(image, threshold=threshold)
            return convert(result.xyxy, result.confidence, result.class_id, names, config["max_detections"])
    elif config["model"] == "yolo26":
        from ultralytics import YOLO
        # Fresh loads are required: fusion discards the unused branch.
        model = YOLO(str(checkpoint))
        check_class_names(model.names, names)
        def predict(image):
            result = model.predict(image, imgsz=config["imgsz"], rect=False, conf=threshold, iou=config["nms_iou"],
                                   max_det=config["max_detections"], device=device, quantize=32,
                                   nms=False if head == "end2end" else None, verbose=False)[0]
            if bool(model.predictor.model.end2end) != (head == "end2end"):
                raise RuntimeError("YOLO26 inference selected the wrong detection head")
            return convert(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                           result.boxes.cls.cpu().numpy(), names, config["max_detections"])
    else:
        legacy_imports(job)
        from models.common import DetectMultiBackend
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes
        model = DetectMultiBackend(str(checkpoint), device=torch.device(device), fp16=False, fuse=True)
        check_class_names(model.names, names)
        def predict(image):
            rgb = np.asarray(image)
            padded = letterbox(rgb, new_shape=(config["imgsz"], config["imgsz"]), auto=False)[0]
            tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).to(device).float() / 255
            with torch.inference_mode():
                raw = model(tensor[None])
                predictions = non_max_suppression(raw, threshold, config["nms_iou"], multi_label=False,
                                                   max_det=config["max_detections"])[0]
                if len(predictions):
                    predictions[:, :4] = scale_boxes(tensor.shape[1:], predictions[:, :4], rgb.shape)
            predictions = predictions.cpu().numpy()
            return convert(predictions[:, :4], predictions[:, 4], predictions[:, 5], names, config["max_detections"])
    predict.model_getter = lambda: model.model.model if config["model"] == "rfdetr" else model.model
    return predict


def check_class_names(actual, expected):
    values = [actual[i] for i in range(len(actual))] if isinstance(actual, dict) else list(actual)
    if values != expected:
        raise ValueError("Checkpoint class order differs from the audited taxonomy")


def convert(boxes, scores, labels, names, maximum):
    result = []
    for box, score, label in zip(boxes, scores, labels):
        label = int(label)
        if not 0 <= label < len(names):
            raise ValueError("Checkpoint labels differ from the audited eleven classes")
        result.append(dict(box=[float(v) for v in box], score=float(score), label=label,
                           category_id=label + 1, class_name=names[label]))
    return sorted(result, key=lambda p: -p["score"])[:maximum]


def predict(job):
    runtime(job)
    from PIL import Image
    run = Path(job["run"]); config = job["config"]
    _, docs = verify_dataset(job["context"]["data"])
    for head in config["heads"]:
        adapter = load_adapter(job, head, "cuda", config["prediction_floor"])
        base = run / "reports" if head == "accuracy" else run / "reports" / head
        for split in ("valid", "test"):
            predictions = {}
            rows = records_for_split(job["context"]["data"], split, docs[split])
            for i, r in enumerate(rows):
                with Image.open(r["path"]) as im:
                    predictions[str(r["image_id"])] = adapter(im.convert("RGB"))
                if (i + 1) % 200 == 0: print(f"{head}/{split}: {i + 1}/{len(rows)} images", flush=True)
            save_json(base / split / "predictions.json", predictions)
        del adapter
        __import__('gc').collect(); __import__('torch').cuda.empty_cache()
    return dict(status="ok")


def benchmark(job):
    torch = runtime(job)
    import numpy as np
    from PIL import Image
    run = Path(job["run"]); config = job["config"]; device = job["device"]; head = job["head"]
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requested but CUDA is unavailable")
    manifest, docs = verify_dataset(job["context"]["data"])
    if dataset_fingerprint(manifest) != job["context"]["dataset_fingerprint"]:
        raise ValueError("Benchmark dataset changed from the training run")
    rows = sorted(records_for_split(job["context"]["data"], "valid", docs["valid"]), key=lambda r: r["image_id"])
    rows = random.Random(config["seed"]).sample(rows, min(config["benchmark_images"], len(rows)))
    path = run / ("decision_threshold.json" if head == "accuracy" else f"decision_threshold_{head}.json")
    decision = json.loads(path.read_text())
    completion = json.loads((run / "training_complete.json").read_text())
    if decision["checkpoint_sha256"] != completion["sha256"]:
        raise ValueError("Benchmark threshold belongs to a different checkpoint")
    adapter = load_adapter(job, head, device, decision["threshold"])
    images = []
    for r in rows:
        with Image.open(r["path"]) as im: images.append(im.convert("RGB"))
    for i in range(config["benchmark_warmup"]): adapter(images[i % len(images)])
    durations = []
    if device == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    for r, image in zip(rows, images):
        if device == "cuda": torch.cuda.synchronize()
        start = time.perf_counter(); adapter(image)
        if device == "cuda": torch.cuda.synchronize()
        durations.append(dict(image_id=r["image_id"], milliseconds=(time.perf_counter() - start) * 1000))
    import pandas as pd
    directory = run / "benchmarks" / head; directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(durations).to_csv(directory / f"{device}_samples.csv", index=False)
    ms = [d["milliseconds"] for d in durations]
    protocol = dict(dataset_fingerprint=dataset_fingerprint(manifest), image_ids=[r["image_id"] for r in rows],
                    seed=config["seed"], warmup_calls=config["benchmark_warmup"], batch_size=1,
                    precision="FP32", cpu_threads=config["cpu_threads"], tf32=False,
                    scope="decoded RGB image through preprocessing, inference and postprocessing; excludes image decoding",
                    runtime="PyTorch native, standard backend fusion, no compilation",
                    hardware=dict(cpu=cpu_name(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                  cuda=torch.version.cuda), torch=importlib.metadata.version("torch"))
    result = dict(status="ok", device=device, head=head, protocol=protocol,
                  checkpoint_sha256=completion["sha256"], parameters=sum(p.numel() for p in adapter.model_getter().parameters()), input_size=config["imgsz"], confidence_threshold=decision["threshold"],
                  nms_iou=config.get("nms_iou") if head != "end2end" else None,
                  mean_ms=float(np.mean(ms)), median_ms=float(np.median(ms)), p95_ms=float(np.percentile(ms, 95)),
                  throughput_fps=1000 / float(np.mean(ms)), samples=len(ms),
                  peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
                  recorded_at=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    save_json(directory / f"{device}.json", result)
    print(f"{head}/{device}: mean {result['mean_ms']:.2f} ms, p95 {result['p95_ms']:.2f} ms")
    return result


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    with contextlib.redirect_stdout(SafeWriter(sys.stdout)), contextlib.redirect_stderr(SafeWriter(sys.stderr)):
        try:
            result = globals()[job["action"]](job)
            save_json(job["result"], result)
        except BaseException as exc:
            traceback.print_exc()
            save_json(job["result"], dict(status="failed", error=redact(str(exc)), oom="out of memory" in str(exc).lower()))
            sys.exit(1)


if __name__ == "__main__":
    main()
