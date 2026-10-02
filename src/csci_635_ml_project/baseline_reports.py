"""Common detection metrics and local reports, independent of model backends."""
from __future__ import annotations

import contextlib
import copy
import html
import io
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baseline_data import dataset_fingerprint, records_for_split, save_json, sha256, verify_dataset

BACKGROUND = "background"


def box_iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(gt, predictions, threshold, iou=0.5):
    """RF-DETR v2's confidence-ordered, class-agnostic localization matching."""
    used, pairs, overlaps = set(), [], []
    tp = fp = 0
    for p in sorted(predictions, key=lambda p: -p["score"]):
        if p["score"] < threshold:
            continue
        candidates = [(box_iou(p["box"], g["box"]), i) for i, g in enumerate(gt) if i not in used]
        overlap, index = max(candidates, default=(0, None))
        if index is not None and overlap >= iou:
            used.add(index)
            g = gt[index]
            overlaps.append(overlap)
            pairs.append((g["class_name"], p["class_name"]))
            if p["category_id"] == g["category_id"]:
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
            pairs.append((BACKGROUND, p["class_name"]))
    pairs.extend((g["class_name"], BACKGROUND) for i, g in enumerate(gt) if i not in used)
    return tp, fp, len(gt) - tp, overlaps, pairs


def detection_metrics(rows, predictions, threshold, iou=0.5) -> dict:
    tp = fp = fn = 0
    overlaps = []
    for r in rows:
        a, b, c, values, _ = match_detections(r["gt"], predictions.get(r["image_id"], []), threshold, iou)
        tp += a; fp += b; fn += c
        overlaps.extend(values)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                mean_matched_iou=float(np.mean(overlaps)) if overlaps else None)


def select_threshold(rows, predictions) -> float:
    return max((detection_metrics(rows, predictions, float(t))["f1"], float(t))
               for t in np.linspace(0, 1, 101))[1]


def load_predictions(path, doc) -> dict:
    raw = json.loads(Path(path).read_text())
    expected = {im["id"] for im in doc["images"]}
    predictions = {int(k): v for k, v in raw.items()}
    if set(predictions) != expected:
        raise ValueError("Prediction image IDs differ from the complete split")
    names = {c["id"]: c["name"] for c in doc["categories"]}
    for detections in predictions.values():
        if len(detections) > 100:
            raise ValueError("Predictions exceed the shared maxDets=100 protocol")
        for p in detections:
            if p["category_id"] not in names or p["class_name"] != names[p["category_id"]]:
                raise ValueError("Prediction class mapping differs from ground truth")
            if len(p["box"]) != 4 or not all(math.isfinite(float(v)) for v in [*p["box"], p["score"]]):
                raise ValueError("Nonfinite prediction")
            if not 0 <= p["score"] <= 1 or p["box"][2] < p["box"][0] or p["box"][3] < p["box"][1]:
                raise ValueError("Invalid score or prediction box")
    return predictions


def coco_metrics(doc, predictions, directory: Path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        gt = COCO()
        gt.dataset = copy.deepcopy(doc)
        gt.dataset.setdefault("info", {})
        gt.createIndex()
        detections = []
        for image_id, items in predictions.items():
            for p in items:
                x, y, x2, y2 = p["box"]
                detections.append(dict(image_id=image_id, category_id=p["category_id"],
                                       bbox=[x, y, x2 - x, y2 - y], score=p["score"]))
        if detections:
            result = gt.loadRes(detections)
        else:
            result = COCO()
            result.dataset = dict(images=doc["images"], categories=doc["categories"], annotations=[])
            result.createIndex()
        evaluator = COCOeval(gt, result, "bbox")
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "coco_summary.txt").write_text(out.getvalue())
    save_json(directory / "predictions.coco.json", detections)
    names = ["mAP_50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    metrics = {k: float(v) if v >= 0 else None for k, v in zip(names, evaluator.stats)}
    per_class = {}
    precision = evaluator.eval["precision"]
    fig, ax = plt.subplots(figsize=(9, 6))
    for k, cid in enumerate(evaluator.params.catIds):
        name = next(c["name"] for c in doc["categories"] if c["id"] == cid)
        values = precision[:, :, k, 0, 2]
        at50 = values[0]
        per_class[name] = dict(AP50=float(at50[at50 >= 0].mean()) if (at50 >= 0).any() else None,
                               mAP_50_95=float(values[values >= 0].mean()) if (values >= 0).any() else None)
        ax.plot(evaluator.params.recThrs, np.where(at50 >= 0, at50, np.nan), label=name)
    ax.set(xlabel="Recall", ylabel="Interpolated precision", title="COCO per-class PR curves at IoU 0.50", ylim=(0, 1.02))
    ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=.2)
    figure(fig, directory / "per_class_pr.png")
    return metrics, per_class


def figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def html_report(directory: Path, title: str, content: dict):
    directory.mkdir(parents=True, exist_ok=True)
    images = "".join(f'<figure><img src="{html.escape(p.name)}"><figcaption>{html.escape(p.stem)}</figcaption></figure>'
                     for p in sorted(directory.glob("*.png")))
    links = "".join(f'<li><a href="{html.escape(str(p.relative_to(directory)))}">{html.escape(str(p.relative_to(directory)))}</a></li>'
                    for p in sorted(directory.rglob("*")) if p.is_file() and p.suffix in {".csv", ".json", ".txt", ".html"}
                    and p != directory / "report.html")
    text = html.escape(json.dumps(content, indent=2, allow_nan=False))
    (directory / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px}'
        'img{max-width:100%}pre{white-space:pre-wrap;background:#f3f5f7;padding:20px}figure{margin:24px 0}</style>'
        f'<title>{html.escape(title)}</title><h1>{html.escape(title)}</h1><pre>{text}</pre><ul>{links}</ul>{images}')


def error_gallery(rows, predictions, threshold, directory):
    from PIL import Image, ImageDraw
    errors, previews = [], []
    for r in rows:
        tp, fp, fn, _, _ = match_detections(r["gt"], predictions[r["image_id"]], threshold)
        if fp + fn:
            errors.append((fp + fn, r["image_id"], r))
    chosen = [r for _, _, r in sorted(errors, key=lambda x: (-x[0], x[1]))[:16]]
    # Save successful examples too, including splits with no errors.
    chosen.extend(r for r in rows[:8] if r["image_id"] not in {x["image_id"] for x in chosen})
    folder = directory / "examples"; folder.mkdir(exist_ok=True)
    for r in chosen:
        with Image.open(r["path"]) as opened:
            im = opened.convert("RGB")
        ratio = min(1, 900 / max(im.size))
        im = im.resize((round(im.width * ratio), round(im.height * ratio)))
        draw = ImageDraw.Draw(im)
        for items, color, prefix in [(r["gt"], "lime", "GT"),
                                    ([p for p in predictions[r["image_id"]] if p["score"] >= threshold], "red", "Pred")]:
            for p in items:
                b = [float(v) * ratio for v in p["box"]]
                draw.rectangle(b, outline=color, width=2)
                label = f"{prefix}: {p['class_name']}" + (f" {p['score']:.2f}" if "score" in p else "")
                draw.text((b[0], max(0, b[1] - 12)), label, fill=color, stroke_width=1, stroke_fill="black")
        name = f"image-{r['image_id']}.jpg"; im.save(folder / name)
        previews.append(f'<figure><img src="{name}"><figcaption>Image {r["image_id"]}; green GT, red prediction</figcaption></figure>')
    (folder / "gallery.html").write_text('<meta charset="utf-8"><style>img{max-width:900px}</style>' + "".join(previews))
    pd.DataFrame([dict(image_id=r["image_id"], errors=count) for count, _, r in errors]).to_csv(directory / "error_images.csv", index=False)


def evaluate_report(doc, rows, predictions, threshold, directory, metadata=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    operating = detection_metrics(rows, predictions, threshold)
    coco, ap = coco_metrics(doc, predictions, directory)
    labels = [c["name"] for c in doc["categories"]] + [BACKGROUND]
    indices = {name: i for i, name in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for r in rows:
        for truth, predicted in match_detections(r["gt"], predictions[r["image_id"]], threshold)[4]:
            cm[indices[truth], indices[predicted]] += 1
    per_class = []
    for i, name in enumerate(labels[:-1]):
        tp = int(cm[i, i]); fp = int(cm[:, i].sum()) - tp; fn = int(cm[i].sum()) - tp
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        per_class.append(dict(class_name=name, support=tp + fn, tp=tp, fp=fp, fn=fn,
                              precision=precision, recall=recall, f1=2 * precision * recall / (precision + recall) if precision + recall else 0,
                              **ap[name]))
    for title, values in [("confusion_counts", cm), ("confusion_normalized", np.divide(
            cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0))]:
        pd.DataFrame(values, index=labels, columns=labels).to_csv(directory / f"{title}.csv")
        fig, ax = plt.subplots(figsize=(12, 10)); ax.imshow(values, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=65, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set(xlabel="Predicted", ylabel="Ground truth", title=title.replace("_", " ").title())
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{values[i,j]:.2f}" if title.endswith("normalized") else str(values[i,j]),
                        ha="center", va="center", fontsize=6,
                        color="white" if values[i,j] > values.max() / 2 else "black")
        figure(fig, directory / f"{title}.png")
    pd.DataFrame(per_class).to_csv(directory / "per_class_metrics.csv", index=False)
    curve = pd.DataFrame([dict(threshold=float(t), **detection_metrics(rows, predictions, float(t))) for t in np.linspace(0, 1, 101)])
    curve.to_csv(directory / "confidence_curves.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in ("precision", "recall", "f1"):
        ax.plot(curve.threshold, curve[key], label=key)
    ax.axvline(threshold, color="black", ls="--", label="frozen validation threshold")
    ax.set(xlabel="Confidence", ylabel="Metric", ylim=(0, 1.02)); ax.legend(); ax.grid(alpha=.2)
    figure(fig, directory / "confidence_curves.png")
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(curve.recall, curve.precision)
    ax.set(xlabel="Micro recall", ylabel="Micro precision", title="Operating-threshold PR curve", ylim=(0, 1.02)); ax.grid(alpha=.2)
    figure(fig, directory / "micro_pr.png")
    metrics = dict(schema_version=1, confidence_threshold=threshold, iou_threshold=.5,
                   operating=operating, coco=coco, per_class=per_class,
                   macro={key: float(np.mean([p[key] for p in per_class])) for key in ("precision", "recall", "f1")},
                   metadata=metadata or {})
    save_json(directory / "metrics.json", metrics)
    error_gallery(rows, predictions, threshold, directory)
    html_report(directory, "Multiclass squirrel detection evaluation", metrics)
    return metrics


def evaluate_run(run: str | Path, data: str | Path, heads=("accuracy",)):
    run = Path(run)
    manifest, docs = verify_dataset(data)
    completion = json.loads((run / "training_complete.json").read_text())
    checkpoint = run / completion["best_checkpoint"]
    if sha256(checkpoint) != completion["sha256"]:
        raise ValueError("Best checkpoint changed after training")
    saved = json.loads((run / "dataset_manifest.json").read_text())
    if dataset_fingerprint(saved) != dataset_fingerprint(manifest):
        raise ValueError("Training and evaluation datasets differ")
    all_metrics = {}
    for head in heads:
        base = run / "reports" if head == "accuracy" else run / "reports" / head
        rows = {s: records_for_split(data, s, docs[s]) for s in ("valid", "test")}
        val_predictions = load_predictions(base / "valid/predictions.json", docs["valid"])
        threshold = select_threshold(rows["valid"], val_predictions)
        decision = dict(threshold=threshold, selection="maximum validation F1 at IoU 0.50; ties choose highest",
                        checkpoint_sha256=sha256(checkpoint), dataset_fingerprint=dataset_fingerprint(manifest), inference_head=head)
        path = run / ("decision_threshold.json" if head == "accuracy" else f"decision_threshold_{head}.json")
        if path.exists() and json.loads(path.read_text()) != decision:
            raise ValueError("Previously frozen decision differs; use a fresh run")
        save_json(path, decision)
        split_metrics = {}
        for split in ("valid", "test"):
            if sha256(checkpoint) != decision["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed before held-out evaluation")
            preds = val_predictions if split == "valid" else load_predictions(base / f"{split}/predictions.json", docs[split])
            split_metrics[split] = evaluate_report(docs[split], rows[split], preds, threshold, base / split,
                                                   dict(split=split, **decision))
        pd.DataFrame({s: m["coco"] for s, m in split_metrics.items()}).to_csv(base / "validation_test_comparison.csv")
        html_report(base, f"Final report: {head}", split_metrics)
        all_metrics[head] = split_metrics
    return all_metrics


def training_report(run):
    run = Path(run); directory = run / "reports"; directory.mkdir(exist_ok=True)
    history = pd.read_csv(run / "native/results.csv")
    history.columns = [str(c).strip() for c in history.columns]
    if history.epoch.iloc[0] == 0:
        history["epoch"] += 1
    history = history.drop_duplicates("epoch", keep="last")
    resource = run / "history.jsonl"
    if resource.is_file():
        resources = pd.DataFrame([json.loads(s) for s in resource.read_text().splitlines()])
        resources = resources.drop_duplicates("epoch", keep="last")
        history = history.merge(resources, on="epoch", how="left")
    history.to_csv(directory / "training_history.csv", index=False)
    for name, columns in [
        ("training_validation_losses", [c for c in history if "loss" in c]),
        ("validation_metrics", [c for c in history if "metrics/" in c]),
        ("learning_rates", [c for c in history if c.startswith("lr/")]),
        ("resources", [c for c in ("seconds", "peak_vram_gib") if c in history]),
    ]:
        if columns:
            fig, ax = plt.subplots(figsize=(12, 5))
            history.plot(x="epoch", y=columns, ax=ax); ax.grid(alpha=.2)
            ax.set_title(name.replace("_", " ").title())
            figure(fig, directory / f"{name}.png")
    return history


def dataset_report(run, data):
    manifest, docs = verify_dataset(data)
    directory = Path(run) / "reports"
    directory.mkdir(exist_ok=True)
    counts = []
    sizes = []
    for split, doc in docs.items():
        names = {c["id"]: c["name"] for c in doc["categories"]}
        for cid, name in names.items():
            annotations = [a for a in doc["annotations"] if a["category_id"] == cid]
            counts.append(dict(split=split, class_name=name, images=len({a["image_id"] for a in annotations}), annotations=len(annotations)))
        sizes.extend(dict(split=split, width=im["width"], height=im["height"]) for im in doc["images"])
    table = pd.DataFrame(counts)
    table.to_csv(directory / "dataset_class_counts.csv", index=False)
    fig, ax = plt.subplots(figsize=(14, 6))
    table.pivot(index="class_name", columns="split", values="images").plot.bar(ax=ax)
    ax.set(ylabel="Images containing class", xlabel="Class", title="Audited dataset: unchanged natural split distributions")
    ax.tick_params(axis="x", labelsize=8)
    figure(fig, directory / "dataset_class_distribution.png")
    sizes = pd.DataFrame(sizes); sizes.to_csv(directory / "image_dimensions.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    for split in docs:
        part = sizes[sizes.split == split]
        ax.scatter(part.width, part.height, s=5, alpha=.15, label=split)
    ax.set(xlabel="Original image width", ylabel="Original image height", title="Audited image dimensions"); ax.legend()
    figure(fig, directory / "dataset_dimensions.png")
    html_report(directory, "Dataset verification and EDA", manifest)
    return table
