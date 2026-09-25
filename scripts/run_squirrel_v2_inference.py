#!/usr/bin/env python3
"""Run provisional, review-first squirrel detection over a v2 JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from PIL import Image


DEFAULT_FLOOR = 0.05
REQUIRED_FIELDS = ("image_id", "image_path", "source", "species_label", "label_basis")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_digest(row: dict[str, Any]) -> str:
    encoded = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    ids = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or any(key not in row for key in REQUIRED_FIELDS):
                raise ValueError(f"manifest line {line_number}: expected object with {REQUIRED_FIELDS}")
            if not isinstance(row["image_id"], str) or not row["image_id"]:
                raise ValueError(f"manifest line {line_number}: image_id must be a nonempty string")
            if not isinstance(row["image_path"], str) or not row["image_path"]:
                raise ValueError(f"manifest line {line_number}: image_path must be a nonempty string")
            if row["image_id"] in ids:
                raise ValueError(f"manifest line {line_number}: duplicate image_id {row['image_id']!r}")
            ids.add(row["image_id"])
            rows.append(row)
    return rows


def resolve_image(repo_root: Path, image_path: str) -> Path:
    relative = Path(image_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("image_path must be repo-relative without parent traversal")
    resolved = (repo_root / relative).resolve()
    if not resolved.is_relative_to(repo_root):
        raise ValueError("image_path resolves outside repo root")
    return resolved


def bucket_for_score(score: float | None, error: str | None = None) -> tuple[str, str]:
    if error:
        return "recheck_needed", "error"
    if score is None:
        return "recheck_needed", "no_detection_above_floor"
    if score > 0.80:
        return "passed", "score_above_0.80"
    if score >= 0.50:
        return "probably_passed", "score_0.50_to_0.80"
    if score > 0.20:
        return "recheck_needed", "score_above_0.20_below_0.50"
    return "recheck_needed", "score_at_or_below_0.20"


def extract_detections(prediction: Any, width: int, height: int, species_label: Any) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Keep valid, in-image boxes and surface invalid model output for review."""
    boxes = prediction.xyxy
    scores = prediction.confidence
    classes = prediction.class_id
    if scores is None or classes is None or len(boxes) != len(scores) or len(boxes) != len(classes):
        raise ValueError("model returned inconsistent detection arrays")
    detections = []
    rejected = 0
    flags: set[str] = set()
    for box, raw_score, raw_class in zip(boxes, scores, classes):
        score = float(raw_score)
        class_id = int(raw_class)
        coords = [float(value) for value in box]
        if len(coords) != 4 or not all(math.isfinite(value) for value in (*coords, score)) or not 0 <= score <= 1:
            rejected += 1
            flags.add("invalid_model_detection")
            continue
        if class_id != 0:
            rejected += 1
            flags.add("unexpected_model_class")
            continue
        bounded = [min(max(coords[0], 0.0), width), min(max(coords[1], 0.0), height),
                   min(max(coords[2], 0.0), width), min(max(coords[3], 0.0), height)]
        localization_flags = []
        if bounded != coords:
            localization_flags.append("out_of_bounds_clipped")
        x1, y1, x2, y2 = bounded
        if x2 <= x1 or y2 <= y1:
            rejected += 1
            flags.add("invalid_model_detection")
            continue
        if (x2 - x1) < max(8, width * 0.01) or (y2 - y1) < max(8, height * 0.01):
            localization_flags.append("tiny_box")
        if x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
            localization_flags.append("edge_box")
        if localization_flags:
            flags.add("possible_poor_localization")
        detections.append({
            "xyxy": bounded,
            "score": score,
            "class_id": 0,
            "class_name": "SQUIRREL",
            "species_label": species_label,
            "species_label_provisional": True,
            "localization_flags": localization_flags,
        })
    detections.sort(key=lambda item: (-item["score"], item["xyxy"]))
    return detections, rejected, sorted(flags)


def infer_row(row: dict[str, Any], repo_root: Path, model: Any, model_sha256: str,
              settings: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)  # Preserve source, species, label basis, and all provenance fields.
    result.update({
        "manifest_row_sha256": row_digest(row),
        "image_size": None,
        "detections": [],
        "rejected_detection_count": 0,
        "model_sha256": model_sha256,
        "model_settings": settings,
        "top_score": None,
        "bucket": "recheck_needed",
        "bucket_reason": "error",
        "review_flags": [],
        "error": None,
    })
    try:
        path = resolve_image(repo_root, row["image_path"])
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
        result["image_size"] = {"width": width, "height": height}
        prediction = model.predict(rgb, threshold=settings["detection_floor"], include_source_image=False)
        detections, rejected, flags = extract_detections(prediction, width, height, row["species_label"])
        result["detections"] = detections
        result["rejected_detection_count"] = rejected
        result["top_score"] = detections[0]["score"] if detections else None
        if detections:
            flags.append("species_label_provisional_per_box")
        else:
            flags.append("no_detection_review")
        if len(detections) > 1:
            flags.append("multiple_detections")
        if rejected:
            flags.append("invalid_model_detection_review")
        result["review_flags"] = sorted(set(flags))
        result["bucket"], result["bucket_reason"] = bucket_for_score(result["top_score"])
        if rejected:
            result["bucket"] = "recheck_needed"
            result["bucket_reason"] = "invalid_model_detection"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["bucket"], result["bucket_reason"] = bucket_for_score(None, result["error"])
        result["review_flags"] = sorted(set(result["review_flags"] + ["inference_error_review"] +
                                           (["no_detection_review"] if not result["detections"] else [])))
    return result


def load_model(weights: Path, device: str) -> Any:
    import torch
    from rfdetr import RFDETRMedium

    model = RFDETRMedium(pretrain_weights=str(weights), device=device)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model.inference(compile=(device == "cuda"), dtype=dtype)
    return model


def read_completed(path: Path, rows: list[dict[str, Any]], sha256: str,
                   settings: dict[str, Any]) -> list[dict[str, Any]]:
    completed = []
    if not path.exists():
        return completed
    with path.open("rb+") as stream:
        line_number = 0
        while True:
            position = stream.tell()
            raw_line = stream.readline()
            if not raw_line:
                break
            line_number += 1
            if not raw_line.endswith(b"\n"):
                if path.suffix == ".partial":
                    stream.truncate(position)  # Drop only an interrupted trailing write.
                    break
                raise ValueError(f"resume file {path} line {line_number} is incomplete")
            try:
                result = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"resume file {path} line {line_number} is invalid") from exc
            if line_number > len(rows) or not isinstance(result, dict) or result.get("image_id") != rows[line_number - 1]["image_id"]:
                raise ValueError(f"resume file {path} does not match manifest order at line {line_number}")
            if (result.get("manifest_row_sha256") != row_digest(rows[line_number - 1]) or
                    result.get("model_sha256") != sha256 or result.get("model_settings") != settings):
                raise ValueError(f"resume file {path} has a different input or model at line {line_number}")
            completed.append(result)
    return completed


def run(manifest: Path, output: Path, repo_root: Path, weights: Path, device: str,
        limit: int | None = None, resume: bool = False, model_factory: Any = load_model) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output = output.resolve()
    allowed_output = (repo_root / "output" / "v2_annotation").resolve()
    if not output.is_relative_to(allowed_output) or output == allowed_output:
        raise ValueError(f"output must be a file under {allowed_output}")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cuda", "cpu"}:
        raise ValueError("--device must be auto, cuda, or cpu")
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
    rows = read_manifest(manifest)
    if limit is not None:
        rows = rows[:limit]
    metadata = json.loads((repo_root / "output" / "model_weights" / "model_metadata.json").read_text())
    if metadata.get("model") != "RFDETRMedium" or metadata.get("class_mapping") != {"0": "SQUIRREL"}:
        raise ValueError("model metadata is not the expected one-class RFDETRMedium checkpoint")
    sha256 = sha256_file(weights)
    if sha256 != metadata.get("sha256"):
        raise ValueError(f"checkpoint SHA256 mismatch: {sha256} != {metadata.get('sha256')}")
    settings = {
        "model": "RFDETRMedium",
        "class_mapping": {"0": "SQUIRREL"},
        "resolution": metadata.get("resolution"),
        "device": device,
        "dtype": "float16" if device == "cuda" else "float32",
        "inference_optimization": "model.inference(compile=True, dtype=torch.float16)" if device == "cuda"
                                  else "model.inference(compile=False, dtype=torch.float32)",
        "detection_floor": DEFAULT_FLOOR,
        "score_semantics": "single RF-DETR detection score; no separate box confidence",
    }
    partial = output.with_name(output.name + ".partial")
    if not resume and (output.exists() or partial.exists()):
        raise FileExistsError("output or partial file exists; use --resume or choose another output")
    source = partial if partial.exists() else output
    completed = read_completed(source, rows, sha256, settings) if resume else []
    output.parent.mkdir(parents=True, exist_ok=True)
    if source == output and output.exists() and len(completed) < len(rows):
        shutil.copyfile(output, partial)
    resumed_count = len(completed)
    if len(completed) < len(rows):
        model = model_factory(weights, device)
        with partial.open("a", encoding="utf-8") as stream:
            for row in rows[len(completed):]:
                result = infer_row(row, repo_root, model, sha256, settings)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                completed.append(result)
    if not output.exists() and not partial.exists():
        partial.touch()
    if partial.exists():
        os.replace(partial, output)
    counts: dict[str, int] = {}
    for result in completed:
        counts[result["bucket"]] = counts.get(result["bucket"], 0) + 1
    return {"output": str(output), "rows": len(completed), "buckets": counts,
            "errors": sum(bool(row["error"]) for row in completed), "resumed_rows": resumed_count}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    output = args.output or repo_root / "output" / "v2_annotation" / "squirrel_v2_inference.jsonl"
    weights = args.weights or repo_root / "output" / "model_weights" / "best_squirrel_rfdetr_medium.pth"
    try:
        summary = run(args.manifest, output, repo_root, weights, args.device, args.limit, args.resume)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
