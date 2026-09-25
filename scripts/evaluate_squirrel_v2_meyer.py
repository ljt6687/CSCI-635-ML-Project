#!/usr/bin/env python3
"""Compare v1 RF-DETR detections with Meyer Trailcam Pascal VOC reference boxes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

from PIL import Image


SOURCE = "meyer_trailcam"
SOURCE_RELATIVE = Path("data/squirrel_sources/meyer_trailcam/raw/DatasetSamples/Squirrel")
THRESHOLDS = (0.20, 0.45, 0.50, 0.80)
IOU_THRESHOLD = 0.50
EXPECTED_IMAGES = 30


def _integer(value: str | None, field: str, path: Path) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid or missing {field}: {value!r}") from exc


def _flag(value: str | None, field: str, path: Path) -> bool | None:
    if value is None:
        return None
    if value not in {"0", "1"}:
        raise ValueError(f"{path}: {field} must be 0 or 1, got {value!r}")
    return value == "1"


def parse_voc(xml_path: Path, image_path: Path) -> dict[str, Any]:
    """Read one VOC file, checking its filename, dimensions, and object boxes."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{xml_path}: invalid XML: {exc}") from exc
    if root.tag != "annotation":
        raise ValueError(f"{xml_path}: expected annotation root")
    if root.findtext("filename") != image_path.name:
        raise ValueError(f"{xml_path}: XML filename does not match {image_path.name}")
    width = _integer(root.findtext("size/width"), "size/width", xml_path)
    height = _integer(root.findtext("size/height"), "size/height", xml_path)
    if width <= 0 or height <= 0:
        raise ValueError(f"{xml_path}: image dimensions must be positive")
    with Image.open(image_path) as image:
        if image.size != (width, height):
            raise ValueError(f"{xml_path}: XML size {(width, height)} differs from image size {image.size}")
    objects = []
    for index, obj in enumerate(root.findall("object")):
        name = (obj.findtext("name") or "").strip()
        if name.casefold() != "squirrel":
            raise ValueError(f"{xml_path}: object {index} has unsupported name {name!r}")
        box = [_integer(obj.findtext(f"bndbox/{field}"), field, xml_path)
               for field in ("xmin", "ymin", "xmax", "ymax")]
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"{xml_path}: object {index} box {box} is outside {width}x{height}")
        objects.append({"name": name, "xyxy": box,
                        "difficult": _flag(obj.findtext("difficult"), "difficult", xml_path),
                        "truncated": _flag(obj.findtext("truncated"), "truncated", xml_path)})
    if not objects:
        raise ValueError(f"{xml_path}: no squirrel objects")
    return {"width": width, "height": height, "objects": objects}


def load_references(repo_root: Path, expected_images: int = EXPECTED_IMAGES) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source_dir = repo_root / SOURCE_RELATIVE
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Meyer squirrel directory not found: {source_dir}")
    jpgs = sorted(p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})
    originals = [p for p in jpgs if not p.stem.casefold().endswith("-rendered")]
    rendered = [p for p in jpgs if p.stem.casefold().endswith("-rendered")]
    xmls = sorted(p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".xml")
    if len(originals) != expected_images:
        raise ValueError(f"Meyer source has {len(originals)} original JPGs; expected {expected_images}")
    if len(xmls) != expected_images:
        raise ValueError(f"Meyer source has {len(xmls)} XML files; expected {expected_images}")
    expected_xmls = {p.with_suffix(".xml") for p in originals}
    actual_xmls = set(xmls)
    if expected_xmls != actual_xmls:
        missing = sorted(str(p.relative_to(source_dir)) for p in expected_xmls - actual_xmls)
        extra = sorted(str(p.relative_to(source_dir)) for p in actual_xmls - expected_xmls)
        raise ValueError(f"Meyer XML/image mismatch: missing={missing}, extra={extra}")
    references = []
    for image in originals:
        relative = image.relative_to(repo_root).as_posix()
        source_relative = image.relative_to(repo_root / "data/squirrel_sources/meyer_trailcam").as_posix()
        references.append({"image_id": f"{SOURCE}:{source_relative}", "image_path": relative,
                           "source": SOURCE, "species_label": "generic_squirrel",
                           "xml_path": image.with_suffix(".xml").relative_to(repo_root).as_posix(),
                           **parse_voc(image.with_suffix(".xml"), image)})
    return references, {"original_jpgs": len(originals), "rendered_jpgs_excluded": len(rendered),
                        "xml_files": len(xmls)}


def manifest_audit(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "output/v2_annotation/source_manifest.jsonl"
    if not path.is_file():
        return {"path": str(path), "present": False, "rendered_rows": []}
    rendered = []
    meyer_count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest line {line_number}: invalid JSON: {exc}") from exc
        if row.get("source") == SOURCE:
            meyer_count += 1
            if "-rendered" in str(row.get("image_path", "")).casefold():
                rendered.append({"line": line_number, "image_path": row.get("image_path")})
    return {"path": str(path), "present": True, "meyer_rows": meyer_count,
            "rendered_rows": rendered}


def _validate_box(box: Any, width: int, height: int, context: str) -> list[float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"{context}: xyxy must contain four numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        raise ValueError(f"{context}: xyxy contains a nonfinite or nonnumeric coordinate")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{context}: xyxy {box} is outside {width}x{height}")
    return [float(v) for v in box]


def load_predictions(path: Path, references: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    expected = {ref["image_path"]: ref for ref in references}
    rows: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"predictions line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"predictions line {line_number}: expected object")
            if row.get("source") != SOURCE:
                continue
            image_path = row.get("image_path")
            image_id = row.get("image_id")
            if image_id in seen_ids:
                raise ValueError(f"predictions line {line_number}: duplicate Meyer image_id {image_id!r}")
            if image_path in rows:
                raise ValueError(f"predictions line {line_number}: duplicate Meyer image_path {image_path!r}")
            if not isinstance(image_id, str) or not isinstance(image_path, str):
                raise ValueError(f"predictions line {line_number}: image_id and image_path must be strings")
            seen_ids.add(image_id)
            if image_path not in expected:
                raise ValueError(f"predictions line {line_number}: unexpected Meyer image_path {image_path!r}; rendered JPGs are excluded")
            ref = expected[image_path]
            if image_id != ref["image_id"]:
                raise ValueError(f"predictions line {line_number}: image_id does not match {image_path!r}")
            if row.get("species_label") != "generic_squirrel":
                raise ValueError(f"predictions line {line_number}: Meyer species_label must be generic_squirrel")
            if row.get("error"):
                raise ValueError(f"predictions line {line_number}: inference error for {image_id}: {row['error']}")
            if not isinstance(row.get("bucket"), str) or not row["bucket"]:
                raise ValueError(f"predictions line {line_number}: missing bucket")
            if not isinstance(row.get("detections"), list):
                raise ValueError(f"predictions line {line_number}: detections must be a list")
            detections = []
            for index, detection in enumerate(row["detections"]):
                context = f"predictions line {line_number} detection {index}"
                if not isinstance(detection, dict):
                    raise ValueError(f"{context}: expected object")
                score = detection.get("score")
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"{context}: score must be finite and between 0 and 1")
                detections.append({"xyxy": _validate_box(detection.get("xyxy"), ref["width"], ref["height"], context),
                                   "score": float(score)})
            rows[image_path] = {"image_id": image_id, "image_path": image_path,
                                "bucket": row["bucket"], "detections": detections}
    missing = sorted(set(expected) - set(rows))
    if missing:
        raise ValueError(f"missing {len(missing)} Meyer prediction rows: {missing}")
    if len(rows) != EXPECTED_IMAGES:
        raise ValueError(f"expected {EXPECTED_IMAGES} Meyer prediction rows; got {len(rows)}")
    return rows


def iou(a: list[float], b: list[float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def compare_image(reference: dict[str, Any], prediction: dict[str, Any], threshold: float) -> dict[str, Any]:
    detections = prediction["detections"]
    order = sorted((i for i, d in enumerate(detections) if d["score"] >= threshold),
                   key=lambda i: (-detections[i]["score"], i))
    unmatched = set(range(len(reference["objects"])))
    matches = []
    false_positives = []
    for index in order:
        candidates = [(iou(detections[index]["xyxy"], reference["objects"][j]["xyxy"]), j)
                      for j in sorted(unmatched)]
        best_iou, best_j = max(candidates, key=lambda pair: (pair[0], -pair[1])) if candidates else (0.0, None)
        if best_j is not None and best_iou >= IOU_THRESHOLD:
            unmatched.remove(best_j)
            matches.append({"detection_index": index, "reference_index": best_j, "iou": best_iou})
        else:
            false_positives.append(index)
    return {"tp": len(matches), "fp": len(false_positives), "fn": len(unmatched),
            "matches": matches, "false_positive_detection_indices": false_positives,
            "missed_reference_indices": sorted(unmatched)}


def _summary(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["tp"] for row in comparisons)
    fp = sum(row["fp"] for row in comparisons)
    fn = sum(row["fn"] for row in comparisons)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    matched_ious = [match["iou"] for row in comparisons for match in row["matches"]]
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": f1, "mean_matched_iou": sum(matched_ious) / len(matched_ious) if matched_ious else None}


def evaluate(predictions: Path, output_dir: Path, repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    references, inventory = load_references(repo_root)
    rows = load_predictions(predictions, references)
    audit = manifest_audit(repo_root)
    per_image = []
    for ref in references:
        prediction = rows[ref["image_path"]]
        comparisons = {f"{threshold:.2f}": compare_image(ref, prediction, threshold) for threshold in THRESHOLDS}
        per_image.append({"image_id": ref["image_id"], "image_path": ref["image_path"],
                          "xml_path": ref["xml_path"], "source": SOURCE, "species_label": "generic_squirrel",
                          "width": ref["width"], "height": ref["height"], "reference_objects": ref["objects"],
                          "bucket": prediction["bucket"], "detections": prediction["detections"],
                          "thresholds": comparisons})
    aggregates = {key: _summary([row["thresholds"][key] for row in per_image])
                  for key in (f"{threshold:.2f}" for threshold in THRESHOLDS)}
    default = "0.45"
    objects = [obj for ref in references for obj in ref["objects"]]
    report = {"source": SOURCE, "species_label": "generic_squirrel", "image_count": len(per_image),
              "reference_object_count": sum(len(ref["objects"]) for ref in references),
              "reference_flags": {"difficult_true": sum(obj["difficult"] is True for obj in objects),
                                  "truncated_true": sum(obj["truncated"] is True for obj in objects),
                                  "difficult_missing": sum(obj["difficult"] is None for obj in objects),
                                  "truncated_missing": sum(obj["truncated"] is None for obj in objects)},
              "source_inventory": inventory, "manifest_audit": audit,
              "bucket_distribution": dict(sorted(Counter(row["bucket"] for row in per_image).items())),
              "score_thresholds": list(THRESHOLDS), "frozen_v1_score_threshold": 0.45,
              "iou_match_threshold": IOU_THRESHOLD, "metrics": aggregates,
              "per_image_misses_and_false_positives_at_0.45": [
                  {"image_id": row["image_id"], "image_path": row["image_path"],
                   "missed_reference_indices": row["thresholds"][default]["missed_reference_indices"],
                   "missed_references": [row["reference_objects"][i]
                                         for i in row["thresholds"][default]["missed_reference_indices"]],
                   "false_positive_detection_indices": row["thresholds"][default]["false_positive_detection_indices"],
                   "false_positive_detections": [row["detections"][i]
                                                 for i in row["thresholds"][default]["false_positive_detection_indices"]]}
                  for row in per_image if row["thresholds"][default]["fn"] or row["thresholds"][default]["fp"]],
              "caveat": "Only 30 selected Meyer trailcam images. XML boxes are reference annotations, not infallible ground truth; these metrics do not establish external generalization.",
              "box_convention": "VOC coordinates compared as continuous xyxy without an inclusive-pixel +1 adjustment."}
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to((repo_root / "data").resolve()):
        raise ValueError("output directory must not be inside source data")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "meyer_per_image.jsonl"
    report_path = output_dir / "meyer_report.json"
    if comparison_path.exists() or report_path.exists():
        raise FileExistsError("comparison output already exists; choose a fresh output directory")
    with comparison_path.open("x", encoding="utf-8") as stream:
        for row in per_image:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Full inference JSONL, including all Meyer images")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.predictions, args.output_dir, args.repo_root)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps({"image_count": report["image_count"], "metrics_at_0.45": report["metrics"]["0.45"],
                      "output_dir": str(args.output_dir.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
