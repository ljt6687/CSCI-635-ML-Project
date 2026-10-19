#!/usr/bin/env python3
"""Build a stratified, review-only PyLabel draft from v2 model predictions.

This script never promotes model predictions to verified training annotations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET


DEFAULT_PASSED_RATE = 0.10
DEFAULT_PROBABLE_RATE = 0.30
DEFAULT_PASSED_MIN = 2
DEFAULT_PROBABLE_MIN = 5
ROUTINE_FLAG = "species_label_provisional_per_box"
DRAFT_SCORE_FLOOR = 0.20
SMALL_BOX_AREA_FRACTION = 0.005
MIN_IMAGES_PER_CLASS = 600
BUCKETS = {"passed", "probably_passed", "recheck_needed"}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = row.get("image_id")
            if not isinstance(image_id, str) or not image_id or image_id in seen:
                raise ValueError(f"{path}:{number}: missing or repeated image_id")
            seen.add(image_id)
            rows.append(row)
    return rows


def sample_key(seed: str, image_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{image_id}".encode()).hexdigest()


def select_review(
    manifest: list[dict],
    predictions: list[dict],
    *,
    seed: str = "squirrel-v2-review-1",
    passed_rate: float = DEFAULT_PASSED_RATE,
    probable_rate: float = DEFAULT_PROBABLE_RATE,
) -> list[dict]:
    """Select every uncertain/flagged image plus stratified confidence samples."""
    if not 0 <= passed_rate <= probable_rate <= 1:
        raise ValueError("sample rates must satisfy 0 <= passed <= probable <= 1")
    by_id = {row["image_id"]: row for row in predictions}
    manifest_ids = {row["image_id"] for row in manifest}
    if set(by_id) != manifest_ids or len(predictions) != len(manifest):
        missing = sorted(manifest_ids - set(by_id))
        extra = sorted(set(by_id) - manifest_ids)
        raise ValueError(f"prediction coverage mismatch: missing={missing[:5]}, extra={extra[:5]}")
    sha_values = {row.get("model_sha256") for row in predictions}
    if len(sha_values) != 1 or None in sha_values:
        raise ValueError("predictions have missing or mixed model hashes")

    selected: dict[str, set[str]] = defaultdict(set)
    strata: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for item in manifest:
        image_id = item["image_id"]
        pred = by_id[image_id]
        for field in ("image_path", "source", "species_label"):
            if item.get(field) != pred.get(field):
                raise ValueError(f"manifest/prediction mismatch for {image_id}: {field}")
        bucket = pred.get("bucket")
        if bucket not in BUCKETS:
            raise ValueError(f"unexpected bucket for {image_id}: {bucket!r}")
        if bucket == "recheck_needed":
            selected[image_id].add("all_recheck_needed")
        if item["source"] == "meyer_trailcam":
            selected[image_id].add("all_meyer_xml_comparison")
        if pred.get("error"):
            selected[image_id].add("inference_error")
        material = [det for det in pred.get("detections", []) if det.get("score", 0) >= DRAFT_SCORE_FLOOR]
        size = pred.get("image_size") or {}
        width, height = size.get("width"), size.get("height")
        if width and height and any(
            (det["xyxy"][2] - det["xyxy"][0]) * (det["xyxy"][3] - det["xyxy"][1])
            / (width * height) < SMALL_BOX_AREA_FRACTION
            for det in material
        ):
            selected[image_id].add("geometry:any_material_box_below_0.5pct")
        if len(material) > 3:
            selected[image_id].add("geometry:more_than_three_material_boxes")
        for flag in pred.get("review_flags", []):
            if flag == ROUTINE_FLAG:
                continue
            if flag == "multiple_detections" and len(material) < 2:
                continue  # Preserve low-score extras, but do not turn every image into mandatory review.
            if flag == "possible_poor_localization" and not any(
                det.get("localization_flags") for det in material
            ):
                continue
            selected[image_id].add(f"flag:{flag}")
        strata[(bucket, item["source"], item["species_label"])].append(image_id)

    for (bucket, _source, _species), ids in sorted(strata.items()):
        if bucket == "recheck_needed":
            continue
        rate, minimum = (
            (passed_rate, DEFAULT_PASSED_MIN)
            if bucket == "passed"
            else (probable_rate, DEFAULT_PROBABLE_MIN)
        )
        count = min(len(ids), max(math.ceil(len(ids) * rate), minimum))
        reason = f"stratified_{bucket}_sample"
        for image_id in sorted(ids, key=lambda value: sample_key(seed, value))[:count]:
            selected[image_id].add(reason)

    queue = []
    for item in manifest:
        image_id = item["image_id"]
        if image_id not in selected:
            continue
        pred = by_id[image_id]
        queue.append({
            "image_id": image_id,
            "image_path": item["image_path"],
            "source": item["source"],
            "species_label": item["species_label"],
            "bucket": pred["bucket"],
            "bucket_reason": pred.get("bucket_reason"),
            "top_score": pred.get("top_score"),
            "review_flags": pred.get("review_flags", []),
            "selection_reasons": sorted(selected[image_id]),
            "error": pred.get("error"),
            "image_size": pred.get("image_size"),
            "detections": pred.get("detections", []),
        })
    priority = {"recheck_needed": 0, "probably_passed": 1, "passed": 2}
    queue.sort(key=lambda row: (priority[row["bucket"]], row["species_label"], row["source"], row["image_id"]))
    return queue


def build_review_coco(queue: list[dict], output_dir: Path, repo_root: Path) -> tuple[dict, dict[str, int]]:
    """Create a COCO draft for PyLabel, referencing original images in place."""
    names = sorted({row["species_label"] for row in queue})
    category_id = {name: index for index, name in enumerate(names, 1)}
    images, annotations, coco_ids = [], [], {}
    for row in queue:
        size = row.get("image_size")
        if not isinstance(size, dict) or not size.get("width") or not size.get("height"):
            continue  # Still present in the queue; the source image needs repair first.
        path = (repo_root / row["image_path"]).resolve()
        if not path.is_relative_to(repo_root) or not path.is_file():
            raise ValueError(f"missing/out-of-repo image: {row['image_path']}")
        image_id = len(images) + 1
        coco_ids[row["image_id"]] = image_id
        annotation_basis = "v1_model_provisional"
        draft_boxes = [det["xyxy"] for det in row["detections"]
                       if det["score"] >= DRAFT_SCORE_FLOOR]
        if row["source"] == "meyer_trailcam":
            xml_path = path.with_suffix(".xml")
            if not xml_path.is_file():
                raise ValueError(f"missing Meyer XML reference: {xml_path}")
            xml_root = ET.parse(xml_path).getroot()
            if (xml_root.findtext("filename") != path.name
                    or int(xml_root.findtext("size/width")) != size["width"]
                    or int(xml_root.findtext("size/height")) != size["height"]):
                raise ValueError(f"Meyer XML image mismatch: {xml_path}")
            draft_boxes = []
            for obj in xml_root.findall("object"):
                if (obj.findtext("name") or "").casefold() != "squirrel":
                    raise ValueError(f"unexpected Meyer XML class: {xml_path}")
                draft_boxes.append([float(obj.findtext(f"bndbox/{coord}"))
                                    for coord in ("xmin", "ymin", "xmax", "ymax")])
            if not draft_boxes:
                raise ValueError(f"Meyer XML has no squirrel boxes: {xml_path}")
            annotation_basis = "meyer_voc_reference"
            row["reference_annotation_path"] = xml_path.relative_to(repo_root).as_posix()
            row["reference_boxes"] = draft_boxes
        row["draft_annotation_basis"] = annotation_basis
        images.append({
            "id": image_id,
            "file_name": path.name,
            "folder": os.path.relpath(path.parent, output_dir),
            "width": int(size["width"]),
            "height": int(size["height"]),
            "annotation_basis": annotation_basis,
        })
        for x1, y1, x2, y2 in draft_boxes:
            if not (0 <= x1 < x2 <= size["width"] and 0 <= y1 < y2 <= size["height"]):
                raise ValueError(f"invalid box in {row['image_id']}")
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": category_id[row["species_label"]],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })
    return {
        "info": {"description": "REVIEW DRAFT ONLY: provisional RF-DETR boxes and image-level species labels"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index, "name": name} for name, index in category_id.items()],
    }, coco_ids


def run(manifest_path: Path, predictions_path: Path, output_dir: Path, repo_root: Path, seed: str,
        passed_rate: float, probable_rate: float) -> dict:
    manifest = read_jsonl(manifest_path)
    predictions = read_jsonl(predictions_path)
    queue = select_review(manifest, predictions, seed=seed, passed_rate=passed_rate,
                          probable_rate=probable_rate)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"review directory exists; preserving any human edits: {output_dir}")
    coco, coco_ids = build_review_coco(queue, output_dir, repo_root.resolve())
    output_dir.mkdir(parents=True)
    with (output_dir / "review_queue.jsonl").open("w", encoding="utf-8") as stream:
        for row in queue:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "review_candidates.coco.json").write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "review_decisions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image_id", "coco_image_id", "source", "bucket",
                                                    "original_species_label", "decision", "final_species_label", "notes"))
        writer.writeheader()
        for row in queue:
            writer.writerow({
                "image_id": row["image_id"], "coco_image_id": coco_ids.get(row["image_id"], ""),
                "source": row["source"], "bucket": row["bucket"],
                "original_species_label": row["species_label"],
                "decision": "", "final_species_label": "", "notes": "",
            })
    candidate_counts = Counter(row["species_label"] for row in predictions
                               if row["bucket"] in {"passed", "probably_passed"}
                               and row.get("detections") and not row.get("error"))
    summary = {
        "input_images": len(manifest),
        "review_images": len(queue),
        "review_coco_images": len(coco["images"]),
        "review_coco_boxes": len(coco["annotations"]),
        "by_bucket_total": dict(Counter(row["bucket"] for row in predictions)),
        "by_bucket_selected": dict(Counter(row["bucket"] for row in queue)),
        "by_source_total": dict(Counter(row["source"] for row in manifest)),
        "by_source_selected": dict(Counter(row["source"] for row in queue)),
        "by_species_total": dict(Counter(row["species_label"] for row in manifest)),
        "by_species_selected": dict(Counter(row["species_label"] for row in queue)),
        "minimum_images_per_class": MIN_IMAGES_PER_CLASS,
        "pre_review_candidate_images_by_species": dict(candidate_counts),
        "pre_review_shortfall_by_species": {label: max(0, MIN_IMAGES_PER_CLASS - candidate_counts[label])
                                             for label in sorted({row["species_label"] for row in manifest})},
        "passed_rate": passed_rate,
        "probable_rate": probable_rate,
        "seed": seed,
        "model_sha256": predictions[0]["model_sha256"] if predictions else None,
        "draft_score_floor": DRAFT_SCORE_FLOOR,
        "small_box_review_area_fraction": SMALL_BOX_AREA_FRACTION,
        "status": "pending_human_review",
    }
    (output_dir / "review_plan.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("output/v2_annotation/source_manifest.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("output/v2_annotation/predictions.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/v2_annotation/review"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", default="squirrel-v2-review-1")
    parser.add_argument("--passed-rate", type=float, default=DEFAULT_PASSED_RATE)
    parser.add_argument("--probable-rate", type=float, default=DEFAULT_PROBABLE_RATE)
    args = parser.parse_args()
    summary = run(args.manifest, args.predictions, args.output_dir, args.repo_root, args.seed,
                  args.passed_rate, args.probable_rate)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
