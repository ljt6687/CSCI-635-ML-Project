#!/usr/bin/env python3
"""Reproducible, read-only sampling of v1 detections for bucket quality review."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFERENCE = ROOT / "output/v2_annotation/squirrel_v2_inference.jsonl"
CATEGORIES = ROOT / "output/v2_annotation/review/review_candidates.coco.json"
OUTPUT = ROOT / "output/v2_annotation/bucket_samples"
BUCKETS = ("passed", "probably_passed")
SAMPLE_SIZE = 100
SEED = 20260925
MIN_SCORE = 0.20


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(inference: Path = INFERENCE, categories_path: Path = CATEGORIES,
          output: Path = OUTPUT, seed: int = SEED) -> dict:
    records = list(read_jsonl(inference))
    by_bucket = {bucket: sorted((r for r in records if r.get("bucket") == bucket),
                                key=lambda row: row["image_id"]) for bucket in BUCKETS}
    categories = json.loads(categories_path.read_text(encoding="utf-8"))["categories"]
    category_id = {category["name"]: category["id"] for category in categories}
    result = {}
    for bucket in BUCKETS:
        population = by_bucket[bucket]
        if len(population) < SAMPLE_SIZE:
            raise ValueError(f"{bucket}: only {len(population)} eligible images")
        # A separate seeded stream makes each sample invariant to other bucket sizes.
        rng = random.Random(f"{seed}:{bucket}")
        sampled = sorted(rng.sample(population, SAMPLE_SIZE), key=lambda row: row["image_id"])
        dest = output / bucket
        dest.mkdir(parents=True, exist_ok=True)
        assessment_path = dest / "assessments.csv"
        if assessment_path.exists():
            with assessment_path.open(newline="", encoding="utf-8") as stream:
                old_ids = {r["image_id"] for r in csv.DictReader(stream)}
            if old_ids != {r["image_id"] for r in sampled}:
                raise ValueError(f"Existing assessments refer to a different sample: {assessment_path}")
        coco_images, coco_annotations, manifest, assessments = [], [], [], []
        for index, row in enumerate(sampled, 1):
            image_path = ROOT / row["image_path"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if row.get("error") or row.get("image_error"):
                raise ValueError(f"Cannot sample errored image: {row['image_id']}")
            label = row["species_label"]
            if label not in category_id:
                raise ValueError(f"Unknown species label: {label}")
            folder = os.path.relpath(image_path.parent, dest)
            if (dest / folder / image_path.name).resolve() != image_path.resolve():
                raise ValueError(f"COCO path resolution failed: {image_path}")
            coco_images.append({"id": index, "file_name": image_path.name,
                                "folder": folder, "width": row["width"], "height": row["height"]})
            selected_detections = [d for d in row["detections"] if d["score"] >= MIN_SCORE]
            for detection in selected_detections:
                x1, y1, x2, y2 = detection["xyxy"]
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"Invalid detection: {row['image_id']}")
                coco_annotations.append({"id": len(coco_annotations) + 1,
                                         "image_id": index, "category_id": category_id[label],
                                         "bbox": [x1, y1, x2 - x1, y2 - y1],
                                         "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            manifest.append({"coco_image_id": index, "image_id": row["image_id"],
                             "image_path": row["image_path"], "source": row["source"],
                             "species_label": label, "bucket": bucket,
                             "top_score": row["top_score"],
                             "draft_box_count": len(selected_detections),
                             "model_sha256": row["model_sha256"]})
            assessments.append({"coco_image_id": index, "image_id": row["image_id"],
                                "bucket": bucket, "tail_coverage": "",
                                "box_usability": "", "review_decision": "", "notes": ""})
        coco = {"info": {"description": f"Squirrel v2 {bucket} v1 detector quality sample",
                         "sample_seed": seed, "population_size": len(population),
                         "sample_size": SAMPLE_SIZE, "min_detection_score": MIN_SCORE,
                         "source_inference": str(inference.relative_to(ROOT)),
                         "status": "review_draft_not_training_data"},
                "licenses": [], "images": coco_images, "annotations": coco_annotations,
                "categories": categories}
        (dest / "draft.coco.json").write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")
        write_csv(dest / "sample_manifest.csv", manifest,
                  ["coco_image_id", "image_id", "image_path", "source", "species_label",
                   "bucket", "top_score", "draft_box_count", "model_sha256"])
        # Preserve a human-edited assessment file on reruns; sample changes require explicit handling.
        if not assessment_path.exists():
            write_csv(assessment_path, assessments,
                      ["coco_image_id", "image_id", "bucket", "tail_coverage",
                       "box_usability", "review_decision", "notes"])
        assert len(coco_images) == SAMPLE_SIZE == len({r["image_id"] for r in manifest})
        assert all(r["bucket"] == bucket for r in manifest)
        assert all((dest / i["folder"] / i["file_name"]).is_file() for i in coco_images)
        result[bucket] = {"population": len(population), "sampled": SAMPLE_SIZE,
                          "boxes": len(coco_annotations),
                          "sources": dict(Counter(r["source"] for r in sampled))}
    (output / "sampling_summary.json").write_text(json.dumps({"seed": seed, "buckets": result}, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(build(seed=args.seed), indent=2))
