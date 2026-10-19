#!/usr/bin/env python3
"""Build a fresh, disjoint 100-image passed-bucket detector quality sample."""
from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

from build_squirrel_v2_bucket_samples import ROOT, INFERENCE, CATEGORIES, read_jsonl

PRIOR = ROOT / "output/v2_annotation/bucket_samples"
REVIEW = ROOT / "output/v2_annotation/review"
OUT = PRIOR / "passed_round2"
SEED = 20260926
SIZE = 100
FLOOR = 0.20


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_generated(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to replace an existing generated artifact: {path}")
        return
    path.write_text(content, encoding="utf-8")


def csv_text(rows: list[dict], fields: list[str]) -> str:
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> dict:
    previous_ids: set[str] = set()
    for bucket in ("passed", "probably_passed"):
        path = PRIOR / bucket / "sample_manifest.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Prior sample manifest required: {path}")
        previous_ids.update(row["image_id"] for row in csv_rows(path))
    saved_csv = REVIEW / "reviewed.coco.csv"
    removed_ids: set[str] = set()
    if saved_csv.is_file():
        decisions_path = REVIEW / "review_decisions.csv"
        if not decisions_path.is_file():
            raise FileNotFoundError(decisions_path)
        surviving_coco_ids = {int(row["img_id"]) for row in csv_rows(saved_csv)}
        removed_ids = {row["image_id"] for row in csv_rows(decisions_path)
                       if int(row["coco_image_id"]) not in surviving_coco_ids}
    source = list(read_jsonl(INFERENCE))
    passed = sorted((row for row in source if row.get("bucket") == "passed"),
                    key=lambda row: row["image_id"])
    eligible = [row for row in passed
                if row["image_id"] not in previous_ids and row["image_id"] not in removed_ids]
    if len(eligible) < SIZE:
        raise ValueError(f"Only {len(eligible)} passed images remain after exclusions")
    rng = random.Random(f"{SEED}:passed_round2")
    selected = sorted(rng.sample(eligible, SIZE), key=lambda row: row["image_id"])
    categories = json.loads(CATEGORIES.read_text(encoding="utf-8"))["categories"]
    category_id = {row["name"]: row["id"] for row in categories}
    OUT.mkdir(parents=True, exist_ok=True)
    images, annotations, manifest, assessments = [], [], [], []
    for iid, row in enumerate(selected, 1):
        path = ROOT / row["image_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if row.get("error") or row.get("image_error"):
            raise ValueError(f"Selected erroneous inference row: {row['image_id']}")
        label = row["species_label"]
        if label not in category_id:
            raise ValueError(f"Unknown species: {label}")
        folder = os.path.relpath(path.parent, OUT)
        if (OUT / folder / path.name).resolve() != path.resolve():
            raise ValueError(f"Path does not resolve: {path}")
        images.append({"id": iid, "file_name": path.name, "folder": folder,
                       "width": row["width"], "height": row["height"]})
        detections = [d for d in row["detections"] if d["score"] >= FLOOR]
        for detection in detections:
            x1, y1, x2, y2 = detection["xyxy"]
            if not (0 <= x1 < x2 <= row["width"] and 0 <= y1 < y2 <= row["height"]):
                raise ValueError(f"Invalid detector box: {row['image_id']}")
            annotations.append({"id": len(annotations) + 1, "image_id": iid,
                                "category_id": category_id[label], "bbox": [x1, y1, x2-x1, y2-y1],
                                "area": (x2-x1)*(y2-y1), "iscrowd": 0})
        manifest.append({"coco_image_id": iid, "image_id": row["image_id"],
                         "image_path": row["image_path"], "source": row["source"],
                         "species_label": label, "bucket": "passed",
                         "top_score": row["top_score"], "draft_box_count": len(detections),
                         "model_sha256": row["model_sha256"]})
        assessments.append({"coco_image_id": iid, "image_id": row["image_id"],
                            "bucket": "passed", "tail_coverage": "", "box_usability": "",
                            "review_decision": "", "notes": ""})
    assert len(images) == len(manifest) == len(assessments) == SIZE
    assert len({row["image_id"] for row in manifest}) == SIZE
    assert not {row["image_id"] for row in manifest} & (previous_ids | removed_ids)
    coco = {"info": {"description": "Fresh passed-bucket raw v1 detector review draft",
                     "status": "quality_review_only_not_training_data", "seed": SEED,
                     "population_size": len(passed), "eligible_after_exclusions": len(eligible),
                     "sample_size": SIZE, "min_detection_score": FLOOR,
                     "source_inference": str(INFERENCE.relative_to(ROOT)),
                     "excluded_prior_sample_count": len(previous_ids),
                     "excluded_saved_review_removed_count": len(removed_ids)},
            "licenses": [], "images": images, "annotations": annotations,
            "categories": categories}
    write_generated(OUT / "draft.coco.json", json.dumps(coco, indent=2) + "\n")
    write_generated(OUT / "sample_manifest.csv", csv_text(manifest,
                    ["coco_image_id", "image_id", "image_path", "source", "species_label",
                     "bucket", "top_score", "draft_box_count", "model_sha256"]))
    assessments_path = OUT / "assessments.csv"
    if not assessments_path.exists():
        assessments_path.write_text(csv_text(assessments,
                              ["coco_image_id", "image_id", "bucket", "tail_coverage",
                               "box_usability", "review_decision", "notes"]), encoding="utf-8")
    elif {row["image_id"] for row in csv_rows(assessments_path)} != {row["image_id"] for row in assessments}:
        raise ValueError(f"Existing assessments are from another sample: {assessments_path}")
    summary = {"seed": SEED, "original_passed_population": len(passed),
               "prior_sample_ids_excluded": len(previous_ids),
               "removed_main_review_ids_excluded": len(removed_ids),
               "eligible_passed": len(eligible), "sampled": SIZE, "raw_detector_boxes": len(annotations),
               "sources": dict(Counter(row["source"] for row in selected)),
               "review_artifact_found": saved_csv.is_file()}
    write_generated(OUT / "sampling_summary.json", json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
