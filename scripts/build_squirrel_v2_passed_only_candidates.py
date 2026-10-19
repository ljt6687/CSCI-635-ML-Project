#!/usr/bin/env python3
"""Build a passed-only *review candidate* with human sample promotions.

No train/validation split, resizing, copying, or final training export occurs.
Original bucket and source provenance are preserved on every image.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scripts.squirrel_v2_review_export import export_rows, read_pylabel_csv
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from squirrel_v2_review_export import export_rows, read_pylabel_csv


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/v2_annotation"
SAMPLES = BASE / "bucket_samples"
REVIEW = BASE / "review"
OUT = BASE / "passed_only"
MIN_SCORE = 0.20


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_manual_sample(sample_name: str, category_names: dict[int, str]) -> tuple[dict, dict, list[dict]]:
    directory = SAMPLES / sample_name
    template = json.loads((directory / "draft.coco.json").read_text(encoding="utf-8"))
    reviewed, audit = export_rows(read_pylabel_csv(directory / "reviewed.pylabel.csv"), template)
    manifest = read_csv(directory / "sample_manifest.csv")
    sample_size = len(manifest)
    if len({row["image_id"] for row in manifest}) != sample_size:
        raise ValueError(f"{sample_name} sample manifest has duplicate images")
    if len(reviewed["images"]) != sample_size or audit["manually_removed_image_ids"]:
        raise ValueError(f"{sample_name} sample review does not retain all {sample_size} images")
    if {int(cat["id"]): cat["name"] for cat in reviewed["categories"]} != category_names:
        raise ValueError(f"{sample_name} category mapping differs from canonical review draft")
    by_local_id = {int(row["coco_image_id"]): row for row in manifest}
    anns = defaultdict(list)
    for ann in reviewed["annotations"]:
        anns[int(ann["image_id"])].append(ann)
    by_source_id = {}
    for image in reviewed["images"]:
        local_id = int(image["id"])
        row = by_local_id[local_id]
        by_source_id[row["image_id"]] = {"manifest": row, "image": image, "annotations": anns[local_id],
                                         "sample_name": sample_name, "sample_bucket": row["bucket"]}
    if len(by_source_id) != sample_size:
        raise ValueError("sample image IDs did not map one to one")
    reviewed["info"] = {"description": f"Recovered human PyLabel {sample_name} sample; review artifact only"}
    return by_source_id, {"audit": audit, "csv_sha256": hashlib.sha256((directory / "reviewed.pylabel.csv").read_bytes()).hexdigest()}, reviewed


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    original_review = json.loads((REVIEW / "review_candidates.coco.json").read_text(encoding="utf-8"))
    category_names = {int(category["id"]): category["name"] for category in original_review["categories"]}
    category_ids = {name: cid for cid, name in category_names.items()}
    manual = {}
    audits = {}
    sample_names = ["passed", "probably_passed_round1" if (SAMPLES / "probably_passed_round1").is_dir() else "probably_passed", "passed_round2"]
    if (SAMPLES / "probably_passed_round1").is_dir() and (SAMPLES / "probably_passed" / "reviewed.pylabel.csv").is_file():
        sample_names.append("probably_passed")
    for name in sample_names:
        mapping, audit, reviewed = load_manual_sample(name, category_names)
        if set(mapping) & set(manual):
            raise ValueError(f"duplicate image across manual samples: {name}")
        manual.update(mapping)
        audits[name] = audit
        (OUT / f"recovered_{name}_sample.coco.json").write_text(json.dumps(reviewed, indent=2) + "\n", encoding="utf-8")
    if len(manual) != 500:
        raise ValueError(f"expected exactly 500 manually reviewed sample images, got {len(manual)}")

    main_summary = json.loads((REVIEW / "reviewed_combined_summary.json").read_text(encoding="utf-8"))
    if main_summary["saved_csv_sha256"] != hashlib.sha256((REVIEW / "reviewed.coco.csv").read_bytes()).hexdigest():
        raise ValueError("main reviewed COCO does not match its current saved CSV")
    reviewed_main = json.loads((REVIEW / "reviewed_combined.coco.json").read_text(encoding="utf-8"))
    reviewed_main_ids = {int(image["id"]) for image in reviewed_main["images"]}
    review_lookup = {row["image_id"]: int(row["coco_image_id"]) for row in read_csv(REVIEW / "review_decisions.csv")}
    reviewed_annotations = defaultdict(list)
    for ann in reviewed_main["annotations"]:
        reviewed_annotations[int(ann["image_id"])].append(ann)

    inference_rows = [json.loads(line) for line in (BASE / "squirrel_v2_inference_iou_union.jsonl").open(encoding="utf-8") if line.strip()]
    inference_by_id = {row["image_id"]: row for row in inference_rows}
    if len(inference_by_id) != len(inference_rows):
        raise ValueError("duplicate full-inference image ID")
    if set(manual) - set(inference_by_id):
        raise ValueError("manually reviewed image absent from frozen inference")
    for image_id, value in manual.items():
        if inference_by_id[image_id]["bucket"] != value["sample_bucket"]:
            raise ValueError(f"sample bucket mismatch for {image_id}")

    selected = {}
    exclusions = []
    for row in inference_rows:
        image_id = row["image_id"]
        original_bucket = row["bucket"]
        if original_bucket != "passed" and image_id not in manual:
            continue
        if image_id in manual:
            source = "human_pylabel_sample"
            candidate_anns = manual[image_id]["annotations"]
        else:
            review_id = review_lookup.get(image_id)
            if review_id is not None and review_id not in reviewed_main_ids:
                exclusions.append({"image_id": image_id, "original_bucket": original_bucket,
                                   "reason": "removed_in_main_pylabel_csv"})
                continue
            if review_id is not None:
                candidate_anns = reviewed_annotations[review_id]
                source = "main_review_combined"
            else:
                candidate_anns = [
                    {"category_id": category_ids[row["species_label"]],
                     "bbox": [d["xyxy"][0], d["xyxy"][1], d["xyxy"][2]-d["xyxy"][0], d["xyxy"][3]-d["xyxy"][1]],
                     "area": (d["xyxy"][2]-d["xyxy"][0])*(d["xyxy"][3]-d["xyxy"][1]),
                     "iscrowd": 0, "detector_score": d["score"],
                     "annotation_basis": "v1_model_iou_union" if d.get("merged_from_detection_indices") else "v1_model_provisional",
                     **({"merged_from_detection_indices": d["merged_from_detection_indices"]}
                        if d.get("merged_from_detection_indices") else {})}
                    for d in row["detections"] if float(d["score"]) >= MIN_SCORE
                ]
                source = "union_merged_detector"
        if not candidate_anns:
            exclusions.append({"image_id": image_id, "original_bucket": original_bucket,
                               "reason": "no_boxes_after_review"})
            continue
        if image_id in selected:
            raise ValueError(f"duplicate selected image {image_id}")
        selected[image_id] = {"row": row, "annotations": candidate_anns, "annotation_source": source}

    if any(x["row"]["bucket"] != "passed" and image_id not in manual for image_id,x in selected.items()):
        raise ValueError("unreviewed lower-confidence bucket leaked into passed selection")
    unpromoted = (set(manual) - set(selected)) - {x["image_id"] for x in exclusions}
    if unpromoted:
        raise ValueError(f"some manually reviewed images were not promoted: {unpromoted}")

    images, annotations, manifest, promotions = [], [], [], []
    image_species = Counter()
    box_species = Counter()
    annotation_id = 1
    for index, image_id in enumerate(sorted(selected), 1):
        chosen = selected[image_id]
        row = chosen["row"]
        source_path = ROOT / row["image_path"]
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        category_set = sorted({category_names[int(ann["category_id"])] for ann in chosen["annotations"]})
        for species in category_set:
            image_species[species] += 1
        image = {"id": index, "file_name": source_path.name,
                 "folder": os.path.relpath(source_path.parent, OUT),
                 "width": int(row["image_size"]["width"]), "height": int(row["image_size"]["height"]),
                 "source_image_id": image_id, "source_path": row["image_path"], "source": row["source"],
                 "source_species_label": row["species_label"],
                 "annotation_species_labels": category_set,
                 "original_bucket": row["bucket"], "effective_bucket": "passed",
                 "annotation_source": chosen["annotation_source"],
                 "model_sha256": row["model_sha256"], "top_score": row["top_score"]}
        images.append(image)
        for original_ann in chosen["annotations"]:
            box = [float(v) for v in original_ann["bbox"]]
            x,y,w,h = box
            if not (0 <= x < x+w <= image["width"] and 0 <= y < y+h <= image["height"]):
                raise ValueError(f"invalid candidate box on {image_id}: {box}")
            category_id = int(original_ann["category_id"])
            ann = {key: value for key,value in original_ann.items()
                   if key not in ("id", "image_id", "category_id", "bbox", "area", "iscrowd")}
            ann.update({"id": annotation_id, "image_id": index, "category_id": category_id,
                        "bbox": box, "area": w*h, "iscrowd": int(original_ann.get("iscrowd",0))})
            annotations.append(ann)
            box_species[category_names[category_id]] += 1
            annotation_id += 1
        entry = {"source_image_id": image_id, "coco_image_id": index,
                 "original_bucket": row["bucket"], "effective_bucket": "passed",
                 "annotation_source": chosen["annotation_source"],
                 "source": row["source"], "source_species_label": row["species_label"],
                 "species_labels": "|".join(category_set),
                 "box_count": len(chosen["annotations"]), "image_path": row["image_path"]}
        manifest.append(entry)
        if image_id in manual:
            promotions.append(dict(entry, manual_sample_bucket=manual[image_id]["sample_name"],
                                   sample_csv_sha256=audits[manual[image_id]["sample_name"]]["csv_sha256"]))

    coco = {"info": {"description": "PASSED-ONLY REVIEW CANDIDATE; human sample annotations override model proposals; not a final training export"},
            "images": images, "annotations": annotations, "categories": original_review["categories"]}
    (OUT / "passed_only_review_candidates.coco.json").write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")
    write_csv(OUT / "passed_only_manifest.csv", manifest,
              ["source_image_id", "coco_image_id", "original_bucket", "effective_bucket", "annotation_source", "source", "source_species_label", "species_labels", "box_count", "image_path"])
    write_csv(OUT / "promoted_human_reviews.csv", promotions,
              ["source_image_id", "coco_image_id", "original_bucket", "effective_bucket", "annotation_source", "source", "source_species_label", "species_labels", "box_count", "image_path", "manual_sample_bucket", "sample_csv_sha256"])
    label_corrections = [{"source_image_id": row["source_image_id"],
                          "original_bucket": row["original_bucket"],
                          "source_species_label": row["source_species_label"],
                          "reviewed_species_labels": row["species_labels"]}
                         for row in promotions if row["source_species_label"] not in row["species_labels"].split("|")]
    write_csv(OUT / "manual_species_label_corrections.csv", label_corrections,
              ["source_image_id", "original_bucket", "source_species_label", "reviewed_species_labels"])
    write_csv(OUT / "excluded_review_images.csv", exclusions, ["image_id", "original_bucket", "reason"])
    normalized = []
    for sample_name, audit in audits.items():
        reverse = {int(row["coco_image_id"]): row["image_id"] for row in read_csv(SAMPLES / sample_name / "sample_manifest.csv")}
        for item in audit["audit"]["normalized_reversed_boxes"]:
            normalized.append({"sample_bucket": sample_name, "source_image_id": reverse[item["image_id"]],
                               "original_xyxy": json.dumps(item["original_xyxy"]),
                               "normalized_xyxy": json.dumps(item["normalized_xyxy"])})
    write_csv(OUT / "normalized_review_boxes.csv", normalized,
              ["sample_bucket", "source_image_id", "original_xyxy", "normalized_xyxy"])
    summary = {"status":"passed_only_review_candidate_pending_sample_validation_and_green_light",
               "original_inference_bucket_counts":dict(Counter(r["bucket"] for r in inference_rows)),
               "selected_images":len(images), "selected_boxes":len(annotations),
               "selected_original_buckets":dict(Counter(x["original_bucket"] for x in manifest)),
               "selected_annotation_sources":dict(Counter(x["annotation_source"] for x in manifest)),
               "promoted_human_review_images":len(promotions),
               "manual_species_label_corrections":len(label_corrections),
               "promoted_original_buckets":dict(Counter(x["original_bucket"] for x in promotions)),
               "sample_review_audit":{bucket:{"csv_sha256":audit["csv_sha256"],
                    "changed_images":len(audit["audit"]["changed_image_ids"]),
                    "normalized_reversed_boxes":len(audit["audit"]["normalized_reversed_boxes"]),
                    "reviewed_boxes":audit["audit"]["boxes"]} for bucket,audit in audits.items()},
               "excluded_review_images":dict(Counter(x["reason"] for x in exclusions)),
               "species_image_counts":dict(sorted(image_species.items())),
               "species_box_counts":dict(sorted(box_species.items())),
               "minimum_images_per_species_for_training":600,
               "species_below_600_images":[name for name,n in sorted(image_species.items()) if name != "generic_squirrel" and n<600],
               "lower_confidence_unreviewed_images_included":0,
               "train_validation_split_created":False}
    (OUT / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
