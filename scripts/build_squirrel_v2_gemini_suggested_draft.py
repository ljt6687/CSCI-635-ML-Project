#!/usr/bin/env python3
"""Build a reversible Gemini-filtered COCO review draft, never a training export."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/v2_annotation/gemini_many_boxes"
REVIEW = ROOT / "output/v2_annotation/review"
OUTPUT = BASE / "review_candidates.gemini_suggested.coco.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def eligible(row: dict) -> tuple[bool, str]:
    if row["source"] == "meyer_trailcam":
        return False, "meyer_xml_reference"
    if row["blind_original_uncertain"] == "True":
        return False, "blind_original_uncertain"
    if row["gemini_visible_squirrel_count"] != row["blind_original_squirrel_count"]:
        return False, "scene_count_disagreement"
    if row["possible_missed_squirrel"] == "True":
        return False, "possible_missed_squirrel"
    if row["gemini_visible_squirrel_count"] == "0" and row["original_second_opinion"] != "NO":
        return False, "empty_not_confirmed_by_high_pass"
    return True, "overlay_and_original_agree"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"preserving existing optional Gemini draft: {OUTPUT}")
    decisions = list(csv.DictReader((REVIEW / "review_decisions.csv").open(newline="", encoding="utf-8")))
    if any((row["decision"] or row["final_species_label"] or row["notes"]).strip() for row in decisions):
        raise ValueError("human decisions have begun; do not rebuild an optional draft over their work")
    coco_id = {row["image_id"]: int(row["coco_image_id"]) for row in decisions if row["coco_image_id"]}
    triage = list(csv.DictReader((BASE / "triage_images.csv").open(newline="", encoding="utf-8")))
    if len(triage) != 63:
        raise ValueError("triage must contain all 63 crowded images")
    cohort = {row["image_id"]: row for row in read_jsonl(BASE / "cohort.jsonl")}
    reviews = {row["image_id"]: row for row in read_jsonl(REVIEW / "review_queue.jsonl")}
    original = json.loads((REVIEW / "review_candidates.coco.json").read_text(encoding="utf-8"))
    coco = json.loads(json.dumps(original))
    by_coco_image = defaultdict(list)
    image_by_coco_id = {image["id"]: image for image in coco["images"]}
    for annotation in coco["annotations"]:
        by_coco_image[annotation["image_id"]].append(annotation)
    keep_ids, removed = set(), []
    applied_images = 0
    skipped = Counter()
    for row in triage:
        use, reason = eligible(row)
        if not use:
            skipped[reason] += 1
            continue
        image_id = row["image_id"]
        source_row = cohort[image_id]
        queue_row = reviews[image_id]
        c_id = coco_id[image_id]
        annotations = sorted(by_coco_image[c_id], key=lambda item: item["id"])
        proposals = source_row["proposals"]
        if len(annotations) != len(proposals):
            raise ValueError(f"COCO proposal count mismatch: {image_id}")
        verdicts = {item["box_id"]: item for item in queue_row["gemini_many_boxes"]["box_reviews"]}
        if set(verdicts) != {item["box_id"] for item in proposals}:
            raise ValueError(f"Gemini box IDs mismatch: {image_id}")
        for annotation, proposal in zip(annotations, proposals):
            x1, y1, x2, y2 = proposal["xyxy"]
            expected = [x1, y1, x2 - x1, y2 - y1]
            if any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(annotation["bbox"], expected)):
                raise ValueError(f"COCO box order/coordinate mismatch: {image_id} {proposal['box_id']}")
            judgment = verdicts[proposal["box_id"]]
            if judgment["verdict"] == "remove":
                removed.append({"image_id": image_id, "coco_image_id": c_id,
                                "annotation_id": annotation["id"], "box_id": proposal["box_id"],
                                "score": proposal["score"], "reason": judgment["reason"]})
            else:
                keep_ids.add(annotation["id"])
                annotation["gemini_provisional_verdict"] = judgment["verdict"]
                annotation["gemini_provisional_reason"] = judgment["reason"]
        applied_images += 1
        image_by_coco_id[c_id]["gemini_suggested_review_draft"] = True
    removed_ids = {row["annotation_id"] for row in removed}
    if len(removed_ids) != len(removed) or removed_ids & keep_ids:
        raise ValueError("duplicate or contradictory annotation actions")
    coco["annotations"] = [item for item in coco["annotations"] if item["id"] not in removed_ids]
    coco.setdefault("info", {})["description"] = (
        "OPTIONAL GEMINI-SUGGESTED REVIEW DRAFT; provisional removals only where overlay and original scene passes agree. "
        "Human correction required; never use as training COCO."
    )
    OUTPUT.write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")
    with (BASE / "suggested_removed_boxes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(removed[0]) if removed else
                                ["image_id", "coco_image_id", "annotation_id", "box_id", "score", "reason"])
        writer.writeheader()
        writer.writerows(removed)
    summary = {
        "status": "optional_provisional_review_draft_pending_human_validation",
        "cohort_images": len(triage), "eligible_images": applied_images,
        "skipped_images": dict(skipped), "original_coco_images": len(original["images"]),
        "original_coco_boxes": len(original["annotations"]),
        "suggested_coco_images": len(coco["images"]),
        "suggested_coco_boxes": len(coco["annotations"]),
        "suggested_removed_boxes": len(removed),
        "canonical_review_coco_unchanged": True,
    }
    (BASE / "suggested_draft_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
