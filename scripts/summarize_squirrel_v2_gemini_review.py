#!/usr/bin/env python3
"""Validate Gemini's 63-image visual audit and attach provisional triage to review."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/v2_annotation/gemini_many_boxes"
REVIEW_QUEUE = ROOT / "output/v2_annotation/review/review_queue.jsonl"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def iou(a: list[float], b: list[float]) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return intersection / (area(a) + area(b) - intersection)


def meyer_reference_boxes(row: dict) -> list[list[float]]:
    if row["source"] != "meyer_trailcam":
        return []
    source = (ROOT / row["image_path"]).resolve()
    xml = ET.parse(source.with_suffix(".xml")).getroot()
    return [[float(obj.findtext(f"bndbox/{name}")) for name in ("xmin", "ymin", "xmax", "ymax")]
            for obj in xml.findall("object")]


def priority_for(row: dict, original: dict, second: str, meyer_conflict: bool) -> tuple[int, str, str]:
    if meyer_conflict:
        return 1, "reference_conflict", "Compare model proposals with Meyer XML and the original frame; retain XML as the starting draft."
    if original["uncertain"] or row["squirrel_count"] != original["squirrel_count"]:
        return 1, "overlay_original_disagreement", "Inspect original at full resolution; Gemini's overlay and unmarked-image reviews disagree or are uncertain."
    if row["squirrel_count"] == 0 and second == "YES":
        return 1, "high_model_disagreement", "Inspect original at full resolution; Gemini High reports a squirrel despite two empty first-pass readings."
    if row["squirrel_count"] == 0 and second == "NO":
        if row["bucket"] == "passed":
            return 1, "passed_likely_empty_three_pass", "Confirm this high-scoring negative on the original, then mark empty in the human decision sheet."
        return 2, "likely_empty_three_pass", "Confirm no squirrel in the original, then mark empty in the human decision sheet."
    if row["missed_squirrel"]:
        return 1, "missed_squirrel", "Find the squirrel in the original and draw or replace a box in PyLabel."
    if row["bucket"] == "passed" and next(box["verdict"] for box in row["box_reviews"] if box["box_id"] == "B1") == "remove":
        return 1, "passed_top_box_disputed", "Inspect the high-scoring top box and correct the proposed instance boxes."
    if row["squirrel_count"] == 0 and second == "NO_RESULT":
        return 2, "empty_two_low_high_unresolved", "Two low-effort readings suggest empty, but the high-effort pass timed out; inspect original at full resolution."
    if any(box["verdict"] == "adjust" for box in row["box_reviews"]):
        return 2, "box_adjustment", "Edit squirrel boxes and remove duplicate or background proposals in PyLabel."
    if any(box["verdict"] == "remove" for box in row["box_reviews"]):
        return 3, "remove_proposals", "Verify and delete duplicate or background boxes in PyLabel."
    return 3, "verify_boxes", "Verify all boxes against distinct visible squirrels."



def main() -> None:
    cohort = jsonl(BASE / "cohort.jsonl")
    if len(cohort) != 63 or len({row["image_id"] for row in cohort}) != 63:
        raise ValueError("cohort is not the frozen 63-image set")
    predictions = {row["image_id"]: row for row in jsonl(ROOT / "output/v2_annotation/squirrel_v2_inference.jsonl")}
    queue = jsonl(REVIEW_QUEUE)
    queue_ids = {row["image_id"] for row in queue}
    if len(queue_ids) != len(queue):
        raise ValueError("duplicate image IDs in review queue")
    if any(row["image_id"] not in queue_ids for row in cohort):
        raise ValueError("a Gemini image is missing from the human review queue")
    second_dir = BASE / "empty_second_look"
    images, boxes = [], []
    low_reviews = {}
    for row in cohort:
        key = row["review_key"]
        low_path = BASE / "reviews" / f"{key}.json"
        if not low_path.exists():
            raise ValueError(f"missing visual review: {key}")
        low = json.loads(low_path.read_text(encoding="utf-8"))
        low_reviews[row["image_id"]] = low
        if low["image_id"] != row["image_id"] or low["model_sha256"] != row["model_sha256"]:
            raise ValueError(f"review provenance mismatch: {key}")
        actual_ids = [item["box_id"] for item in low["box_reviews"]]
        expected_ids = [item["box_id"] for item in row["proposals"]]
        if sorted(actual_ids) != sorted(expected_ids) or len(actual_ids) != len(expected_ids):
            raise ValueError(f"box IDs mismatch: {key}")
        original_path = BASE / "original_reviews" / f"{key}.json"
        if not original_path.exists():
            raise ValueError(f"missing blind original-image review: {key}")
        original = json.loads(original_path.read_text(encoding="utf-8"))
        if original["image_id"] != row["image_id"]:
            raise ValueError(f"blind original image ID mismatch: {key}")
        opinion_path = second_dir / f"{key}.json"
        second = (json.loads(opinion_path.read_text(encoding="utf-8"))["verdict"]
                  if opinion_path.exists() else ("NO_RESULT" if low["squirrel_count"] == 0 else "NOT_APPLICABLE"))
        if low["squirrel_count"] and second != "NOT_APPLICABLE":
            raise ValueError(f"second empty pass on nonempty review: {key}")
        references = meyer_reference_boxes(row)
        by_id = {item["box_id"]: item for item in low["box_reviews"]}
        model = predictions[row["image_id"]]
        top_verdict = by_id["B1"]["verdict"]
        max_top_iou = max((iou(row["proposals"][0]["xyxy"], reference) for reference in references), default=None)
        meyer_conflict = bool(references and top_verdict == "keep" and max_top_iou < 0.5)
        rank, category, action = priority_for({**low, "bucket": row["bucket"]}, original, second, meyer_conflict)
        verdicts = Counter(item["verdict"] for item in low["box_reviews"])
        images.append({
            "priority": rank, "triage_category": category, "image_id": row["image_id"],
            "bucket": row["bucket"], "source": row["source"], "image_path": row["image_path"],
            "species_label": model["species_label"], "top_score": model["top_score"],
            "material_box_count": len(row["proposals"]),
            "gemini_visible_squirrel_count": low["squirrel_count"],
            "blind_original_squirrel_count": original["squirrel_count"],
            "blind_original_uncertain": original["uncertain"],
            "blind_original_evidence": original["evidence"],
            "original_second_opinion": second, "top_box_verdict": top_verdict,
            "keep_boxes": verdicts["keep"], "adjust_boxes": verdicts["adjust"],
            "remove_boxes": verdicts["remove"], "uncertain_boxes": verdicts["uncertain"],
            "possible_missed_squirrel": low["missed_squirrel"],
            "meyer_top_iou": max_top_iou, "human_action": action,
            "gemini_notes": low["overall_notes"],
        })
        for det in row["proposals"]:
            judgment = by_id[det["box_id"]]
            boxes.append({"image_id": row["image_id"], "bucket": row["bucket"],
                          "box_id": det["box_id"], "score": det["score"],
                          "xyxy": json.dumps(det["xyxy"]), "gemini_verdict": judgment["verdict"],
                          "gemini_reason": judgment["reason"],
                          "meyer_reference_max_iou": max((iou(det["xyxy"], ref) for ref in references), default=None)})
    images.sort(key=lambda row: (row["priority"], row["triage_category"], row["bucket"], row["image_id"]))
    boxes.sort(key=lambda row: (row["image_id"], int(row["box_id"][1:])))
    with (BASE / "triage_images.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(images[0]))
        writer.writeheader(); writer.writerows(images)
    with (BASE / "triage_boxes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(boxes[0]))
        writer.writeheader(); writer.writerows(boxes)
    summary = {
        "cohort_images": len(images), "material_model_boxes": len(boxes),
        "by_bucket": dict(Counter(row["bucket"] for row in images)),
        "gemini_visible_squirrel_count": dict(Counter(row["gemini_visible_squirrel_count"] for row in images)),
        "box_verdicts": dict(Counter(row["gemini_verdict"] for row in boxes)),
        "top_box_verdicts": dict(Counter(row["top_box_verdict"] for row in images)),
        "first_pass_empty_images": sum(row["gemini_visible_squirrel_count"] == 0 for row in images),
        "blind_original_counts": dict(Counter(row["blind_original_squirrel_count"] for row in images)),
        "overlay_original_count_agree": sum(row["gemini_visible_squirrel_count"] == row["blind_original_squirrel_count"]
                                            and not row["blind_original_uncertain"] for row in images),
        "blind_original_uncertain_images": sum(row["blind_original_uncertain"] for row in images),
        "empty_consensus_three_pass": sum(row["gemini_visible_squirrel_count"] == 0
                                          and row["blind_original_squirrel_count"] == 0
                                          and row["original_second_opinion"] == "NO" for row in images),
        "empty_second_opinions": dict(Counter(row["original_second_opinion"] for row in images
                                             if row["gemini_visible_squirrel_count"] == 0)),
        "possible_missed_squirrel_images": sum(row["possible_missed_squirrel"] for row in images),
        "triage_categories": dict(Counter(row["triage_category"] for row in images)),
        "meyer_reference_conflicts": sum(row["triage_category"] == "reference_conflict" for row in images),
        "status": "provisional_gemini_triage_pending_human_review",
    }
    (BASE / "triage_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    backup = REVIEW_QUEUE.with_name("review_queue_pre_gemini.jsonl")
    if not backup.exists():
        shutil.copy2(REVIEW_QUEUE, backup)
    image_lookup = {row["image_id"]: row for row in images}
    for item in queue:
        triage = image_lookup.get(item["image_id"])
        if triage is None:
            continue
        low = low_reviews[item["image_id"]]
        item["gemini_many_boxes"] = {
            "provisional": True, "model": low["gemini_model"],
            "visible_squirrel_count": triage["gemini_visible_squirrel_count"],
            "blind_original_squirrel_count": triage["blind_original_squirrel_count"],
            "blind_original_uncertain": triage["blind_original_uncertain"],
            "blind_original_evidence": triage["blind_original_evidence"],
            "empty_original_second_opinion": triage["original_second_opinion"],
            "possible_missed_squirrel": triage["possible_missed_squirrel"],
            "triage_category": triage["triage_category"], "human_action": triage["human_action"],
            "box_reviews": low["box_reviews"], "overall_notes": low["overall_notes"],
        }
        item["selection_reasons"] = sorted(set(item["selection_reasons"] + ["gemini_many_boxes_audit"]))
    temp = REVIEW_QUEUE.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for item in queue:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    temp.replace(REVIEW_QUEUE)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
