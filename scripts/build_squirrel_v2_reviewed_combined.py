#!/usr/bin/env python3
"""Reconcile saved PyLabel edits, Gemini verdicts, and heavy duplicate boxes.

Outputs are review drafts. No training export or approval decision is made.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scripts.squirrel_v2_review_export import export_rows, read_pylabel_csv
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from squirrel_v2_review_export import export_rows, read_pylabel_csv


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "output/v2_annotation/review"
GEMINI = ROOT / "output/v2_annotation/gemini_many_boxes"
IOU_THRESHOLD = 0.80


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def box_iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def merge_heavy_overlaps(coco: dict, protected_images: set[int]) -> tuple[dict, list[dict]]:
    """Merge model boxes of the same class when IoU is strictly above 0.80."""
    images = {int(image["id"]): image for image in coco["images"]}
    groups = defaultdict(list)
    for annotation in coco["annotations"]:
        groups[int(annotation["image_id"])].append(annotation)
    output, audit = [], []
    for iid, annotations in groups.items():
        image = images[iid]
        if iid in protected_images or image.get("annotation_basis") != "v1_model_provisional":
            output.extend(annotations)
            continue
        parent = list(range(len(annotations)))

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        qualifying = []
        for left in range(len(annotations)):
            for right in range(left + 1, len(annotations)):
                a, b = annotations[left], annotations[right]
                if int(a["category_id"]) != int(b["category_id"]):
                    continue
                if a.get("gemini_provisional_verdict") == "adjust" or b.get("gemini_provisional_verdict") == "adjust":
                    continue
                overlap = box_iou(a["bbox"], b["bbox"])
                if overlap > IOU_THRESHOLD:
                    parent[root(right)] = root(left)
                    qualifying.append((int(a["id"]), int(b["id"]), overlap))
        components = defaultdict(list)
        for index, item in enumerate(annotations):
            components[root(index)].append(item)
        for members in components.values():
            if len(members) == 1:
                output.append(members[0])
                continue
            first = min(members, key=lambda item: int(item["id"]))
            x1 = min(float(item["bbox"][0]) for item in members)
            y1 = min(float(item["bbox"][1]) for item in members)
            x2 = max(float(item["bbox"][0]) + float(item["bbox"][2]) for item in members)
            y2 = max(float(item["bbox"][1]) + float(item["bbox"][3]) for item in members)
            merged = dict(first)
            merged["bbox"] = [x1, y1, x2 - x1, y2 - y1]
            merged["area"] = (x2 - x1) * (y2 - y1)
            member_ids = sorted(int(item["id"]) for item in members)
            merged["merged_from_annotation_ids"] = member_ids
            merged["annotation_basis"] = "v1_model_iou_union"
            output.append(merged)
            audit.append({"coco_image_id": iid, "file_name": image["file_name"],
                          "category_id": first["category_id"], "retained_annotation_id": first["id"],
                          "merged_annotation_ids": json.dumps(member_ids),
                          "original_boxes": json.dumps([item["bbox"] for item in members]),
                          "union_bbox": json.dumps(merged["bbox"]),
                          "max_trigger_iou": max(overlap for a,b,overlap in qualifying if a in member_ids and b in member_ids)})
    result = dict(coco)
    result["annotations"] = sorted(output, key=lambda item: int(item["id"]))
    return result, audit


def apply_gemini(coco: dict, original: dict) -> tuple[dict, list[dict]]:
    decisions = list(csv.DictReader((REVIEW / "review_decisions.csv").open(newline="", encoding="utf-8")))
    image_ids = {row["image_id"]: int(row["coco_image_id"]) for row in decisions}
    source_rows = {row["image_id"]: row for row in read_jsonl(GEMINI / "cohort.jsonl")}
    triage_rows = list(csv.DictReader((GEMINI / "triage_images.csv").open(newline="", encoding="utf-8")))
    verdict_rows = list(csv.DictReader((GEMINI / "triage_boxes.csv").open(newline="", encoding="utf-8")))
    verdicts = {(row["image_id"], row["box_id"]): row for row in verdict_rows}
    original_by_image = defaultdict(list)
    for item in original["annotations"]:
        original_by_image[int(item["image_id"])].append(item)
    present_images = {int(item["id"]) for item in coco["images"]}
    current_by_id = {int(item["id"]): item for item in coco["annotations"]}
    removed_ids = set()
    audit = []
    for row in triage_rows:
        string_id = row["image_id"]
        iid = image_ids[string_id]
        proposals = source_rows[string_id]["proposals"]
        annotations = sorted(original_by_image[iid], key=lambda item: int(item["id"]))
        if row["source"] == "meyer_trailcam":
            for proposal in proposals:
                judgment = verdicts[(string_id, proposal["box_id"])]
                audit.append({"image_id": string_id, "coco_image_id": iid, "box_id": proposal["box_id"],
                              "original_annotation_id": "", "verdict": judgment["gemini_verdict"],
                              "action": "meyer_xml_reference_preserved", "reason": judgment["gemini_reason"]})
            continue
        if len(annotations) != len(proposals):
            raise ValueError(f"Gemini proposal count mismatch: {string_id}")
        for annotation, proposal in zip(annotations, proposals):
            x1,y1,x2,y2 = proposal["xyxy"]
            expected = [x1,y1,x2-x1,y2-y1]
            if any(abs(float(a)-float(b)) > 1e-4 for a,b in zip(annotation["bbox"], expected)):
                raise ValueError(f"Gemini proposal box mismatch: {string_id} {proposal['box_id']}")
            judgment = verdicts[(string_id, proposal["box_id"])]
            verdict = judgment["gemini_verdict"]
            aid = int(annotation["id"])
            if iid not in present_images:
                action = "manual_image_removal_has_precedence"
            elif aid not in current_by_id:
                action = "manual_box_edit_has_precedence"
            elif verdict == "remove":
                removed_ids.add(aid)
                action = "gemini_box_removed"
            else:
                current_by_id[aid]["gemini_provisional_verdict"] = verdict
                current_by_id[aid]["gemini_provisional_reason"] = judgment["gemini_reason"]
                action = "retained_pending_adjustment" if verdict == "adjust" else "gemini_box_kept"
            audit.append({"image_id": string_id, "coco_image_id": iid, "box_id": proposal["box_id"],
                          "original_annotation_id": aid, "verdict": verdict, "action": action,
                          "reason": judgment["gemini_reason"]})
    if len(audit) != len(verdict_rows):
        raise ValueError("not all Gemini verdicts were reconciled")
    result = dict(coco)
    result["annotations"] = [item for item in coco["annotations"] if int(item["id"]) not in removed_ids]
    return result, audit


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source_csv = REVIEW / "reviewed.coco.csv"
    original = json.loads((REVIEW / "review_candidates.coco.json").read_text(encoding="utf-8"))
    saved, manual_audit = export_rows(read_pylabel_csv(source_csv), original)
    gemini, gemini_audit = apply_gemini(saved, original)
    combined, union_audit = merge_heavy_overlaps(gemini, set(manual_audit["changed_image_ids"]))
    combined["info"] = dict(combined.get("info", {}))
    combined["info"]["description"] = (
        "REVIEW DRAFT ONLY: saved PyLabel image/box edits, Gemini provisional box verdicts, "
        "and same-class model box unions for IoU > 0.80. Not approved for training."
    )
    for path, data in ((REVIEW / "reviewed.coco.json", saved),
                       (REVIEW / "reviewed_combined.coco.json", combined)):
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    image_by_id = {int(item["id"]): item for item in original["images"]}
    removed_rows = [{"coco_image_id": iid, "file_name": image_by_id[iid]["file_name"],
                     "folder": image_by_id[iid]["folder"], "action": "removed_in_saved_pylabel_csv"}
                    for iid in manual_audit["manually_removed_image_ids"]]
    write_csv(REVIEW / "manual_removed_images.csv", removed_rows, list(removed_rows[0]))
    write_csv(REVIEW / "gemini_reconciliation.csv", gemini_audit,
              ["image_id", "coco_image_id", "box_id", "original_annotation_id", "verdict", "action", "reason"])
    write_csv(REVIEW / "box_unions.csv", union_audit,
              ["coco_image_id", "file_name", "category_id", "retained_annotation_id",
               "merged_annotation_ids", "original_boxes", "union_bbox", "max_trigger_iou"])
    summary = {
        "status": "review_draft_pending_human_validation",
        "saved_csv_sha256": hashlib.sha256(source_csv.read_bytes()).hexdigest(),
        "original_images": len(original["images"]), "original_boxes": len(original["annotations"]),
        "pylabel_export_images": len(saved["images"]), "pylabel_export_boxes": len(saved["annotations"]),
        "manual_removed_images": len(removed_rows), "manual_changed_surviving_images": len(manual_audit["changed_image_ids"]),
        "gemini_actions": dict(Counter(row["action"] for row in gemini_audit)),
        "gemini_filtered_boxes": len(saved["annotations"]) - len(gemini["annotations"]),
        "box_union_groups": len(union_audit),
        "box_union_reduction": len(gemini["annotations"]) - len(combined["annotations"]),
        "combined_images": len(combined["images"]), "combined_boxes": len(combined["annotations"]),
        "overlap_metric": "intersection_over_union", "overlap_threshold_strictly_greater_than": IOU_THRESHOLD,
        "pending_box_adjustments": sum(row["action"] == "retained_pending_adjustment" for row in gemini_audit),
        "manual_decisions_not_populated": True,
    }
    (REVIEW / "reviewed_combined_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
