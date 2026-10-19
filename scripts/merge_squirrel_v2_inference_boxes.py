#!/usr/bin/env python3
"""Write a separate, union-merged copy of all frozen squirrel v1 detections.

This preserves the original inference JSONL and every image/bucket assignment.
Human and Gemini review corrections take precedence for selected images later.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/v2_annotation/squirrel_v2_inference.jsonl"
TARGET = ROOT / "output/v2_annotation/squirrel_v2_inference_iou_union.jsonl"
SUMMARY = ROOT / "output/v2_annotation/inference_iou_union_summary.json"
THRESHOLD = 0.80


def iou_xyxy(a: list[float], b: list[float]) -> float:
    left = max(a[0], b[0]); top = max(a[1], b[1])
    right = min(a[2], b[2]); bottom = min(a[3], b[3])
    intersection = max(0.0, right-left) * max(0.0, bottom-top)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection/union if union else 0.0


def union_detections(detections: list[dict]) -> tuple[list[dict], list[list[int]]]:
    parent = list(range(len(detections)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for left in range(len(detections)):
        a = detections[left]
        for right in range(left+1, len(detections)):
            b = detections[right]
            if a["class_id"] != b["class_id"] or a.get("species_label") != b.get("species_label"):
                continue
            if iou_xyxy(a["xyxy"], b["xyxy"]) > THRESHOLD:
                parent[root(right)] = root(left)
    components: dict[int,list[int]] = {}
    for index in range(len(detections)):
        components.setdefault(root(index), []).append(index)
    merged, groups = [], []
    for indices in sorted(components.values(), key=lambda part: min(part)):
        if len(indices) == 1:
            merged.append(detections[indices[0]])
            continue
        members = [detections[i] for i in indices]
        best = max(members, key=lambda item: item["score"])
        item = dict(best)
        item["xyxy"] = [min(d["xyxy"][0] for d in members), min(d["xyxy"][1] for d in members),
                        max(d["xyxy"][2] for d in members), max(d["xyxy"][3] for d in members)]
        item["localization_flags"] = sorted({flag for d in members for flag in d.get("localization_flags", [])})
        item["merged_from_detection_indices"] = indices
        item["box_merge_method"] = "same_class_iou_strictly_above_0.80_union"
        merged.append(item)
        groups.append(indices)
    merged.sort(key=lambda item: float(item["score"]), reverse=True)
    return merged, groups


def main() -> None:
    stats = Counter()
    with SOURCE.open(encoding="utf-8") as input_stream, TARGET.open("w", encoding="utf-8") as output_stream:
        for line in input_stream:
            if not line.strip():
                continue
            row = json.loads(line)
            original = row["detections"]
            merged, groups = union_detections(original)
            row["detections"] = merged
            row["box_postprocess"] = {"method": "same_class_iou_union", "iou_threshold_strictly_greater_than": THRESHOLD,
                                       "original_detection_count": len(original), "merged_group_count": len(groups)}
            if merged:
                assert abs(row["top_score"] - max(item["score"] for item in merged)) < 1e-9
            output_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["images"] += 1
            stats["original_detections"] += len(original)
            stats["merged_detections"] += len(merged)
            stats["merged_groups"] += len(groups)
            stats["images_with_union"] += bool(groups)
            stats[f"bucket_{row['bucket']}"] += 1
    summary = dict(stats)
    summary.update({"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    "output_sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest(),
                    "overlap_metric": "intersection_over_union", "threshold_strictly_greater_than": THRESHOLD,
                    "source_preserved": True, "status": "postprocessed_inference_not_training_export"})
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
