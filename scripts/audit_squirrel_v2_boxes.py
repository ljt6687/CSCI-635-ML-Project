#!/usr/bin/env python3
"""Audit provisional v1 squirrel boxes without treating them as ground truth.

Area is measured against the original image, so mixed image resolutions are
comparable. The 0.20 floor matches the editable review draft; 0.05 is the
archived proposal floor and 0.50 shows stronger proposals.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


BUCKETS = ("passed", "probably_passed", "recheck_needed")
FLOORS = (0.05, 0.20, 0.50)
AREA_EDGES = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25)
AREA_LABELS = ("<0.1%", "0.1-0.5%", "0.5-1%", "1-2%", "2-5%", "5-10%", "10-25%", ">=25%")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def area_fraction(det: dict, width: int, height: int) -> float:
    x1, y1, x2, y2 = det["xyxy"]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"invalid box {det['xyxy']} for image {width}x{height}")
    return (x2 - x1) * (y2 - y1) / (width * height)


def percentile(sorted_values: list[float], percent: float) -> float | None:
    if not sorted_values:
        return None
    index = (len(sorted_values) - 1) * percent / 100
    lower = int(index)
    fraction = index - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[min(lower + 1, len(sorted_values) - 1)] * fraction


def count_bin(count: int) -> str:
    return "4+" if count >= 4 else str(count)


def area_bin(fraction: float) -> str:
    for edge, label in zip(AREA_EDGES, AREA_LABELS):
        if fraction < edge:
            return label
    return AREA_LABELS[-1]


def audit(predictions: list[dict], review_ids: set[str]) -> tuple[dict, list[dict]]:
    if len({row["image_id"] for row in predictions}) != len(predictions):
        raise ValueError("duplicate image_id in predictions")
    if review_ids - {row["image_id"] for row in predictions}:
        raise ValueError("review queue contains images absent from predictions")
    buckets: dict[str, dict] = {
        bucket: {
            "images": 0,
            "in_review": 0,
            "source_images": Counter(),
            "counts_by_floor": {str(floor): Counter() for floor in FLOORS},
            "areas_at_0.20": [],
            "top_areas": [],
            "flags": Counter(),
            "flags_in_review": Counter(),
            "sources": defaultdict(Counter),
        }
        for bucket in BUCKETS
    }
    per_image = []
    for row in predictions:
        bucket = row["bucket"]
        if bucket not in buckets:
            raise ValueError(f"unrecognized bucket: {bucket}")
        width, height = row["width"], row["height"]
        detections = row["detections"]
        areas = [area_fraction(det, width, height) for det in detections]
        if any(det["score"] < 0.05 for det in detections):
            raise ValueError(f"below archived proposal floor: {row['image_id']}")
        material = [area for area, det in zip(areas, detections) if det["score"] >= 0.20]
        top_area = areas[0] if areas else None
        counts = {str(floor): sum(det["score"] >= floor for det in detections) for floor in FLOORS}
        flags = {
            "more_than_three_0.20": counts["0.2"] > 3,
            "more_than_three_0.05": counts["0.05"] > 3,
            "top_below_0.1pct": top_area is not None and top_area < 0.001,
            "top_below_0.5pct": top_area is not None and top_area < 0.005,
            "top_below_1pct": top_area is not None and top_area < 0.01,
            "any_0.20_below_0.1pct": any(area < 0.001 for area in material),
            "any_0.20_below_0.5pct": any(area < 0.005 for area in material),
            "any_0.20_below_1pct": any(area < 0.01 for area in material),
            "zero_0.20_boxes": counts["0.2"] == 0,
            "invalid_detection": row.get("bucket_reason") == "invalid_model_detection",
        }
        flags["small_or_many_priority"] = flags["any_0.20_below_0.5pct"] or flags["more_than_three_0.20"]
        flags["small_or_many_watchlist"] = flags["any_0.20_below_1pct"] or flags["more_than_three_0.20"]
        result = buckets[bucket]
        result["images"] += 1
        result["source_images"][row["source"]] += 1
        in_review = row["image_id"] in review_ids
        result["in_review"] += int(in_review)
        for floor, count in counts.items():
            result["counts_by_floor"][floor][count_bin(count)] += 1
        result["areas_at_0.20"].extend(material)
        if top_area is not None:
            result["top_areas"].append(top_area)
        for name, value in flags.items():
            if value:
                result["flags"][name] += 1
                result["sources"][row["source"]][name] += 1
                if in_review:
                    result["flags_in_review"][name] += 1
        per_image.append({
            "image_id": row["image_id"], "image_path": row["image_path"],
            "source": row["source"], "species_label": row["species_label"],
            "bucket": bucket, "bucket_reason": row["bucket_reason"],
            "top_score": row["top_score"], "top_area_fraction": top_area,
            "min_material_area_fraction": min(material) if material else None,
            "boxes_0.05": counts["0.05"], "boxes_0.20": counts["0.2"],
            "boxes_0.50": counts["0.5"], "in_review": in_review,
            **flags,
        })

    model_hashes = {row.get("model_sha256") for row in predictions}
    if len(model_hashes) != 1 or None in model_hashes:
        raise ValueError("predictions have missing or mixed model hashes")
    summary = {"image_total": len(predictions), "review_total": len(review_ids),
               "model_sha256": next(iter(model_hashes)),
               "area_unit": "fraction_of_original_image", "counts_meaning": "model_proposals_not_verified_instances",
               "buckets": {}}
    for bucket, result in buckets.items():
        areas = sorted(result.pop("areas_at_0.20"))
        top_areas = sorted(result.pop("top_areas"))
        result["source_images"] = dict(sorted(result["source_images"].items()))
        result["counts_by_floor"] = {
            floor: {name: counter.get(name, 0) for name in ("0", "1", "2", "3", "4+")}
            for floor, counter in result["counts_by_floor"].items()
        }
        result["material_box_area"] = {
            "box_count": len(areas),
            "percentiles_fraction": {str(p): percentile(areas, p) for p in (10, 25, 50, 75, 90)},
            "histogram": {name: sum(area_bin(area) == name for area in areas) for name in AREA_LABELS},
        }
        result["top_box_area"] = {
            "box_count": len(top_areas),
            "percentiles_fraction": {str(p): percentile(top_areas, p) for p in (10, 25, 50, 75, 90)},
            "histogram": {name: sum(area_bin(area) == name for area in top_areas) for name in AREA_LABELS},
        }
        result["flags"] = dict(result["flags"])
        result["flags_in_review"] = dict(result["flags_in_review"])
        result["sources"] = {name: dict(counts) for name, counts in sorted(result["sources"].items())}
        summary["buckets"][bucket] = result
    return summary, per_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("output/v2_annotation/squirrel_v2_inference.jsonl"))
    parser.add_argument("--review-queue", type=Path, default=Path("output/v2_annotation/review/review_queue.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/v2_annotation/box_audit"))
    args = parser.parse_args()
    review_ids = {row["image_id"] for row in read_jsonl(args.review_queue)}
    summary, per_image = audit(read_jsonl(args.predictions), review_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    priority = [row for row in per_image if row["small_or_many_priority"] or row["invalid_detection"]]
    priority.sort(key=lambda row: (
        {"recheck_needed": 0, "probably_passed": 1, "passed": 2}[row["bucket"]],
        not row["invalid_detection"], not row["more_than_three_0.20"],
        not row["top_below_0.1pct"], not row["top_below_0.5pct"], row["image_id"],
    ))
    with (args.output_dir / "geometry_priority.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(priority)
    print(json.dumps({"images": summary["image_total"], "priority_images": len(priority),
                      "by_bucket": {bucket: data["images"] for bucket, data in summary["buckets"].items()},
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
