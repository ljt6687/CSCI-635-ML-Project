"""Lossless PyLabel CSV/DataFrame to COCO export for squirrel review drafts.

PyLabel's ExportToCoco raises UnboundLocalError on images with a blank
category row. This exporter keeps such images and deliberately omits a box.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _value(row: dict, key: str) -> str:
    value = row.get(key, "")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _number(row: dict, key: str) -> float:
    value = _value(row, key)
    if not value:
        raise ValueError(f"missing {key} in image {_value(row, 'img_id')}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {key} in image {_value(row, 'img_id')}")
    return result


def read_pylabel_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def export_rows(rows: list[dict], template: dict) -> tuple[dict, dict]:
    """Return COCO and audit, preserving PyLabel image deletions and empty rows.

    Existing annotation IDs survive exact box matches, so later Gemini
    verdicts can be joined without relying on row order.
    """
    image_by_id = {int(item["id"]): item for item in template["images"]}
    category_by_id = {int(item["id"]): item["name"] for item in template["categories"]}
    category_by_name = {name: cid for cid, name in category_by_id.items()}
    old_by_image = defaultdict(list)
    for item in template["annotations"]:
        old_by_image[int(item["image_id"])].append(item)
    seen_image_ids: set[int] = set()
    new_by_image = defaultdict(list)
    new_id = max((int(item["id"]) for item in template["annotations"]), default=0) + 1
    reused_ids: set[int] = set()
    normalized_reversed_boxes: list[dict] = []
    for row in rows:
        image_id = int(_number(row, "img_id"))
        if image_id not in image_by_id:
            raise ValueError(f"PyLabel image {image_id} absent from template")
        image = image_by_id[image_id]
        if _value(row, "img_filename") != image["file_name"]:
            raise ValueError(f"image {image_id} filename mismatch")
        if _value(row, "img_folder") != image["folder"]:
            raise ValueError(f"image {image_id} folder mismatch")
        if int(_number(row, "img_width")) != int(image["width"]) or int(_number(row, "img_height")) != int(image["height"]):
            raise ValueError(f"image {image_id} dimensions mismatch")
        seen_image_ids.add(image_id)
        cid_text = _value(row, "cat_id")
        cname = _value(row, "cat_name")
        bbox_present = any(_value(row, key) for key in ("ann_bbox_xmin", "ann_bbox_ymin", "ann_bbox_xmax", "ann_bbox_ymax"))
        if not cid_text and not cname:
            if bbox_present:
                raise ValueError(f"image {image_id} has a box without category")
            continue
        if cname and cname not in category_by_name:
            raise ValueError(f"unknown category {cname!r} on image {image_id}")
        cid = category_by_name[cname] if cname else int(float(cid_text))
        if cid not in category_by_id:
            raise ValueError(f"unknown category ID {cid} on image {image_id}")
        if cid_text and int(float(cid_text)) != cid:
            raise ValueError(f"category ID/name disagree on image {image_id}")
        x1, y1, x2, y2 = (_number(row, key) for key in ("ann_bbox_xmin", "ann_bbox_ymin", "ann_bbox_xmax", "ann_bbox_ymax"))
        original_endpoints = [x1, y1, x2, y2]
        if x1 > x2 or y1 > y2:
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            normalized_reversed_boxes.append({"image_id": image_id, "original_xyxy": original_endpoints,
                                              "normalized_xyxy": [x1, y1, x2, y2]})
        if not (0 <= x1 < x2 <= float(image["width"]) and 0 <= y1 < y2 <= float(image["height"])):
            raise ValueError(f"invalid or out-of-bounds box on image {image_id}: {(x1,y1,x2,y2)}")
        box = [x1, y1, x2 - x1, y2 - y1]
        match = next((item for item in old_by_image[image_id]
                      if int(item["id"]) not in reused_ids and int(item["category_id"]) == cid
                      and all(abs(float(a)-float(b)) < 1e-4 for a,b in zip(item["bbox"], box))), None)
        if match is not None:
            ann = dict(match)
            reused_ids.add(int(ann["id"]))
        else:
            ann = {"id": new_id, "image_id": image_id, "category_id": cid,
                   "bbox": box, "area": box[2]*box[3], "iscrowd": 0,
                   "annotation_basis": "human_pylabel_csv"}
            new_id += 1
        new_by_image[image_id].append(ann)
    if not rows:
        raise ValueError("PyLabel export has no rows")
    images = [dict(item) for item in template["images"] if int(item["id"]) in seen_image_ids]
    annotations = [item for image in images for item in new_by_image[int(image["id"])]]
    removed = [int(item["id"]) for item in template["images"] if int(item["id"]) not in seen_image_ids]
    changed = []
    for image in images:
        iid = int(image["id"])
        old_ids = {int(item["id"]) for item in old_by_image[iid]}
        new_ids = {int(item["id"]) for item in new_by_image[iid]}
        if old_ids != new_ids:
            changed.append(iid)
    coco = {key: copy.deepcopy(value) for key, value in template.items() if key not in ("images", "annotations")}
    coco["images"] = images
    coco["annotations"] = annotations
    coco.setdefault("info", {})["description"] = "REVIEW DRAFT ONLY: saved PyLabel edits; not approved for training"
    audit = {"input_rows": len(rows), "images": len(images), "boxes": len(annotations),
             "manually_removed_image_ids": removed, "changed_image_ids": changed,
             "reused_original_box_ids": len(reused_ids),
             "normalized_reversed_boxes": normalized_reversed_boxes}
    return coco, audit


def export_dataframe(df, template_path: Path, output_path: Path) -> dict:
    """Export a live PyLabel DataFrame, including blank-category image rows."""
    if output_path.exists():
        raise FileExistsError(f"preserving existing review draft: {output_path}")
    template = json.loads(template_path.read_text(encoding="utf-8"))
    coco, audit = export_rows(df.to_dict("records"), template)
    output_path.write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")
    return audit
