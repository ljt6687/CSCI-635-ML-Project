#!/usr/bin/env python3
"""Build an immutable, audited 11-class COCO export from squirrel-v2-clean.

The input split is provenance, not a trusted training split. Exact duplicate
records are removed; observation groups and tight perceptual risk components
are assigned together before the new train/valid/test split is written.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any

import imagehash
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/squirrel-v2-clean"
DECISIONS = ROOT / "output/v2_annotation/audit_v2/decisions.json"
DESTINATION = ROOT / "data/squirrel-v2-audited-coco"
SPLITS = ("train", "valid", "test")
LONG_SIDE = 1152
SEED = 42
PHASH_GROUP_DISTANCE = 4


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_fingerprint(source: Path) -> str:
    digest = hashlib.sha256()
    for split in SPLITS:
        digest.update((source / split / "_annotations.coco.json").read_bytes())
    return digest.hexdigest()


def image_key(split: str, file_name: str) -> str:
    return f"{split}/{file_name}"


def image_fingerprint(entries: list[Entry]) -> str:
    lines = sorted(f"{entry.key}|{entry.source_pixel_hash}\n" for entry in entries)
    return sha256_bytes("".join(lines).encode())


def turn_point(x: float, y: float, width: int, height: int, orientation: int) -> tuple[float, float]:
    """Map an original-image boundary point through an EXIF orientation."""
    if orientation == 2:
        return width - x, y
    if orientation == 3:
        return width - x, height - y
    if orientation == 4:
        return x, height - y
    if orientation == 5:
        return y, x
    if orientation == 6:
        return height - y, x
    if orientation == 7:
        return height - y, width - x
    if orientation == 8:
        return y, width - x
    return x, y


def transformed_box(
    bbox: list[float], width: int, height: int, orientation: int, out_width: int, out_height: int
) -> list[float] | None:
    if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox):
        return None
    x, y, w, h = map(float, bbox)
    if x < -0.05 or y < -0.05 or w <= 0 or h <= 0 or x + w > width + 0.05 or y + h > height + 0.05:
        return None
    x, y = max(0.0, x), max(0.0, y)
    w, h = min(w, width - x), min(h, height - y)
    corners = [turn_point(a, b, width, height, orientation) for a, b in
               ((x, y), (x + w, y), (x, y + h), (x + w, y + h))]
    oriented_width, oriented_height = (height, width) if orientation in (5, 6, 7, 8) else (width, height)
    xs = [p[0] * out_width / oriented_width for p in corners]
    ys = [p[1] * out_height / oriented_height for p in corners]
    left, right = max(0.0, min(xs)), min(float(out_width), max(xs))
    top, bottom = max(0.0, min(ys)), min(float(out_height), max(ys))
    if right - left <= 0.5 or bottom - top <= 0.5:
        return None
    return [round(left, 4), round(top, 4), round(right - left, 4), round(bottom - top, 4)]


class UnionFind:
    def __init__(self, count: int):
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


@dataclass
class Entry:
    split: str
    file_name: str
    original_image: dict[str, Any]
    category_id: int
    annotations: list[dict[str, Any]]
    group_id: str
    stage_path: Path
    width: int
    height: int
    source_pixel_hash: str
    output_pixel_hash: str
    phash: int
    output_boxes: list[tuple[int, list[float]]]
    exif_orientation: int
    assigned_split: str = ""
    risk_group_id: str = ""

    @property
    def key(self) -> str:
        return image_key(self.split, self.file_name)


def load_source(source: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    documents = {split: json.loads((source / split / "_annotations.coco.json").read_text()) for split in SPLITS}
    categories = sorted(documents["train"]["categories"], key=lambda c: int(c["id"]))
    expected_ids = list(range(1, 12))
    if [int(c["id"]) for c in categories] != expected_ids or len({c["name"] for c in categories}) != 11:
        raise ValueError("Expected 11 distinct COCO categories with IDs 1..11")
    for split, doc in documents.items():
        if sorted(doc["categories"], key=lambda c: int(c["id"])) != categories:
            raise ValueError(f"{split}: category schema differs from train")
    return documents, categories


def prepare_entries(
    source: Path, documents: dict[str, dict[str, Any]], decisions: dict[str, Any], stage: Path
) -> tuple[list[Entry], list[dict[str, Any]]]:
    exclusions = {image_key(d["split"], d["file_name"]): d["reason"] for d in decisions.get("exclude", [])}
    relabels = {image_key(d["split"], d["file_name"]): d for d in decisions.get("relabel", [])}
    annotation_exclusions = {}
    for decision in decisions.get("exclude_annotations", []):
        key = (image_key(decision["split"], decision["file_name"]), int(decision["annotation_id"]))
        if key in annotation_exclusions or not decision.get("reason") or not decision.get("evidence"):
            raise ValueError(f"Invalid annotation exclusion decision: {key}")
        annotation_exclusions[key] = decision
    seen_annotation_exclusions: set[tuple[str, int]] = set()
    category_by_name = {c["name"]: int(c["id"]) for c in documents["train"]["categories"]}
    entries: list[Entry] = []
    removed: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    stage.mkdir(parents=True)

    for split, doc in documents.items():
        images = doc["images"]
        image_ids = [i["id"] for i in images]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(f"{split}: duplicate image IDs")
        anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in doc["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)
        for orphan in set(anns_by_image) - set(image_ids):
            removed.append({"key": f"{split}/annotation:{orphan}", "reason": "orphan annotation"})

        for item in images:
            name = item["file_name"]
            key = image_key(split, name)
            if key in seen_keys:
                raise ValueError(f"Duplicate image key: {key}")
            seen_keys.add(key)
            if key in exclusions:
                removed.append({"key": key, "reason": exclusions[key]})
                continue
            split_dir = (source / split).resolve()
            path = (split_dir / name).resolve()
            if not path.is_relative_to(split_dir) or not path.is_file():
                removed.append({"key": key, "reason": "missing or unsafe image path"})
                continue
            raw_anns = anns_by_image.get(item["id"], [])
            kept_anns = []
            for ann in raw_anns:
                ann_key = (key, int(ann["id"]))
                if ann_key in annotation_exclusions:
                    seen_annotation_exclusions.add(ann_key)
                    removed.append({"key": f"{key}#annotation:{ann['id']}",
                                    "reason": annotation_exclusions[ann_key]["reason"]})
                else:
                    kept_anns.append(ann)
            raw_anns = kept_anns
            if not raw_anns:
                removed.append({"key": key, "reason": "missing annotation"})
                continue
            cids = {int(a.get("category_id", -1)) for a in raw_anns}
            if len(cids) != 1 or not cids.issubset(set(category_by_name.values())):
                removed.append({"key": key, "reason": "mixed or unknown class annotations"})
                continue
            old_id = next(iter(cids))
            new_id = old_id
            if key in relabels:
                change = relabels[key]
                if (change.get("from") != next(c["name"] for c in doc["categories"] if c["id"] == old_id)
                        or change.get("to") not in category_by_name or not change.get("evidence")):
                    raise ValueError(f"Unsupported relabel decision: {key}")
                new_id = category_by_name[change["to"]]

            try:
                with Image.open(path) as original:
                    original.load()
                    width, height = original.size
                    if (width, height) != (item["width"], item["height"]):
                        removed.append({"key": key, "reason": "source dimension mismatch"})
                        continue
                    orientation = int(original.getexif().get(274, 1) or 1)
                    if orientation not in range(1, 9):
                        orientation = 1
                    source_rgb = original.convert("RGB")
                    source_pixel_hash = sha256_bytes(
                        f"{width}x{height}".encode() + source_rgb.tobytes()
                    )
                    rgb = ImageOps.exif_transpose(original).convert("RGB")
                    ow, oh = rgb.size
                    scale = min(1.0, LONG_SIDE / max(ow, oh))
                    new_width, new_height = max(1, round(ow * scale)), max(1, round(oh * scale))
                    if (new_width, new_height) != (ow, oh):
                        rgb = rgb.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    output_boxes = []
                    for ann in raw_anns:
                        box = transformed_box(ann.get("bbox", []), width, height, orientation, new_width, new_height)
                        if box is not None:
                            output_boxes.append((new_id, box))
                        else:
                            removed.append({"key": f"{key}#annotation:{ann.get('id')}",
                                            "reason": "invalid bbox"})
                    if not output_boxes:
                        removed.append({"key": key, "reason": "no valid boxes after transform"})
                        continue
                    unique = sha256_bytes(key.encode())[:20] + ".jpg"
                    stage_path = stage / unique
                    rgb.save(stage_path, format="JPEG", quality=92, subsampling=0)
                with Image.open(stage_path) as encoded:
                    encoded.load()
                    encoded_rgb = encoded.convert("RGB")
                    output_pixel_hash = sha256_bytes(
                        f"{new_width}x{new_height}:".encode() + encoded_rgb.tobytes()
                    )
                    phash = int(str(imagehash.phash(encoded_rgb)), 16)
            except Exception as exc:
                removed.append({"key": key, "reason": f"unreadable image: {type(exc).__name__}"})
                continue
            entries.append(Entry(split, name, item, new_id, raw_anns,
                                 str(item.get("group_id") or item.get("source_image_id") or key),
                                 stage_path, new_width, new_height, source_pixel_hash,
                                 output_pixel_hash, phash, output_boxes, orientation))
    unknown_decisions = (set(exclusions) | set(relabels)) - seen_keys
    if unknown_decisions:
        raise ValueError(f"Decisions reference absent input images: {sorted(unknown_decisions)[:5]}")
    unknown_annotations = set(annotation_exclusions) - seen_annotation_exclusions
    if unknown_annotations:
        raise ValueError(f"Decisions reference absent or excluded annotations: {sorted(unknown_annotations)[:5]}")
    return entries, removed


def dedupe_exact(entries: list[Entry], removed: list[dict[str, Any]]) -> list[Entry]:
    by_hash: dict[str, list[Entry]] = defaultdict(list)
    for entry in entries:
        by_hash[entry.source_pixel_hash].append(entry)
    kept = []
    for group in by_hash.values():
        if len({e.category_id for e in group}) > 1:
            raise ValueError(f"Exact duplicate has conflicting classes: {[e.key for e in group]}")
        group.sort(key=lambda e: (-len(e.output_boxes), SPLITS.index(e.split), e.key))
        kept.append(group[0])
        for duplicate in group[1:]:
            removed.append({"key": duplicate.key, "reason": f"exact pixel duplicate of {group[0].key}"})
            duplicate.stage_path.unlink(missing_ok=True)
    return sorted(kept, key=lambda e: e.key)


def risk_components(entries: list[Entry], decisions: dict[str, Any]) -> tuple[list[list[int]], int]:
    uf = UnionFind(len(entries))
    index = {e.key: i for i, e in enumerate(entries)}
    for component in (decisions.get("confirmed_duplicate_components", [])
                      + decisions.get("visually_related_components", [])):
        members = [index[image_key(d["split"], d["file_name"])] for d in component
                   if image_key(d["split"], d["file_name"]) in index]
        for other in members[1:]:
            uf.union(members[0], other)
    by_group: dict[str, int] = {}
    by_output_hash: dict[str, int] = {}
    for i, entry in enumerate(entries):
        if entry.group_id in by_group:
            uf.union(i, by_group[entry.group_id])
        by_group[entry.group_id] = i
        if entry.output_pixel_hash in by_output_hash:
            other = entries[by_output_hash[entry.output_pixel_hash]]
            if entry.category_id != other.category_id:
                raise ValueError(f"Processed duplicate has conflicting classes: {other.key}, {entry.key}")
            uf.union(i, by_output_hash[entry.output_pixel_hash])
        by_output_hash[entry.output_pixel_hash] = i

    # With five chunks, Hamming distance <= 4 leaves at least one unchanged.
    # This conservatively isolates all tight perceptual risks in one split.
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    tight_edges = 0
    for i, entry in enumerate(entries):
        candidates: set[int] = set()
        for part in range(5):
            chunk = (entry.phash >> (13 * part)) & (0x1FFF if part < 4 else 0xFFF)
            candidates.update(buckets[(part, chunk)])
            buckets[(part, chunk)].append(i)
        for j in candidates:
            if (entry.phash ^ entries[j].phash).bit_count() <= PHASH_GROUP_DISTANCE:
                tight_edges += 1
                uf.union(i, j)
    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(entries)):
        components[uf.find(i)].append(i)
    for component in components.values():
        group_id = sha256_bytes("|".join(sorted(entries[i].key for i in component)).encode())[:16]
        for i in component:
            entries[i].risk_group_id = group_id
    return list(components.values()), tight_edges


def assign_splits(entries: list[Entry], components: list[list[int]]) -> list[dict[str, Any]]:
    moves = []
    groups = []
    for component in components:
        originals = {entries[i].split for i in component}
        assigned = next(iter(originals)) if len(originals) == 1 and len(component) <= 4 else "train"
        for i in component:
            entries[i].assigned_split = assigned
            if entries[i].split != assigned:
                moves.append({"key": entries[i].key, "from": entries[i].split, "to": assigned,
                              "reason": "group or perceptual-risk split isolation"})
        groups.append((component, len(originals) == 1 and len(component) <= 4))

    totals = Counter(e.category_id for e in entries)
    counts = {split: Counter(e.category_id for e in entries if e.assigned_split == split) for split in SPLITS}
    rng = random.Random(SEED)
    eligible = [c for c, movable in groups if movable and entries[c[0]].assigned_split == "train"
                and len({entries[i].category_id for i in c}) == 1]
    rng.shuffle(eligible)
    eligible.sort(key=len, reverse=True)
    for split in ("valid", "test"):
        for cid in sorted(totals, key=lambda c: totals[c]):
            target = round(0.15 * totals[cid])
            while counts[split][cid] < target:
                deficit = target - counts[split][cid]
                choices = [c for c in eligible if entries[c[0]].category_id == cid and len(c) <= deficit + 1]
                if not choices:
                    break
                component = min(choices, key=lambda c: (abs(deficit - len(c)), len(c)))
                eligible.remove(component)
                for i in component:
                    entries[i].assigned_split = split
                    counts["train"][cid] -= 1
                    counts[split][cid] += 1
                    moves.append({"key": entries[i].key, "from": "train", "to": split,
                                  "reason": "class-stratified rebalance"})
    return moves


def write_export(
    entries: list[Entry], categories: list[dict[str, Any]], destination: Path,
    fingerprint: str, decisions: dict[str, Any], decisions_sha: str,
    removed: list[dict[str, Any]], moves: list[dict[str, Any]], tight_edges: int,
    input_image_count: int,
) -> dict[str, Any]:
    category_names = {int(c["id"]): c["name"] for c in categories}
    annotation_hashes = {}
    split_counts = {}
    annotation_counts = {}
    per_class = {category_names[c]: {s: 0 for s in SPLITS} for c in category_names}
    output_hash_splits: dict[str, str] = {}
    for split in SPLITS:
        folder = destination / split
        folder.mkdir(parents=True)
        coco_images, coco_annotations = [], []
        subset = sorted((e for e in entries if e.assigned_split == split), key=lambda e: e.key)
        for iid, entry in enumerate(subset, 1):
            output_name = entry.stage_path.name
            target = folder / output_name
            shutil.move(str(entry.stage_path), target)
            with Image.open(target) as check:
                check.load()
                if check.size != (entry.width, entry.height):
                    raise ValueError(f"Output dimensions changed: {entry.key}")
            old = output_hash_splits.setdefault(entry.output_pixel_hash, split)
            if old != split:
                raise ValueError(f"Output pixel duplicate across {old}/{split}: {entry.key}")
            per_class[category_names[entry.category_id]][split] += 1
            coco_images.append({"id": iid, "file_name": output_name,
                                "width": entry.width, "height": entry.height,
                                "source_split": entry.split, "source_file_name": entry.file_name,
                                "source_image_id": entry.original_image.get("source_image_id"),
                                "group_id": entry.group_id, "risk_group_id": entry.risk_group_id,
                                "species_label": category_names[entry.category_id],
                                "source": entry.original_image.get("source", "unknown")})
            for cid, box in entry.output_boxes:
                x, y, w, h = box
                if not (x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= entry.width + 0.001
                        and y + h <= entry.height + 0.001):
                    raise ValueError(f"Invalid output box for {entry.key}: {box}")
                coco_annotations.append({"id": len(coco_annotations) + 1, "image_id": iid,
                                         "category_id": cid, "bbox": box,
                                         "area": round(w * h, 4), "iscrowd": 0})
        document = {"info": {"version": "3.0", "description": "Audited squirrel-v2 multiclass export"},
                    "licenses": documents_licenses(decisions), "categories": categories,
                    "images": coco_images, "annotations": coco_annotations}
        ann_path = folder / "_annotations.coco.json"
        ann_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        annotation_hashes[split] = sha256_bytes(ann_path.read_bytes())
        split_counts[split] = len(coco_images)
        annotation_counts[split] = len(coco_annotations)

    if any(not all(row[s] > 0 for s in SPLITS) for row in per_class.values()):
        raise ValueError("An output split is missing one or more classes")
    marker = {"version": 3, "name": "squirrel-v2-audited-multiclass", "qa_status": "passed",
              "source_fingerprint": fingerprint, "source_image_fingerprint": decisions["source_image_fingerprint"],
              "audit_decisions_sha256": decisions_sha,
              "num_classes": len(categories), "categories": [c["name"] for c in categories],
              "split_counts": split_counts, "annotation_counts": annotation_counts,
              "annotation_hashes": annotation_hashes, "species_distribution": per_class,
              "risk_grouping": {"perceptual_hash_max_distance": PHASH_GROUP_DISTANCE,
                                "includes_observation_groups": True,
                                "includes_reviewed_related_scenes": True},
              "preprocessing": {"long_side": LONG_SIDE, "rgb": True, "jpeg_quality": 92,
                                "exif_oriented": True, "upscaled": False}}
    (destination / "decisions_applied.json").write_text(json.dumps({"removed": removed, "moves": moves}, indent=2) + "\n")
    summary = {"source_fingerprint": fingerprint, "audit_decisions_sha256": decisions_sha,
               "input_images": input_image_count, "output_images": sum(split_counts.values()),
               "output_annotations": sum(annotation_counts.values()), "removed": Counter(r["reason"] for r in removed),
               "moved_images": len(moves), "perceptual_edges_grouped": tight_edges, "perceptual_group_distance": PHASH_GROUP_DISTANCE,
               "species_distribution": per_class, "split_counts": split_counts}
    summary["removed"] = dict(summary["removed"])
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in entries:
        group_counts[category_names[entry.category_id]][entry.risk_group_id] += 1
    summary["largest_risk_group_per_class"] = {
        name: max(groups.values(), default=0) for name, groups in group_counts.items()
    }
    (destination / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (destination / ".complete.json").write_text(json.dumps(marker, indent=2) + "\n")
    return summary


def documents_licenses(decisions: dict[str, Any]) -> list[dict[str, Any]]:
    # The input combines several source licenses; preserve per-image attribution in
    # source metadata rather than incorrectly assigning one license to every image.
    return [{"id": 1, "name": "See source manifest for per-image license"}]


def export_dataset(source: Path, decisions_path: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"Output exists; keep versioned exports immutable: {destination}")
    fingerprint = source_fingerprint(source)
    decision_bytes = decisions_path.read_bytes()
    decisions_sha = sha256_bytes(decision_bytes)
    decisions = json.loads(decision_bytes)
    if decisions.get("source_fingerprint") != fingerprint:
        raise ValueError("Audit decisions do not match the source COCO fingerprint")
    if decisions.get("qa_passed") is not True or decisions.get("unresolved"):
        raise ValueError("Audit gate is not passed; resolve or quarantine the reported cases first")
    documents, categories = load_source(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="squirrel-v2-audited-", dir=destination.parent) as tmp:
        staging_root = Path(tmp)
        entries, removed = prepare_entries(source, documents, decisions, staging_root / "staging")
        content_fingerprint = image_fingerprint(entries)
        if decisions.get("source_image_fingerprint") != content_fingerprint:
            raise ValueError("Audit decisions do not match the source image fingerprint")
        entries = dedupe_exact(entries, removed)
        components, tight_edges = risk_components(entries, decisions)
        moves = assign_splits(entries, components)
        if not entries:
            raise ValueError("No usable images remain")
        summary = write_export(entries, categories, staging_root / "export", fingerprint,
                               decisions, decisions_sha, removed, moves, tight_edges,
                               sum(len(doc["images"]) for doc in documents.values()))
        (staging_root / "export").replace(destination)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--decisions", type=Path, default=DECISIONS)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    summary = export_dataset(args.source, args.decisions, args.destination)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
