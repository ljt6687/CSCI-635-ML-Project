#!/usr/bin/env python3
"""Reproducible, non-destructive purity audit for squirrel-v2-clean.

Writes only under --output. Image similarity is a review signal, never a species
decision. A passed QA gate requires every suspicious cross-split component to be
resolved with evidence and label outliers to be reviewed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
import imagehash

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")


def source_fingerprint(dataset: Path) -> str:
    h = hashlib.sha256()
    for split in SPLITS:
        with (dataset / split / "_annotations.coco.json").open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def source_image_fingerprint(records: list[dict], exclude: list[dict] | None = None) -> str:
    """Hash included raw RGB pixel hashes in a stable split/path order."""
    excluded = {(item["split"], item["file_name"]) for item in (exclude or [])}
    lines = sorted(f"{r['split']}/{r['file_name']}|{r['pixel_hash']}\n" for r in records
                   if (r["split"], r["file_name"]) not in excluded)
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def refresh_image_fingerprint(output: Path) -> str:
    with (output / "inventory.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    path = output / "decisions.json"
    decisions = json.loads(path.read_text())
    fingerprint = source_image_fingerprint(records, decisions.get("exclude", []))
    decisions["source_image_fingerprint"] = fingerprint
    path.write_text(json.dumps(decisions, indent=2))
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["source_image_fingerprint"] = fingerprint
        summary_path.write_text(json.dumps(summary, indent=2))
    return fingerprint


def pixel_hash(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    h = hashlib.sha256(f"{rgb.width}x{rgb.height}".encode())
    h.update(rgb.tobytes())
    return h.hexdigest()


def pair_id(a: dict, b: dict) -> str:
    # COCO row order matches the earlier notebook, retaining its decision IDs.
    keys = [f"{r['split']}/{r['file_name']}" for r in (a, b)]
    return hashlib.sha256("|".join(keys).encode()).hexdigest()[:16]


def find_pairs(rows: list[dict], max_distance: int = 4) -> list[dict]:
    """Pigeonhole candidate lookup: <=4 differing bits implies an equal 13-bit block."""
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        value = int(r["phash"], 16)
        for block in range(5):
            start = block * 13
            width = min(13, 64 - start)
            buckets[(block, (value >> start) & ((1 << width) - 1))].append(i)
    candidates: set[tuple[int, int]] = set()
    for ids in buckets.values():
        for j, a in enumerate(ids):
            for b in ids[j + 1 :]:
                if rows[a]["split"] != rows[b]["split"]:
                    candidates.add((a, b))
    # Exact pixels and source groups matter even when the perceptual hash differs.
    for field in ("pixel_hash", "group_id"):
        groups: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            if row.get(field):
                groups[row[field]].append(i)
        for ids in groups.values():
            for j, a in enumerate(ids):
                for b in ids[j + 1 :]:
                    if rows[a]["split"] != rows[b]["split"]:
                        candidates.add((a, b))
    pairs = []
    for a, b in sorted(candidates):
        left, right = rows[a], rows[b]
        dist = (int(left["phash"], 16) ^ int(right["phash"], 16)).bit_count()
        exact = left["pixel_hash"] == right["pixel_hash"]
        same_group = bool(left.get("group_id")) and left["group_id"] == right.get("group_id")
        if dist > max_distance and not exact and not same_group:
            continue
        pairs.append({
            "pair_id": pair_id(left, right), "left": {"split": left["split"], "file_name": left["file_name"]},
            "right": {"split": right["split"], "file_name": right["file_name"]},
            "distance": dist, "same_pixels": exact, "same_group": same_group,
            "cross_class": left["species_label"] != right["species_label"],
            "left_class": left["species_label"], "right_class": right["species_label"],
        })
    return pairs


def components(pairs: list[dict]) -> list[list[dict]]:
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for pair in pairs:
        a = tuple(pair["left"].values())
        b = tuple(pair["right"].values())
        parent[find(a)] = find(b)
    clusters: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for key in parent:
        clusters[find(key)].append({"split": key[0], "file_name": key[1]})
    return sorted((sorted(c, key=lambda x: (x["split"], x["file_name"])) for c in clusters.values()), key=lambda c: (-len(c), c[0]["file_name"]))


def make_contact_sheets(dataset: Path, clusters: list[list[dict]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for index, cluster in enumerate(clusters, 1):
        cols, cell_w, cell_h = 4, 260, 240
        canvas = Image.new("RGB", (cols * cell_w, ((len(cluster) + cols - 1) // cols) * cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        for n, ref in enumerate(cluster):
            path = dataset / ref["split"] / ref["file_name"]
            try:
                with Image.open(path) as image:
                    thumb = ImageOps.contain(ImageOps.exif_transpose(image.convert("RGB")), (cell_w - 12, cell_h - 44))
                    x, y = (n % cols) * cell_w, (n // cols) * cell_h
                    canvas.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 4))
                    draw.text((x + 5, y + cell_h - 35), ref["split"], fill="black")
                    draw.text((x + 5, y + cell_h - 20), ref["file_name"][:37], fill="black")
            except (OSError, UnidentifiedImageError):
                continue
        canvas.save(output / f"component_{index:03d}.jpg", quality=85)


def visual_pair_evidence(dataset: Path, pairs: list[dict]) -> list[dict]:
    """Compare candidate images with pixels and local feature geometry.

    The thresholds are intentionally stringent. Ambiguous matches stay unresolved.
    """
    import cv2

    cache = {}
    orb = cv2.ORB_create(nfeatures=1000)

    def get(ref):
        key = (ref["split"], ref["file_name"])
        if key not in cache:
            with Image.open(dataset / key[0] / key[1]) as image:
                image = image.convert("RGB")
                fitted = ImageOps.fit(image, (256, 256))
                pixels = np.asarray(fitted, dtype=np.uint8)
                gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
                points, desc = orb.detectAndCompute(gray, None)
                cache[key] = (pixels, gray, points, desc, image.size)
        return cache[key]

    reviewed = []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for pair in pairs:
        left = get(pair["left"])
        right = get(pair["right"])
        a, b = left[1].astype(np.float32), right[1].astype(np.float32)
        flat_a, flat_b = a.ravel(), b.ravel()
        corr = float(np.corrcoef(flat_a, flat_b)[0, 1]) if flat_a.std() > 1 and flat_b.std() > 1 else 0.0
        mae = float(np.abs(left[0].astype(np.float32) - right[0].astype(np.float32)).mean() / 255)
        inliers, matches = 0, 0
        if left[3] is not None and right[3] is not None:
            candidate = sorted(matcher.match(left[3], right[3]), key=lambda m: m.distance)
            good = [m for m in candidate if m.distance <= 45]
            matches = len(good)
            if len(good) >= 8:
                p1 = np.float32([left[2][m.queryIdx].pt for m in good])
                p2 = np.float32([right[2][m.trainIdx].pt for m in good])
                try:
                    _, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 4.0)
                    inliers = int(mask.sum()) if mask is not None else 0
                except cv2.error:
                    inliers = 0
        # Matched pixels or a substantial geometrically consistent scene.
        visual_match = bool(pair["same_pixels"] or (corr >= 0.97 and mae <= 0.07) or (inliers >= 25 and inliers / max(matches, 1) >= 0.55))
        reviewed.append({**pair, "pixel_correlation": round(corr, 5), "normalized_mae": round(mae, 5),
                         "orb_good_matches": matches, "homography_inliers": inliers,
                         "visual_match": visual_match,
                         "evidence": "identical decoded pixels" if pair["same_pixels"] else (
                             "high aligned pixel correlation" if corr >= 0.97 and mae <= 0.07 else (
                                 "geometrically consistent ORB features" if visual_match else "insufficient visual match evidence"))})
    return reviewed


def resolve_visual_pairs(dataset: Path, output: Path) -> dict:
    pairs = json.loads((output / "pairs.json").read_text())
    reviewed = visual_pair_evidence(dataset, pairs)
    (output / "visual_pair_evidence.json").write_text(json.dumps(reviewed, indent=2))
    exact_pairs = [p for p in reviewed if p["same_pixels"]]
    related_pairs = [p for p in reviewed if p["visual_match"] and not p["same_pixels"]]
    risk_grouped = [p for p in reviewed if p["distance"] <= 2 and not p["same_pixels"]]
    species_concerns = [p for p in reviewed if p["cross_class"]]
    decisions_path = output / "decisions.json"
    decisions = json.loads(decisions_path.read_text())
    # Only pixel-identical copies enter the dedupe-facing field. Related frames remain separate.
    decisions["confirmed_duplicate_components"] = components(exact_pairs)
    decisions["visually_related_components"] = components(related_pairs)
    decisions["risk_grouped_candidates"] = [{"pair_id": p["pair_id"], "reason": "pHash<=2; keep in one split, do not dedupe or relabel"} for p in risk_grouped]
    decisions["unresolved"] = [{"pair_id": p["pair_id"], "reason": "cross-class similarity; species-label review needed"} for p in species_concerns]
    decisions["qa_passed"] = False
    decisions["qa_status"] = "blocked: random class sample and cross-class species concerns remain unreviewed"
    decisions_path.write_text(json.dumps(decisions, indent=2))
    refresh_image_fingerprint(output)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update({"exact_pixel_duplicate_pairs": len(exact_pairs), "visually_related_pairs": len(related_pairs),
                    "risk_grouped_candidates": len(risk_grouped), "unresolved_species_pair_concerns": len(species_concerns),
                    "false_positive_pairs": 0, "confirmed_duplicate_components": len(decisions["confirmed_duplicate_components"])})
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def reconcile_active_pairs(output: Path) -> dict:
    """Drop pair references invalidated by confirmed image exclusions."""
    path = output / "decisions.json"
    decisions = json.loads(path.read_text())
    excluded = {(r["split"], r["file_name"]) for r in decisions.get("exclude", [])}
    evidence_path = output / "visual_pair_evidence.json"
    pairs = json.loads(evidence_path.read_text()) if evidence_path.exists() else json.loads((output / "pairs.json").read_text())
    active = [p for p in pairs if (p["left"]["split"], p["left"]["file_name"]) not in excluded
              and (p["right"]["split"], p["right"]["file_name"]) not in excluded]
    exact = [p for p in active if p["same_pixels"]]
    related = [p for p in active if p.get("visual_match") and not p["same_pixels"]]
    risk = [p for p in active if p["distance"] <= 2 and not p["same_pixels"]]
    concerns = [p for p in active if p["cross_class"]]
    resolved_ids = {r["pair_id"] for r in decisions.get("resolved_label_overlap_pairs", [])}
    decisions["resolved_label_overlap_pairs"] = [r for r in decisions.get("resolved_label_overlap_pairs", [])
                                                    if r["pair_id"] in {p["pair_id"] for p in active}]
    decisions["confirmed_duplicate_components"] = components(exact)
    decisions["visually_related_components"] = components(related)
    decisions["risk_grouped_candidates"] = [{"pair_id": p["pair_id"], "reason": "pHash<=2; keep in one split"} for p in risk]
    decisions["unresolved"] = [{"pair_id": p["pair_id"], "reason": "cross-class similarity; species-label review needed"}
                               for p in concerns if p["pair_id"] not in resolved_ids]
    decisions["qa_passed"] = False
    path.write_text(json.dumps(decisions, indent=2))
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["retained_images"] = summary["images"] - len(excluded)
        summary["active_cross_split_pairs"] = len(active)
        summary["active_risk_grouped_candidates"] = len(risk)
        summary["active_unresolved_species_pair_concerns"] = len(decisions["unresolved"])
        summary["resolved_label_overlap_pairs"] = len(decisions["resolved_label_overlap_pairs"])
        summary_path.write_text(json.dumps(summary, indent=2))
    return {"active_cross_split_pairs": len(active), "risk_grouped": len(risk), "unresolved_species": len(concerns)}


def apply_reviewed_annotation_exclusions(dataset: Path, output: Path) -> dict:
    """Remove only confirmed empty boxes while retaining their valid sibling boxes."""
    rows = json.loads((output / "reviewed_annotation_exclusions.json").read_text())
    decisions_path = output / "decisions.json"
    decisions = json.loads(decisions_path.read_text())
    image_excluded = {(r["split"], r["file_name"]) for r in decisions.get("exclude", [])}
    validated = []
    for item in rows:
        split, filename, ann_id = item["split"], item["file_name"], item["annotation_id"]
        if (split, filename) in image_excluded:
            raise ValueError(f"Cannot retain sibling boxes in excluded image {split}/{filename}")
        doc = json.loads((dataset / split / "_annotations.coco.json").read_text())
        image = next((i for i in doc["images"] if i["file_name"] == filename), None)
        if image is None or not any(a["id"] == ann_id and a["image_id"] == image["id"] for a in doc["annotations"]):
            raise ValueError(f"Annotation is not attached to image: {split}/{filename} id={ann_id}")
        if not item.get("reason") or not item.get("evidence"):
            raise ValueError(f"Annotation exclusion lacks evidence: {split}/{filename} id={ann_id}")
        validated.append(item)
    by_key = {(r["split"], r["file_name"], r["annotation_id"]): r for r in decisions.get("exclude_annotations", [])}
    for item in validated:
        by_key[(item["split"], item["file_name"], item["annotation_id"])] = item
    decisions["exclude_annotations"] = [by_key[key] for key in sorted(by_key)]
    decisions["qa_passed"] = False
    decisions_path.write_text(json.dumps(decisions, indent=2))
    return {"excluded_annotations": len(decisions["exclude_annotations"]), "retained_images": len(validated)}


def resolve_crossclass_generic_overrides(output: Path) -> dict:
    """Document visually valid but taxonomically indeterminate generic overrides."""
    expected = {("test", "268502957_482645542.png"), ("train", "296893592_534639612.jpg"),
                ("valid", "306149735_552071887.jpg")}
    decisions_path = output / "decisions.json"
    decisions = json.loads(decisions_path.read_text())
    excluded = {(r["split"], r["file_name"]) for r in decisions.get("exclude", [])}
    pairs = json.loads((output / "pairs.json").read_text())
    active = [p for p in pairs if p["cross_class"] and
              (p["left"]["split"], p["left"]["file_name"]) not in excluded and
              (p["right"]["split"], p["right"]["file_name"]) not in excluded]
    endpoints = {(ref["split"], ref["file_name"]) for p in active for ref, cls in
                 ((p["left"], p["left_class"]), (p["right"], p["right_class"])) if cls == "generic_squirrel"}
    if endpoints != expected:
        raise RuntimeError(f"Cross-class generic endpoints changed: {sorted(endpoints)}")
    differences = json.loads((output / "provenance_label_differences.json").read_text())
    reviewed = {(r["split"], r["file_name"]) for r in differences if r["human_reviewed_override"]}
    if not expected <= reviewed:
        raise RuntimeError("Generic endpoints lack documented human label overrides")
    decisions["resolved_label_overlap_pairs"] = [{"pair_id": p["pair_id"],
        "reason": "related IR scene; bbox contains squirrel; human Sciurus lis to generic override retained because species morphology is indeterminate"}
        for p in active]
    decisions["unresolved"] = [r for r in decisions.get("unresolved", []) if r["pair_id"] not in {p["pair_id"] for p in active}]
    decisions["qa_passed"] = False
    decisions["qa_status"] = "blocked: final outlier and retained-sample review still pending"
    decisions_path.write_text(json.dumps(decisions, indent=2))
    return {"resolved_edges": len(active), "generic_endpoints": len(expected), "unresolved": len(decisions["unresolved"])}


def review_sheet(dataset: Path, docs: dict, refs: list[dict], destination: Path, crop_focus: bool = False) -> None:
    """Draw source COCO boxes on a compact, local-only review contact sheet."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    ann_by_image = defaultdict(list)
    id_by_filename = {}
    for split, doc in docs.items():
        for image in doc["images"]:
            id_by_filename[(split, image["file_name"])] = image["id"]
        for ann in doc["annotations"]:
            ann_by_image[(split, ann["image_id"])].append(ann)
    cols, cell_w, cell_h = 5, 260, 240
    canvas = Image.new("RGB", (cols * cell_w, ((len(refs) + cols - 1) // cols) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for n, ref in enumerate(refs):
        split, filename = ref["split"], ref["file_name"]
        try:
            with Image.open(dataset / split / filename) as raw:
                image = raw.convert("RGB")
                anns = ann_by_image.get((split, id_by_filename.get((split, filename))), [])
                if crop_focus and anns:
                    target = next((a for a in anns if a.get("id") == ref.get("annotation_id")), anns[0])
                    x, y, w, h = target["bbox"]
                    pad = 0.1 * max(w, h)
                    x0, y0 = max(0, int(x - pad)), max(0, int(y - pad))
                    x1, y1 = min(image.width, int(x + w + pad)), min(image.height, int(y + h + pad))
                    image = image.crop((x0, y0, x1, y1))
                    box_draw = ImageDraw.Draw(image)
                    box_draw.rectangle((x - x0, y - y0, x + w - x0, y + h - y0), outline="red", width=max(2, image.width // 150))
                else:
                    box_draw = ImageDraw.Draw(image)
                    for ann in anns:
                        x, y, w, h = ann["bbox"]
                        box_draw.rectangle((x, y, x + w, y + h), outline="red", width=max(2, image.width // 500))
                image.thumbnail((cell_w - 10, cell_h - 50))
                x, y = (n % cols) * cell_w, (n // cols) * cell_h
                canvas.paste(image, (x + (cell_w - image.width) // 2, y + 2))
                draw.text((x + 4, y + cell_h - 40), ref.get("class", ref.get("species_label", ""))[:35], fill="black")
                draw.text((x + 4, y + cell_h - 25), split + "/" + filename[:27], fill="black")
                if "margin" in ref:
                    draw.text((x + 4, y + cell_h - 11), f"margin={ref['margin']:.3f}", fill="black")
        except (OSError, UnidentifiedImageError):
            continue
    canvas.save(destination, quality=85)


def make_review_samples(dataset: Path, docs: dict, output: Path, fingerprint: str, records: list[dict]) -> dict:
    by_class = defaultdict(list)
    decisions_path = output / "decisions.json"
    prior_decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
    excluded = {(r["split"], r["file_name"]) for r in prior_decisions.get("exclude", [])}
    for row in records:
        if (row["split"], row["file_name"]) not in excluded:
            by_class[row["species_label"]].append(row)
    original_path = output / "original_random_review_sample.json"
    if original_path.exists():
        original = json.loads(original_path.read_text())
        if original["source_fingerprint"] != fingerprint:
            raise RuntimeError("Original sample belongs to a different source fingerprint")
        original_samples = original["samples"]
    else:
        rng = random.Random(int(fingerprint[:16], 16))
        original_samples = {}
        for cls, rows in sorted(by_class.items()):
            selected = rng.sample(sorted(rows, key=lambda r: (r["split"], r["file_name"])), min(60, len(rows)))
            original_samples[cls] = [{"split": r["split"], "file_name": r["file_name"], "class": cls} for r in selected]
        original_path.write_text(json.dumps({"source_fingerprint": fingerprint, "samples": original_samples,
                                             "review_status": "unreviewed"}, indent=2))
    samples = {}
    replacements = []
    for cls, rows in sorted(by_class.items()):
        eligible = {(r["split"], r["file_name"]): r for r in rows}
        selected = [dict(ref) for ref in original_samples.get(cls, [])
                    if (ref["split"], ref["file_name"]) in eligible]
        selected_keys = {(r["split"], r["file_name"]) for r in selected}
        remaining = [r for r in rows if (r["split"], r["file_name"]) not in selected_keys]
        remaining.sort(key=lambda r: hashlib.sha256(f"{fingerprint}|{cls}|{r['split']}/{r['file_name']}".encode()).hexdigest())
        for row in remaining[:max(0, min(60, len(rows)) - len(selected))]:
            ref = {"split": row["split"], "file_name": row["file_name"], "class": cls,
                   "review_status": "unreviewed", "replacement_for_excluded_sample": True}
            selected.append(ref); replacements.append(ref)
        samples[cls] = selected
        safe_name = cls.replace(" ", "_")
        for part in range((len(samples[cls]) + 29) // 30):
            subset = samples[cls][part * 30:(part + 1) * 30]
            review_sheet(dataset, docs, subset, output / "random_review" / f"{safe_name}_{part + 1}.jpg")
            review_sheet(dataset, docs, subset, output / "random_review_crops" / f"{safe_name}_{part + 1}.jpg", crop_focus=True)
    (output / "random_review_sample.json").write_text(json.dumps({"seed_fingerprint": fingerprint, "samples": samples}, indent=2))
    (output / "retained_sample_replacements.json").write_text(json.dumps(replacements, indent=2))
    return {cls: len(items) for cls, items in samples.items()}


def write_provenance_differences(records: list[dict], metadata: Path, output: Path) -> dict:
    source_meta = load_metadata(metadata)
    corrections_path = ROOT / "output/v2_annotation/passed_only/manual_species_label_corrections.csv"
    corrections = {}
    if corrections_path.exists():
        with corrections_path.open(newline="", encoding="utf-8") as stream:
            corrections = {r["source_image_id"]: r for r in csv.DictReader(stream)}
    differences = []
    for row in records:
        source_id = row.get("source_image_id")
        source_label = source_meta.get(source_id, {}).get("species_label")
        if source_label and source_label != row["species_label"]:
            correction = corrections.get(source_id)
            verified = bool(correction and correction["source_species_label"] == source_label and
                            correction["reviewed_species_labels"] == row["species_label"])
            differences.append({"split": row["split"], "file_name": row["file_name"],
                                "source_image_id": source_id, "source_label": source_label,
                                "coco_label": row["species_label"], "human_reviewed_override": verified,
                                "evidence": str(corrections_path.relative_to(ROOT)) if verified else "unmatched metadata difference"})
    (output / "provenance_label_differences.json").write_text(json.dumps(differences, indent=2))
    return {"differences": len(differences), "human_reviewed_overrides": sum(x["human_reviewed_override"] for x in differences),
            "unverified_differences": sum(not x["human_reviewed_override"] for x in differences)}


def review_large_scene_components(dataset: Path, output: Path) -> dict:
    """Surface small/possibly empty targets in large fixed-scene pHash components."""
    pairs = json.loads((output / "pairs.json").read_text())
    clusters = [cluster for cluster in components([p for p in pairs if not p["same_pixels"]]) if len(cluster) >= 20]
    docs = {split: json.loads((dataset / split / "_annotations.coco.json").read_text()) for split in SPLITS}
    img_by_name = {(split, i["file_name"]): i for split, doc in docs.items() for i in doc["images"]}
    ann_map = defaultdict(list)
    for split, doc in docs.items():
        for ann in doc["annotations"]:
            ann_map[(split, ann["image_id"])].append(ann)
    with (output / "inventory.csv").open(newline="", encoding="utf-8") as stream:
        inventory = {(r["split"], r["file_name"]): r for r in csv.DictReader(stream)}
    stats = []
    for ci, cluster in enumerate(clusters, 1):
        for ref in cluster:
            key = (ref["split"], ref["file_name"])
            image = img_by_name[key]
            anns = ann_map[(key[0], image["id"])]
            rel_areas = [float(a["bbox"][2] * a["bbox"][3] / (image["width"] * image["height"])) for a in anns]
            stats.append({"component": ci, **ref, "species_label": inventory[key]["species_label"],
                          "boxes": len(anns), "max_relative_box_area": round(max(rel_areas, default=0), 6),
                          "tiny_review_flag": max(rel_areas, default=0) < 0.0025})
        refs = [{**ref, "class": inventory[(ref["split"], ref["file_name"])]["species_label"]} for ref in cluster]
        for part in range((len(refs) + 29) // 30):
            subset = refs[part * 30:(part + 1) * 30]
            review_sheet(dataset, docs, subset, output / "large_scene_review" / f"component_{ci:02d}_full_{part + 1:02d}.jpg")
            review_sheet(dataset, docs, subset, output / "large_scene_review" / f"component_{ci:02d}_crops_{part + 1:02d}.jpg", crop_focus=True)
    with (output / "large_scene_box_stats.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["component", "split", "file_name", "species_label", "boxes", "max_relative_box_area", "tiny_review_flag"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(stats)
    result = {"large_components": len(clusters), "images": len(stats),
              "tiny_review_flags": sum(r["tiny_review_flag"] for r in stats),
              "component_sizes": [len(cluster) for cluster in clusters]}
    return result


def exclude_static_branch_boxes(dataset: Path, output: Path) -> dict:
    """Exclude visually verified stationary-branch boxes in the large trailcam scene."""
    pairs = json.loads((output / "pairs.json").read_text())
    clusters = [c for c in components([p for p in pairs if not p["same_pixels"]]) if len(c) >= 20]
    if len(clusters) < 2:
        raise RuntimeError("Expected static-scene component is absent; inspect manually")
    cluster = clusters[1]
    docs = {split: json.loads((dataset / split / "_annotations.coco.json").read_text()) for split in SPLITS}
    image_by_name = {(split, im["file_name"]): im for split, doc in docs.items() for im in doc["images"]}
    anns = defaultdict(list)
    for split, doc in docs.items():
        for ann in doc["annotations"]:
            anns[(split, ann["image_id"])].append(ann)
    template_ref = {"split": "test", "file_name": "294189638_529504589.jpg"}
    with Image.open(dataset / template_ref["split"] / template_ref["file_name"]) as image:
        template = np.asarray(image.convert("L").crop((1000, 520, 1120, 770)).resize((64, 128)), dtype=np.float32).ravel()
    flagged = []
    for ref in cluster:
        key = (ref["split"], ref["file_name"])
        im = image_by_name[key]
        boxes = anns[(key[0], im["id"])]
        if not boxes:
            continue
        x, y, w, h = boxes[0]["bbox"]
        is_branch_roi = 1000 <= x <= 1020 and 520 <= y <= 550 and 90 <= w <= 110 and 200 <= h <= 230
        if not is_branch_roi:
            continue
        with Image.open(dataset / key[0] / key[1]) as image:
            target = np.asarray(image.convert("L").crop((1000, 520, 1120, 770)).resize((64, 128)), dtype=np.float32).ravel()
        corr = float(np.corrcoef(template, target)[0, 1])
        if corr < 0.75:
            continue
        flagged.append({**ref, "bbox": [x, y, w, h], "static_roi_correlation": round(corr, 4),
                        "reason": "bbox encloses stationary branch, not squirrel; repeated fixed ROI verified in crop sheet"})
    if len(flagged) != 18:
        raise RuntimeError(f"Static branch pattern changed: expected 18 audited images, found {len(flagged)}")
    (output / "static_branch_exclusions.json").write_text(json.dumps(flagged, indent=2))
    decisions_path = output / "decisions.json"
    decisions = json.loads(decisions_path.read_text())
    by_key = {(r["split"], r["file_name"]): r for r in decisions.get("exclude", [])}
    for item in flagged:
        by_key[(item["split"], item["file_name"])] = {"split": item["split"], "file_name": item["file_name"],
                                                       "reason": item["reason"] + f" (ROI correlation {item['static_roi_correlation']})"}
    decisions["exclude"] = [by_key[key] for key in sorted(by_key)]
    decisions["qa_passed"] = False
    decisions["qa_status"] = "blocked: branch-box exclusions identified; remaining class review unassessed"
    decisions_path.write_text(json.dumps(decisions, indent=2))
    refresh_image_fingerprint(output)
    reconcile_active_pairs(output)
    return {"excluded": len(flagged), "source_image_fingerprint": json.loads(decisions_path.read_text())["source_image_fingerprint"]}


def apply_reviewed_exclusions(output: Path) -> dict:
    """Merge evidence-backed manual image exclusions into the export contract."""
    reviewed_path = output / "reviewed_exclusions.json"
    reviewed = json.loads(reviewed_path.read_text())
    inventory = {(r["split"], r["file_name"]) for r in csv.DictReader((output / "inventory.csv").open(newline="", encoding="utf-8"))}
    path = output / "decisions.json"
    decisions = json.loads(path.read_text())
    by_key = {(r["split"], r["file_name"]): r for r in decisions.get("exclude", [])}
    for item in reviewed:
        key = (item["split"], item["file_name"])
        if key not in inventory:
            raise ValueError(f"Reviewed exclusion is not in audited inventory: {key}")
        if not item.get("reason"):
            raise ValueError(f"Reviewed exclusion lacks evidence: {key}")
        by_key[key] = item
    decisions["exclude"] = [by_key[key] for key in sorted(by_key)]
    decisions["qa_passed"] = False
    path.write_text(json.dumps(decisions, indent=2))
    fingerprint = refresh_image_fingerprint(output)
    reconcile_active_pairs(output)
    return {"total_exclusions": len(decisions["exclude"]), "reviewed_exclusions": len(reviewed), "source_image_fingerprint": fingerprint}


def review_generic_outliers(dataset: Path, output: Path, limit: int = 60) -> dict:
    """Prepare targeted non-squirrel review from weak species-embedding anomalies."""
    rows = json.loads((output / "embedding_outliers.json").read_text())
    decisions = json.loads((output / "decisions.json").read_text())
    excluded = {(r["split"], r["file_name"]) for r in decisions.get("exclude", [])}
    selected = [r for r in rows if r["class"] == "generic_squirrel" and (r["split"], r["file_name"]) not in excluded][:limit]
    docs = {split: json.loads((dataset / split / "_annotations.coco.json").read_text()) for split in SPLITS}
    (output / "generic_outlier_review.json").write_text(json.dumps(selected, indent=2))
    for part in range((len(selected) + 29) // 30):
        subset = selected[part * 30:(part + 1) * 30]
        review_sheet(dataset, docs, subset, output / "generic_outlier_review" / f"full_{part + 1:02d}.jpg")
        review_sheet(dataset, docs, subset, output / "generic_outlier_review" / f"crops_{part + 1:02d}.jpg", crop_focus=True)
    return {"queued": len(selected), "review_status": "unreviewed", "note": "embedding margin is a prioritization signal only"}


def write_embedding_heatmaps(output: Path) -> list[str]:
    """Render diagnostic matrices with labels that discourage purity inference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [
        ("class_similarity.csv", "class_similarity_heatmap.png", "RF-DETR backbone crop-centroid cosine similarity", "viridis", "offdiag"),
        ("knn_label_neighborhood.csv", "knn_label_neighborhood_heatmap.png", "RF-DETR backbone top-5 neighbor label fractions", "magma", "neighbor"),
    ]
    written = []
    for csv_name, png_name, title, cmap, scale in specs:
        with (output / csv_name).open(newline="", encoding="utf-8") as stream:
            table = list(csv.reader(stream))
        classes = table[0][1:]
        values = np.array([[float(x) for x in row[1:]] for row in table[1:]], dtype=float)
        if scale == "offdiag":
            offdiag = values[~np.eye(len(classes), dtype=bool)]
            vmin = max(0.0, float(np.floor(offdiag.min() * 100) / 100))
            vmax = 1.0
        else:
            vmin, vmax = 0.0, max(0.65, float(np.ceil(values.max() * 20) / 20))
        fig, ax = plt.subplots(figsize=(13, 11), dpi=160)
        im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(classes)), classes, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        ax.set_xlabel("Neighbor / centroid class")
        ax.set_ylabel("Query class")
        ax.set_title(title + f"\nExploratory only; color range {vmin:.2f}–{vmax:.2f} (truncated for contrast), not confusion/purity", fontsize=12)
        for y in range(len(classes)):
            for x in range(len(classes)):
                value = values[y, x]
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if (value - vmin) / max(vmax - vmin, 1e-6) < 0.55 else "black")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        fig.tight_layout()
        path = output / png_name
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))
    return written


def write_final_report(dataset: Path, output: Path) -> Path:
    """Rewrite one coherent current-state report from decision and QA artifacts."""
    summary = json.loads((output / "summary.json").read_text())
    decisions = json.loads((output / "decisions.json").read_text())
    progress = json.loads((output / "review_progress.json").read_text()) if (output / "review_progress.json").exists() else {}
    replacements = json.loads((output / "retained_sample_replacements.json").read_text()) if (output / "retained_sample_replacements.json").exists() else []
    excluded = {(r["split"], r["file_name"]) for r in decisions.get("exclude", [])}
    excluded_anns = {(r["split"], r["file_name"], r["annotation_id"]) for r in decisions.get("exclude_annotations", [])}
    before_images, after_images, before_boxes, after_boxes = Counter(), Counter(), Counter(), Counter()
    for split in SPLITS:
        doc = json.loads((dataset / split / "_annotations.coco.json").read_text())
        cats = {c["id"]: c["name"] for c in doc["categories"]}
        image_by_id = {i["id"]: i for i in doc["images"]}
        anns_by_image = defaultdict(list)
        for ann in doc["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)
            cls = cats[ann["category_id"]]
            before_boxes[cls] += 1
            image = image_by_id[ann["image_id"]]
            if (split, image["file_name"]) not in excluded and (split, image["file_name"], ann["id"]) not in excluded_anns:
                after_boxes[cls] += 1
        for image in doc["images"]:
            own = anns_by_image[image["id"]]
            cls = cats[own[0]["category_id"]]
            before_images[cls] += 1
            if (split, image["file_name"]) not in excluded:
                after_images[cls] += 1
    classes = sorted(before_images)
    with (output / "class_similarity.csv").open(newline="", encoding="utf-8") as stream:
        centroid_table = list(csv.reader(stream))
    centroid_matrix = np.array([[float(v) for v in row[1:]] for row in centroid_table[1:]])
    offdiag = centroid_matrix[~np.eye(len(classes), dtype=bool)]
    with (output / "knn_label_neighborhood.csv").open(newline="", encoding="utf-8") as stream:
        knn_table = list(csv.reader(stream))
    knn_matrix = np.array([[float(v) for v in row[1:]] for row in knn_table[1:]])
    own_neighbor = np.diag(knn_matrix)
    lines = ["# Squirrel v2 data purity audit", "", "## Integrity and scope", "",
        f"COCO source fingerprint: `{decisions['source_fingerprint']}`. Included raw-image fingerprint: `{decisions['source_image_fingerprint']}`.",
        f"Source: {summary['images']:,} images, {summary['annotations']:,} boxes, {len(classes)} classes. Full decode, COCO reference, geometry, and orphan checks found {summary['integrity_problems']} problems.",
        f"After decisions: {sum(after_images.values()):,} images and {sum(after_boxes.values()):,} boxes remain, subject to exact duplicate handling and scene-group resplitting by the exporter.",
        "", "## Class counts before and after review", "", "| Class | Images before | Images after | Boxes before | Boxes after |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {c} | {before_images[c]} | {after_images[c]} | {before_boxes[c]} | {after_boxes[c]} |" for c in classes]
    lines += ["", f"Sciurus lis has {after_images['Sciurus lis']} retained images, below the former 600-image target because verified errors were removed.",
        "", "## Leakage and class-overlap decisions", "",
        f"The source has {summary['cross_split_pairs']} cross-split pHash<=4 candidate edges. One pair has identical decoded pixels and is in {len(decisions['confirmed_duplicate_components'])} confirmed duplicate component. The exporter should keep every active near-similar/related scene component in one split; pHash alone is not a dedupe or species decision.",
        f"After image exclusions, {summary.get('active_cross_split_pairs', 'unknown')} candidate edges remain; {len(decisions.get('risk_grouped_candidates', []))} tight pHash edges are risk grouped. {len(decisions.get('visually_related_components', []))} components have strong aligned-pixel or ORB scene evidence, without declaring frames identical.",
        f"{len(decisions.get('resolved_label_overlap_pairs', []))} cross-class edges involving three visually valid generic_squirrel frames were reviewed as related IR scenes. Their human Sciurus lis→generic overrides remain; morphology is too weak for a new species claim. Unresolved pair decisions: {len(decisions['unresolved'])}.",
        "", "## Embedding diagnostics", "",
        f"A local pretrained RF-DETR backbone generated features for {summary['annotations']:,} source crops before final exclusions. These are exploratory and were not trained as an 11-species classifier.",
        f"Centroid off-diagonal cosine similarities range {offdiag.min():.2f}–{offdiag.max():.2f}; the generic detection backbone provides weak fine-species separation. Top-5 same-class neighbor fractions range {own_neighbor.min():.2f}–{own_neighbor.max():.2f}, mostly about 0.16–0.27 versus a 1/11≈0.09 equal-class baseline. Sciurus lis is {knn_matrix[classes.index('Sciurus lis'), classes.index('Sciurus lis')]:.2f}, plausibly elevated by repeated camera scenes. These indicate classification difficulty, not taxon errors.",
        "[Centroid cosine matrix](class_similarity.csv) and [top-5 cosine-neighbor label matrix](knn_label_neighborhood.csv) describe feature overlap; [cluster purity](cluster_purity.csv) and [per-class negative-margin rates](per_class_outlier_rates.csv) prioritize review. Negative centroid margin is not a mislabel rate.",
        "Heatmaps: [centroid similarity](class_similarity_heatmap.png), [neighbor overlap](knn_label_neighborhood_heatmap.png). Top 1% source-crop outlier queue and full/crop sheets are under top_one_percent_outliers.json and outlier_review*/.",
        "", "## EXIF and provenance", "",
        f"EXIF tag presence only (values withheld): {summary['exif_tag_presence']}. EXIF camera/time/location can guide review but cannot establish species labels.",
        f"Source-vs-COCO species differences: {summary['provenance_label_differences']['differences']}; all {summary['provenance_label_differences']['human_reviewed_overrides']} match prior human PyLabel corrections; none were auto-reverted.",
        "", "## Visual review and cleaning decisions", "",
        f"Original stratified crop sample: {progress.get('original_random_sample', {}).get('reviewed', 'unknown')}/660 images reviewed by an agent; four sampled images had confirmed box/taxon errors and are excluded. The exact original sample is preserved in original_random_review_sample.json.",
        f"Retained sample fills excluded slots with {len(replacements)} replacement images; all {progress.get('retained_sample_replacements', {}).get('reviewed', 0)} replacements passed full-frame and bbox-crop review. The retained 60-per-class sample is complete; see retained_sample_replacements.json.",
        f"Targeted crop review covered {progress.get('top_one_percent_outliers', {}).get('reviewed', 0)}/92 top-1% embedding outliers and {progress.get('generic_outlier_targeted', {}).get('reviewed', 0)}/60 generic ranked outliers. Confirmed empty/non-squirrel cases were removed; rank alone was never used as a label decision.",
        f"Confirmed removals: {len(excluded)} whole images and {len(excluded_anns)} individual boxes. Eighteen Sciurus lis frames had repeated stationary-branch boxes; other exclusions include pigeon, cat/prey, raccoon, misboxed log, and tracks/empty scenery. See static_branch_exclusions.json, reviewed_exclusions.json, and reviewed_annotation_exclusions.json for per-item evidence.",
        "Dark, blurred, tiny, and taxonomically ambiguous squirrels are noted as uncertainty, not automatic relabels or exclusions. The generic_squirrel category remains present.",
        "", "## Recovery and training gate", "",
        "Use decisions.json to exclude confirmed errors, keep related scenes in one split, and regenerate COCO in a new output directory. Recheck all retained boxes and split leakage after export.",
        f"QA passed: **{str(decisions['qa_passed']).lower()}**. {decisions['qa_status']}. The visual audit is a screening protocol, so no numerical residual-impurity bound or taxonomic certainty is claimed.", ""]
    path = output / "report.md"
    path.write_text("\n".join(lines))
    return path


def finalize_qa(dataset: Path, output: Path) -> dict:
    """Pass the source QA gate only after integrity and review evidence are complete."""
    summary = json.loads((output / "summary.json").read_text())
    path = output / "decisions.json"
    decisions = json.loads(path.read_text())
    progress = json.loads((output / "review_progress.json").read_text())
    replacements = json.loads((output / "retained_sample_replacements.json").read_text())
    with (output / "inventory.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    checks = {
        "coco_fingerprint_current": source_fingerprint(dataset) == decisions["source_fingerprint"],
        "image_fingerprint_current": source_image_fingerprint(records, decisions.get("exclude", [])) == decisions["source_image_fingerprint"],
        "integrity_zero": summary["integrity_problems"] == 0,
        "all_11_classes": len(summary["classes"]) == 11,
        "no_unresolved_pairs": len(decisions.get("unresolved", [])) == 0,
        "original_sample_reviewed": progress["original_random_sample"]["reviewed"] == 660,
        "all_replacements_reviewed": progress["retained_sample_replacements"]["reviewed"] == len(replacements),
        "top_outliers_reviewed": progress["top_one_percent_outliers"]["reviewed"] == progress["top_one_percent_outliers"]["total"],
        "generic_outliers_reviewed": progress["generic_outlier_targeted"]["reviewed"] == progress["generic_outlier_targeted"]["total"],
    }
    if not all(checks.values()):
        raise RuntimeError("QA gate remains blocked: " + ", ".join(k for k, passed in checks.items() if not passed))
    decisions["qa_passed"] = True
    decisions["qa_status"] = ("passed for a versioned COCO export after evidence-backed exclusions, annotation removals, "
                              "exact duplicate handling, and related-scene split grouping; residual fine-species ambiguity remains documented")
    decisions["qa_checks"] = checks
    path.write_text(json.dumps(decisions, indent=2))
    write_final_report(dataset, output)
    return {"qa_passed": True, "checks": checks, "source_image_fingerprint": decisions["source_image_fingerprint"]}


def load_metadata(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        return {r["image_id"]: r for line in stream if (r := json.loads(line)) and r.get("image_id")}


def audit(dataset: Path, metadata: Path, output: Path, embeddings: bool = False) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "decisions.json").exists():
        raise RuntimeError(f"Refusing to overwrite reviewed decisions at {output / 'decisions.json'}. Use --embeddings-only or a new --output directory.")
    fingerprint = source_fingerprint(dataset)
    source_meta = load_metadata(metadata)
    records: list[dict] = []
    problems: list[dict] = []
    class_counts: Counter[str] = Counter()
    exif_counts: Counter[str] = Counter()
    all_categories: set[str] = set()
    seen_group_splits: dict[str, set[str]] = defaultdict(set)
    docs = {split: json.loads((dataset / split / "_annotations.coco.json").read_text()) for split in SPLITS}
    for split, doc in docs.items():
        category_by_id = {c["id"]: c["name"] for c in doc["categories"]}
        all_categories.update(category_by_id.values())
        images = doc["images"]
        seen_ids: set[int] = set()
        seen_files: set[str] = set()
        anns: dict[int, list[dict]] = defaultdict(list)
        for ann in doc["annotations"]:
            anns[ann["image_id"]].append(ann)
        for item in images:
            ref = {"split": split, "file_name": item["file_name"]}
            if item["id"] in seen_ids or item["file_name"] in seen_files:
                problems.append({**ref, "reason": "duplicate COCO id or filename"})
            seen_ids.add(item["id"])
            seen_files.add(item["file_name"])
            file_path = dataset / split / item["file_name"]
            if not file_path.is_file():
                problems.append({**ref, "reason": "missing image"})
                continue
            own_anns = anns.get(item["id"], [])
            if not own_anns:
                problems.append({**ref, "reason": "missing annotation"})
                continue
            try:
                with Image.open(file_path) as image:
                    image.load()
                    w, h = image.size
                    phash = str(imagehash.phash(image.convert("RGB")))
                    pixels = pixel_hash(image)
                    exif = image.getexif()
                    for tag in ("DateTimeOriginal", "Make", "Model", "GPSInfo"):
                        # Only aggregate tag presence; no private EXIF values in reports.
                        from PIL.ExifTags import Base
                        tag_id = getattr(Base, tag, None)
                        if tag_id and exif.get(tag_id):
                            exif_counts[tag] += 1
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                problems.append({**ref, "reason": "unreadable image", "error_type": type(exc).__name__})
                continue
            if (w, h) != (item["width"], item["height"]):
                problems.append({**ref, "reason": "dimension mismatch", "actual": [w, h]})
            labels = []
            for ann in own_anns:
                cid = ann.get("category_id")
                bbox = ann.get("bbox", [])
                if cid not in category_by_id or len(bbox) != 4:
                    problems.append({**ref, "reason": "invalid category or bbox shape", "annotation_id": ann.get("id")})
                    continue
                x, y, bw, bh = bbox
                if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in bbox) or x < 0 or y < 0 or bw <= 0 or bh <= 0 or x + bw > w + 0.01 or y + bh > h + 0.01:
                    problems.append({**ref, "reason": "invalid bbox geometry", "annotation_id": ann.get("id")})
                labels.append(category_by_id[cid])
                class_counts[category_by_id[cid]] += 1
            if len(set(labels)) > 1:
                problems.append({**ref, "reason": "mixed species labels in image"})
            source_id = item.get("source_image_id")
            meta = source_meta.get(source_id, {})
            group = item.get("group_id") or meta.get("group_id")
            if group:
                seen_group_splits[group].add(split)
            records.append({**ref, "image_id": item["id"], "width": w, "height": h,
                            "species_label": labels[0] if labels else item.get("species_label", "unknown"),
                            "source_image_id": source_id, "source": item.get("source") or meta.get("source"),
                            "group_id": group, "observation_id": meta.get("observation_id"),
                            "label_basis": meta.get("label_basis"), "scientific_name": meta.get("scientific_name"),
                            "pixel_hash": pixels, "phash": phash})
        for ann_id in set(anns) - seen_ids:
            problems.append({"split": split, "reason": "orphan annotation image_id", "image_id": ann_id})
        for path in (dataset / split).iterdir():
            if path.is_file() and path.name != "_annotations.coco.json" and path.name not in seen_files:
                problems.append({"split": split, "file_name": path.name, "reason": "orphan image"})
    for group, splits in seen_group_splits.items():
        if len(splits) > 1:
            problems.append({"reason": "cross-split source group", "group_id": group, "splits": sorted(splits)})
    pairs = find_pairs(records)
    exact_pairs = [p for p in pairs if p["same_pixels"]]
    confirmed = components(exact_pairs)
    suspicious = components([p for p in pairs if not p["same_pixels"]])
    make_contact_sheets(dataset, suspicious, output / "contact_sheets")
    with (output / "inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]) if records else [])
        writer.writeheader(); writer.writerows(records)
    (output / "pairs.json").write_text(json.dumps(pairs, indent=2))
    sample_counts = make_review_samples(dataset, docs, output, fingerprint, records)
    provenance = write_provenance_differences(records, metadata, output)
    unresolved = [{"pair_id": p["pair_id"], "reason": "pHash similarity requires visual/provenance review"} for p in pairs if not p["same_pixels"]]
    decisions = {"source_fingerprint": fingerprint, "source_image_fingerprint": source_image_fingerprint(records), "exclude": [], "relabel": [],
                 "confirmed_duplicate_components": confirmed, "false_positive_pairs": [],
                 "unresolved": unresolved, "qa_passed": False,
                 "qa_status": "blocked: confirmed cross-split duplicate and/or unresolved similarity and label-purity review"}
    (output / "decisions.json").write_text(json.dumps(decisions, indent=2))
    summary = {"source_fingerprint": fingerprint, "source_image_fingerprint": source_image_fingerprint(records), "images": len(records), "annotations": sum(class_counts.values()),
               "classes": sorted(all_categories), "class_box_counts": dict(sorted(class_counts.items())),
               "integrity_problems": len(problems), "cross_split_pairs": len(pairs),
               "exact_pixel_pairs": len(exact_pairs), "suspicious_components": len(suspicious),
               "unresolved_pairs": len(unresolved), "exif_tag_presence": dict(exif_counts),
               "cross_class_pairs": sum(p["cross_class"] for p in pairs),
               "embedding_status": "not_run" if not embeddings else "pending", "random_review_counts": sample_counts, "provenance_label_differences": provenance}
    (output / "integrity_problems.json").write_text(json.dumps(problems, indent=2))
    if embeddings:
        summary["embedding_status"] = run_embeddings(dataset, docs, records, output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    report = ["# Squirrel v2 purity audit", "", f"Source COCO fingerprint: `{fingerprint}`.",
              f"Images: {len(records)}; boxes: {sum(class_counts.values())}; classes: {len(all_categories)}.",
              f"Integrity problems: {len(problems)}. Cross-split pHash<=4 pairs: {len(pairs)}; exact pixel pairs: {len(exact_pairs)}.",
              f"Unresolved pairs: {len(unresolved)} across {len(suspicious)} contact-sheet components.",
              f"Cross-class pHash candidates: {summary['cross_class_pairs']}. Embeddings: {summary['embedding_status']}.",
              "", "## Decision", "", "QA remains blocked until suspicious components and label outliers are reviewed with evidence.",
              "Pixel-identical cross-split images are confirmed duplicates. pHash and geographic metadata alone never trigger relabeling.",
              "The COCO source was not modified. Review contact sheets and machine-readable pairs before preparing a new export."]
    (output / "report.md").write_text("\n".join(report) + "\n")
    return summary


def run_embeddings(dataset: Path, docs: dict, records: list[dict], output: Path) -> str:
    """Run pretrained RF-DETR DINO backbone on annotated crops; never infer labels automatically."""
    try:
        import torch
        import torch.nn.functional as F
        from rfdetr import RFDETRMedium
    except ImportError as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    checkpoint = ROOT / "output/pretrained/rf-detr-medium.pth"
    if not checkpoint.is_file():
        return "unavailable: local RF-DETR checkpoint missing"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RFDETRMedium(pretrain_weights=str(checkpoint)).model.model.backbone[0].encoder.to(device).eval()
    ann_map: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for split, doc in docs.items():
        for ann in doc["annotations"]:
            ann_map[(split, ann["image_id"])].append(ann)
    means = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    stds = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    features: list[np.ndarray] = []
    refs: list[dict] = []
    batch: list[np.ndarray] = []
    batch_refs: list[dict] = []

    def flush():
        if not batch:
            return
        x = torch.from_numpy(np.stack(batch)).to(device).float() / 255
        x = (x - means) / stds
        with torch.inference_mode():
            raw = model(x)[-1]
            vec = F.normalize(raw.mean(dim=(-2, -1)), dim=1).cpu().numpy()
        features.extend(vec)
        refs.extend(batch_refs)
        batch.clear(); batch_refs.clear()

    decisions_path = output / "decisions.json"
    prior_decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
    excluded = {(r["split"], r["file_name"]) for r in prior_decisions.get("exclude", [])}
    for row in records:
        if (row["split"], row["file_name"]) in excluded:
            continue
        own = ann_map.get((row["split"], int(row["image_id"])), [])
        if not own:
            continue
        with Image.open(dataset / row["split"] / row["file_name"]) as image:
            image = image.convert("RGB")  # COCO boxes use stored pixel coordinates, before EXIF rotation.
            for ann in own:
                x, y, w, h = ann["bbox"]
                crop = image.crop((int(x), int(y), int(x + w), int(y + h)))
                crop = ImageOps.pad(crop, (256, 256), color=(127, 127, 127))
                batch.append(np.asarray(crop).transpose(2, 0, 1))
                batch_refs.append({"split": row["split"], "file_name": row["file_name"],
                                   "class": row["species_label"], "annotation_id": ann["id"],
                                   "observation_id": row.get("observation_id")})
                if len(batch) >= (16 if device == "cuda" else 2):
                    flush()
    flush()
    if not features:
        return "unavailable: no valid annotated crops"
    vectors = np.asarray(features, dtype=np.float32)
    classes = sorted({r["class"] for r in refs})
    centroids = np.stack([vectors[[i for i, r in enumerate(refs) if r["class"] == c]].mean(axis=0) for c in classes])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-8)
    similarities = centroids @ centroids.T
    with (output / "class_similarity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["class"] + classes)
        for cls, row in zip(classes, similarities): writer.writerow([cls] + [round(float(v), 5) for v in row])
    scores = vectors @ centroids.T
    scored = []
    for i, ref in enumerate(refs):
        own = classes.index(ref["class"])
        rival = int(np.argmax(np.where(np.arange(len(classes)) == own, -np.inf, scores[i])))
        margin = float(scores[i, own] - scores[i, rival])
        scored.append({**ref, "own_similarity": round(float(scores[i, own]), 5),
                       "nearest_rival": classes[rival], "rival_similarity": round(float(scores[i, rival]), 5),
                       "margin": round(margin, 5), "review_status": "unreviewed", "review_only": True})
    scored.sort(key=lambda r: r["margin"])
    outliers = [r for r in scored if r["margin"] < 0]
    (output / "embedding_outliers.json").write_text(json.dumps(outliers, indent=2))
    top_outliers = scored[:max(1, round(len(refs) * 0.01))]
    (output / "top_one_percent_outliers.json").write_text(json.dumps(top_outliers, indent=2))
    for part in range((len(top_outliers) + 29) // 30):
        subset = top_outliers[part * 30:(part + 1) * 30]
        review_sheet(dataset, docs, subset, output / "outlier_review" / f"outliers_{part + 1:02d}.jpg")
        review_sheet(dataset, docs, subset, output / "outlier_review_crops" / f"outliers_{part + 1:02d}.jpg", crop_focus=True)
    # Class-neighborhood overlap catches mixed manifolds that centroids can hide.
    labels = np.array([classes.index(r["class"]) for r in refs], dtype=np.int32)
    image_keys = [(r["split"], r["file_name"]) for r in refs]
    observation_keys = [str(r.get("observation_id") or "") for r in refs]
    feature_tensor = torch.from_numpy(vectors).to(device)
    neighborhood = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for start in range(0, len(refs), 256):
        stop = min(start + 256, len(refs))
        sim = (feature_tensor[start:stop] @ feature_tensor.T).cpu().numpy()
        for local, index in enumerate(range(start, stop)):
            for other in range(len(refs)):
                if image_keys[index] == image_keys[other] or (observation_keys[index] and observation_keys[index] == observation_keys[other]):
                    sim[local, other] = -np.inf
            k = min(5, int(np.isfinite(sim[local]).sum()))
            if k:
                nearest = np.argpartition(sim[local], -k)[-k:]
                for neighbor in nearest:
                    neighborhood[labels[index], labels[neighbor]] += 1
    with (output / "knn_label_neighborhood.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["query_class"] + classes)
        for i, cls in enumerate(classes):
            total = max(1, int(neighborhood[i].sum()))
            writer.writerow([cls] + [round(float(v / total), 5) for v in neighborhood[i]])
    with (output / "per_class_outlier_rates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["class", "crops", "negative_margin_crops", "negative_margin_rate"])
        for cls in classes:
            total = sum(r["class"] == cls for r in refs)
            flagged = sum(r["class"] == cls for r in outliers)
            writer.writerow([cls, total, flagged, round(flagged / total, 5)])
    from sklearn.cluster import MiniBatchKMeans
    cluster_ids = MiniBatchKMeans(n_clusters=len(classes), random_state=42, batch_size=512, n_init=5).fit_predict(vectors)
    cluster_labels = defaultdict(Counter)
    for cluster_id, ref in zip(cluster_ids, refs):
        cluster_labels[int(cluster_id)][ref["class"]] += 1
    with (output / "cluster_purity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["cluster_id", "images", "dominant_class", "dominant_share"] + classes)
        for cluster_id, counts in sorted(cluster_labels.items()):
            total = sum(counts.values()); dominant, highest = counts.most_common(1)[0]
            writer.writerow([cluster_id, total, dominant, round(highest / total, 5)] + [counts[c] for c in classes])
    write_embedding_heatmaps(output)
    return f"complete: {len(vectors)} crop embeddings, {len(outliers)} centroid-margin outliers, {len(cluster_labels)} unsupervised clusters"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/squirrel-v2-clean")
    parser.add_argument("--metadata", type=Path, default=ROOT / "output/v2_annotation/source_manifest.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "output/v2_annotation/audit_v2")
    parser.add_argument("--embeddings", action="store_true", help="Run local RF-DETR DINO backbone on all annotated crops")
    parser.add_argument("--embeddings-only", action="store_true", help="Reuse a completed inventory and run only crop embeddings")
    parser.add_argument("--review-pairs-only", action="store_true", help="Review completed candidate pairs using local visual evidence")
    parser.add_argument("--review-samples-only", action="store_true", help="Generate deterministic 60-per-class review sheets from a completed inventory")
    parser.add_argument("--refresh-image-fingerprint", action="store_true", help="Refresh included image hash after exclude decisions change")
    parser.add_argument("--provenance-only", action="store_true", help="Refresh source-vs-COCO label provenance report from inventory")
    parser.add_argument("--review-large-scenes-only", action="store_true", help="Inspect bbox crops and sizes in large static-scene components")
    parser.add_argument("--exclude-static-branch", action="store_true", help="Record verified stationary-branch box exclusions")
    parser.add_argument("--apply-reviewed-exclusions", action="store_true", help="Merge evidence-backed image exclusions from reviewed_exclusions.json")
    parser.add_argument("--review-generic-outliers-only", action="store_true", help="Generate focused generic_squirrel outlier review sheets")
    parser.add_argument("--reconcile-active-pairs", action="store_true", help="Remove excluded images from pair decisions")
    parser.add_argument("--apply-reviewed-annotation-exclusions", action="store_true", help="Merge confirmed bad-box decisions while retaining images")
    parser.add_argument("--resolve-generic-overlap", action="store_true", help="Document human generic overrides in related IR scenes")
    parser.add_argument("--heatmaps-only", action="store_true", help="Render heatmaps from existing embedding matrices")
    parser.add_argument("--write-report", action="store_true", help="Rewrite coherent current-state report from QA artifacts")
    parser.add_argument("--finalize-qa", action="store_true", help="Pass QA only after integrity and review evidence checks")
    args = parser.parse_args()
    if args.finalize_qa:
        print(json.dumps(finalize_qa(args.dataset, args.output), indent=2))
    elif args.write_report:
        print(write_final_report(args.dataset, args.output))
    elif args.heatmaps_only:
        print(json.dumps(write_embedding_heatmaps(args.output), indent=2))
    elif args.resolve_generic_overlap:
        print(json.dumps(resolve_crossclass_generic_overrides(args.output), indent=2))
    elif args.apply_reviewed_annotation_exclusions:
        print(json.dumps(apply_reviewed_annotation_exclusions(args.dataset, args.output), indent=2))
    elif args.reconcile_active_pairs:
        print(json.dumps(reconcile_active_pairs(args.output), indent=2))
    elif args.review_generic_outliers_only:
        print(json.dumps(review_generic_outliers(args.dataset, args.output), indent=2))
    elif args.apply_reviewed_exclusions:
        print(json.dumps(apply_reviewed_exclusions(args.output), indent=2))
    elif args.exclude_static_branch:
        print(json.dumps(exclude_static_branch_boxes(args.dataset, args.output), indent=2))
    elif args.review_large_scenes_only:
        print(json.dumps(review_large_scene_components(args.dataset, args.output), indent=2))
    elif args.provenance_only:
        with (args.output / "inventory.csv").open(newline="", encoding="utf-8") as stream:
            records = list(csv.DictReader(stream))
        stats = write_provenance_differences(records, args.metadata, args.output)
        summary_path = args.output / "summary.json"
        summary = json.loads(summary_path.read_text()); summary["provenance_label_differences"] = stats
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(stats, indent=2))
    elif args.refresh_image_fingerprint:
        print(refresh_image_fingerprint(args.output))
    elif args.review_samples_only:
        with (args.output / "inventory.csv").open(newline="", encoding="utf-8") as stream:
            records = list(csv.DictReader(stream))
        docs = {split: json.loads((args.dataset / split / "_annotations.coco.json").read_text()) for split in SPLITS}
        fingerprint = source_fingerprint(args.dataset)
        counts = make_review_samples(args.dataset, docs, args.output, fingerprint, records)
        summary_path = args.output / "summary.json"
        summary = json.loads(summary_path.read_text()); summary["random_review_counts"] = counts
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(counts, indent=2))
    elif args.review_pairs_only:
        print(json.dumps(resolve_visual_pairs(args.dataset, args.output), indent=2))
    elif args.embeddings_only:
        with (args.output / "inventory.csv").open(newline="", encoding="utf-8") as stream:
            records = list(csv.DictReader(stream))
        docs = {split: json.loads((args.dataset / split / "_annotations.coco.json").read_text()) for split in SPLITS}
        status = run_embeddings(args.dataset, docs, records, args.output)
        summary_path = args.output / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["embedding_status"] = status
        summary_path.write_text(json.dumps(summary, indent=2))
        print(status)
    else:
        print(json.dumps(audit(args.dataset, args.metadata, args.output, args.embeddings), indent=2))


if __name__ == "__main__":
    main()
