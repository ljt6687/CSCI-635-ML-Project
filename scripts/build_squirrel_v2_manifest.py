#!/usr/bin/env python3
"""Build deterministic, read-only-source manifest for v2 squirrel annotation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
REQUIRED_FIELDS = ("image_id", "image_path", "source", "species_label", "label_basis")

NAMED_FOCAL_SPECIES = {
    "Callosciurus erythraeus",
    "Sciurus aureogaster",
    "Sciurus carolinensis",
    "Sciurus granatensis",
    "Sciurus griseus",
    "Sciurus lis",
    "Sciurus niger",
    "Sciurus vulgaris",
    "Tamiasciurus douglasii",
    "Tamiasciurus hudsonicus",
}


def normalize_class_label(raw_class: str | None, raw_scientific: str | None = None) -> str:
    """Normalize class label to 10 focal species or shared generic_squirrel category.

    Maps 'squirrel_generic' to 'generic_squirrel' and groups non-focal species
    (such as Glaucomys volans and Glaucomys sabrinus) under 'generic_squirrel'.
    """
    if raw_class == "squirrel_generic" or raw_class == "generic_squirrel":
        return "generic_squirrel"
    if raw_class in NAMED_FOCAL_SPECIES:
        return raw_class
    if raw_scientific in NAMED_FOCAL_SPECIES:
        return raw_scientific
    return "generic_squirrel"


def compute_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of file content in 1MB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> tuple[int | None, int | None, str | None, str | None]:
    """Inspect image with Pillow for readability, dimensions, and format.

    Returns:
        (width, height, format, error_message)
    """
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
            img_format = img.format
        return width, height, img_format, None
    except Exception as exc:
        return None, None, None, f"{type(exc).__name__}: {exc}"


def cluster_burst_groups(files: list[Path]) -> dict[Path, str]:
    """Group image files into conservative bursts based on sequence numbers in filenames.

    Files with consecutive or near-consecutive numbers (diff <= 5) are grouped together.
    """
    file_info: list[tuple[int | None, str, Path]] = []
    for f in files:
        m = re.search(r"(\d+)", f.stem)
        num = int(m.group(1)) if m else None
        file_info.append((num, f.name, f))

    numbered = [entry for entry in file_info if entry[0] is not None]
    unnum = [entry for entry in file_info if entry[0] is None]

    burst_map: dict[Path, str] = {}

    if numbered:
        numbered.sort(key=lambda entry: entry[0])  # type: ignore[arg-type]
        clusters: list[list[tuple[int, str, Path]]] = []
        curr_cluster: list[tuple[int, str, Path]] = [numbered[0]]  # type: ignore[list-item]

        for entry in numbered[1:]:
            prev_num = curr_cluster[-1][0]
            curr_num = entry[0]  # type: ignore[assignment]
            if curr_num - prev_num <= 5:
                curr_cluster.append(entry)  # type: ignore[arg-type]
            else:
                clusters.append(curr_cluster)
                curr_cluster = [entry]  # type: ignore[list-item]
        clusters.append(curr_cluster)

        for cluster in clusters:
            first_num = cluster[0][0]
            last_num = cluster[-1][0]
            if len(cluster) > 1:
                group_id = f"harrybaines_burst_IMG_{first_num:04d}_IMG_{last_num:04d}"
            else:
                group_id = f"harrybaines_burst_IMG_{first_num:04d}"
            for _, _, path in cluster:
                burst_map[path] = group_id

    for _, name, path in unnum:
        burst_map[path] = f"harrybaines_burst_{path.stem}"

    return burst_map


def process_inaturalist(
    repo_root: Path,
    report: dict[str, Any],
    seen_keys: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Process pilot iNaturalist images, joining against authoritative metadata and validating manifest."""
    source_dir = repo_root / "data" / "squirrel_sources" / "inaturalist"
    manifest_path = source_dir / "manifest.jsonl"
    obs_path = repo_root / "research" / "squirrel_expansion" / "metadata" / "inat_observations.jsonl"

    if not source_dir.is_dir():
        raise FileNotFoundError(f"iNaturalist source directory not found: {source_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"iNaturalist download manifest not found: {manifest_path}")
    if not obs_path.is_file():
        raise FileNotFoundError(f"Authoritative iNaturalist metadata not found: {obs_path}")

    # Load authoritative observations
    obs_lookup: dict[tuple[int, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    with obs_path.open("r", encoding="utf-8") as stream:
        for line_num, line in enumerate(stream, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                obs_obj = json.loads(line_str)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {obs_path} line {line_num}: {exc}") from exc
            obs_id = obs_obj.get("observation_id")
            for photo in obs_obj.get("photos", []):
                photo_id = photo.get("photo_id")
                if obs_id is not None and photo_id is not None:
                    obs_lookup[(int(obs_id), int(photo_id))] = (obs_obj, photo)

    # Load download manifest
    manifest_rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_num, line in enumerate(stream, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                row = json.loads(line_str)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {manifest_path} line {line_num}: {exc}") from exc
            manifest_rows.append(row)

    # Inventory disk files
    disk_images: dict[str, Path] = {}
    for p in source_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            disk_images[p.name] = p

    report["inaturalist"] = {
        "manifest_rows": len(manifest_rows),
        "disk_images": len(disk_images),
        "missing_files": 0,
        "unmatched_observations": 0,
        "species_mismatches": 0,
        "corrupt_images": 0,
    }

    results: list[dict[str, Any]] = []
    seen_files: set[str] = set()

    for item in manifest_rows:
        filename = item.get("filename")
        if not filename:
            raise ValueError(f"Manifest entry missing filename: {item}")
        seen_files.add(filename)

        image_file = disk_images.get(filename)
        status_flags: list[str] = ["image_level_label_provisional_for_boxes"]
        metadata_error: str | None = None
        image_error: str | None = None

        obs_id_raw = item.get("observation_id")
        photo_id_raw = item.get("photo_id")
        obs_id = int(obs_id_raw) if obs_id_raw is not None else None
        photo_id = int(photo_id_raw) if photo_id_raw is not None else None

        obs_meta: dict[str, Any] | None = None
        photo_meta: dict[str, Any] | None = None

        raw_scientific = item.get("scientific_name")

        if obs_id is not None and photo_id is not None and (obs_id, photo_id) in obs_lookup:
            obs_meta, photo_meta = obs_lookup[(obs_id, photo_id)]
            authoritative_scientific = obs_meta.get("scientific_name")
            if authoritative_scientific:
                raw_scientific = authoritative_scientific
            if item.get("scientific_name") and item.get("scientific_name") != authoritative_scientific:
                status_flags.append("manifest_species_mismatch")
                report["inaturalist"]["species_mismatches"] += 1
            status_flags.append("inat_metadata_verified")
        else:
            report["inaturalist"]["unmatched_observations"] += 1
            status_flags.append("unmatched_inat_observation")
            metadata_error = (
                f"Pair (observation_id={obs_id}, photo_id={photo_id}) not found in inat_observations.jsonl"
            )

        species_label = normalize_class_label(None, raw_scientific)

        if not image_file or not image_file.exists():
            report["inaturalist"]["missing_files"] += 1
            status_flags.append("image_file_missing")
            image_error = f"File {filename} missing on disk in {source_dir}"
            width, height, img_fmt, sha256_val = None, None, None, None
        else:
            width, height, img_fmt, img_err = inspect_image(image_file)
            if img_err:
                report["inaturalist"]["corrupt_images"] += 1
                status_flags.append("image_read_error")
                image_error = img_err
            sha256_val = compute_sha256(image_file)

        # Provenance attribution / licensing
        license_str = (
            (photo_meta.get("license") if photo_meta else None)
            or item.get("photo_license")
            or item.get("observation_license")
        )
        attribution_str = (
            (photo_meta.get("attribution") if photo_meta else None)
            or item.get("attribution")
        )
        source_url_str = (
            (obs_meta.get("observation_url") if obs_meta else None)
            or item.get("observation_url")
        )
        photo_url_str = (
            (photo_meta.get("original_url") if photo_meta else None)
            or item.get("original_url")
        )

        repo_rel_path = (source_dir / filename).relative_to(repo_root).as_posix()
        source_rel_path = filename

        if obs_id is not None and photo_id is not None:
            seen_keys.add((obs_id, photo_id))

        row: dict[str, Any] = {
            "image_id": f"inaturalist:{source_rel_path}",
            "image_path": repo_rel_path,
            "source": "inaturalist",
            "species_label": species_label,
            "scientific_name": raw_scientific,
            "label_basis": "inat_metadata",
            "species_label_provisional": True,
            "observation_id": obs_id,
            "photo_id": photo_id,
            "source_url": source_url_str,
            "photo_url": photo_url_str,
            "license": license_str,
            "attribution": attribution_str,
            "group_id": str(obs_id) if obs_id is not None else f"inat_{filename}",
            "image_sha256": sha256_val,
            "width": width,
            "height": height,
            "image_format": img_fmt,
            "status_flags": sorted(status_flags),
            "image_error": image_error,
            "metadata_error": metadata_error,
        }
        results.append(row)

    # Check for unmanifested disk images
    unmanifested = set(disk_images.keys()) - seen_files
    if unmanifested:
        report["inaturalist"]["unmanifested_disk_images"] = sorted(unmanifested)
        for fname in sorted(unmanifested):
            fpath = disk_images[fname]
            width, height, img_fmt, img_err = inspect_image(fpath)
            sha256_val = compute_sha256(fpath)
            repo_rel_path = fpath.relative_to(repo_root).as_posix()
            results.append({
                "image_id": f"inaturalist:{fname}",
                "image_path": repo_rel_path,
                "source": "inaturalist",
                "species_label": None,
                "scientific_name": None,
                "label_basis": "inat_metadata",
                "species_label_provisional": True,
                "observation_id": None,
                "photo_id": None,
                "source_url": None,
                "photo_url": None,
                "license": None,
                "attribution": None,
                "group_id": f"inat_{fname}",
                "image_sha256": sha256_val,
                "width": width,
                "height": height,
                "image_format": img_fmt,
                "status_flags": ["image_level_label_provisional_for_boxes", "unmanifested_disk_image", "unresolvable_metadata"],
                "image_error": img_err,
                "metadata_error": f"Image file {fname} on disk is not present in inaturalist manifest.jsonl",
            })

    return results


def process_inaturalist_bulk(
    repo_root: Path,
    report: dict[str, Any],
    seen_keys: set[tuple[int, int]],
    spot_check_sample_size: int = 50,
) -> list[dict[str, Any]]:
    """Process inaturalist_bulk still images with manifest reuse, existence checks, and Pillow spot check."""
    bulk_dir = repo_root / "data" / "squirrel_sources" / "inaturalist_bulk"
    manifest_path = repo_root / "research" / "squirrel_scale" / "metadata" / "images" / "manifest.jsonl"
    obs_path = repo_root / "research" / "squirrel_scale" / "metadata" / "images" / "observations.jsonl"

    if not bulk_dir.is_dir():
        raise FileNotFoundError(f"inaturalist_bulk directory not found: {bulk_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Bulk manifest not found: {manifest_path}")
    if not obs_path.is_file():
        raise FileNotFoundError(f"Bulk observations metadata not found: {obs_path}")

    # Load authoritative bulk observations for validation
    obs_classes: dict[tuple[int, int], str] = {}
    with obs_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line_str = line.strip()
            if not line_str:
                continue
            o = json.loads(line_str)
            obs_id = o.get("observation_id")
            obs_cls = o.get("class")
            for p in o.get("photos", []):
                p_id = p.get("photo_id")
                if obs_id is not None and p_id is not None and obs_cls:
                    obs_classes[(int(obs_id), int(p_id))] = obs_cls

    # Load download manifest
    manifest_entries: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line_str = line.strip()
            if line_str:
                manifest_entries.append(json.loads(line_str))

    report["inaturalist_bulk"] = {
        "manifest_total_entries": len(manifest_entries),
        "skipped_pilot_overlap": 0,
        "skipped_gif_animations": 0,
        "missing_on_disk": [],
        "valid_still_images": 0,
        "unmanifested_disk_images": [],
        "spot_check_count": 0,
        "spot_check_failures": [],
    }

    # Inventory disk files
    disk_still_files = {
        p.name: p
        for p in bulk_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and p.suffix.lower() != ".gif"
    }

    results: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    seen_manifest_files: set[str] = set()

    for item in manifest_entries:
        path_str = item.get("path", "")
        is_pilot = (
            item.get("source") == "existing_pilot"
            or "data/squirrel_sources/inaturalist/" in path_str
        )
        obs_id_val = int(item["observation_id"]) if item.get("observation_id") is not None else None
        photo_id_val = int(item["photo_id"]) if item.get("photo_id") is not None else None
        key = (obs_id_val, photo_id_val) if obs_id_val is not None and photo_id_val is not None else None

        if is_pilot or (key is not None and key in seen_keys):
            report["inaturalist_bulk"]["skipped_pilot_overlap"] += 1
            continue

        # Exclude animated GIF files
        if item.get("format") == "GIF" or path_str.lower().endswith(".gif"):
            report["inaturalist_bulk"]["skipped_gif_animations"] += 1
            continue

        file_path = repo_root / path_str
        seen_manifest_files.add(file_path.name)

        if not file_path.is_file():
            missing_info = {
                "path": path_str,
                "observation_id": obs_id_val,
                "photo_id": photo_id_val,
                "class": item.get("class"),
            }
            report["inaturalist_bulk"]["missing_on_disk"].append(missing_info)
            continue

        if key is not None:
            seen_keys.add(key)

        candidate_rows.append(item)

    report["inaturalist_bulk"]["valid_still_images"] = len(candidate_rows)

    # Check for unmanifested still images on disk
    unmanifested = set(disk_still_files.keys()) - seen_manifest_files
    if unmanifested:
        report["inaturalist_bulk"]["unmanifested_disk_images"] = sorted(unmanifested)

    # Spot-check a deterministic small sample with Pillow
    if candidate_rows and spot_check_sample_size > 0:
        step = max(1, len(candidate_rows) // spot_check_sample_size)
        sample = candidate_rows[::step][:spot_check_sample_size]
        report["inaturalist_bulk"]["spot_check_count"] = len(sample)

        for s_item in sample:
            s_path = repo_root / s_item["path"]
            w, h, fmt, err = inspect_image(s_path)
            if err:
                report["inaturalist_bulk"]["spot_check_failures"].append({
                    "path": s_item["path"],
                    "error": err,
                })
            elif (w, h) != (s_item.get("width"), s_item.get("height")):
                report["inaturalist_bulk"]["spot_check_failures"].append({
                    "path": s_item["path"],
                    "error": f"Dimension mismatch: manifest={(s_item.get('width'), s_item.get('height'))} != Pillow={(w, h)}",
                })

    for item in candidate_rows:
        obs_id_val = int(item["observation_id"]) if item.get("observation_id") is not None else None
        photo_id_val = int(item["photo_id"]) if item.get("photo_id") is not None else None
        key = (obs_id_val, photo_id_val) if obs_id_val is not None and photo_id_val is not None else None

        raw_class = item.get("class")
        raw_scientific = item.get("scientific_name")
        species_label = normalize_class_label(raw_class, raw_scientific)

        status_flags = ["image_level_label_provisional_for_boxes", "inat_bulk_metadata_verified"]
        metadata_error: str | None = None

        if key is not None and key in obs_classes:
            obs_cls = obs_classes[key]
            if raw_class and raw_class != obs_cls:
                status_flags.append("class_metadata_mismatch")
                metadata_error = f"Manifest class {raw_class} != observations.jsonl class {obs_cls}"
        elif key is not None:
            status_flags.append("unmatched_in_bulk_observations")

        path_obj = Path(item["path"])
        repo_rel_path = path_obj.as_posix()
        source_rel_path = path_obj.name

        row: dict[str, Any] = {
            "image_id": f"inaturalist_bulk:{source_rel_path}",
            "image_path": repo_rel_path,
            "source": "inaturalist_bulk",
            "species_label": species_label,
            "scientific_name": raw_scientific,
            "label_basis": "inat_metadata",
            "species_label_provisional": True,
            "observation_id": obs_id_val,
            "photo_id": photo_id_val,
            "source_url": item.get("observation_url"),
            "photo_url": item.get("original_url"),
            "license": item.get("photo_license") or item.get("observation_license"),
            "attribution": item.get("attribution"),
            "group_id": str(obs_id_val) if obs_id_val is not None else f"inat_bulk_{source_rel_path}",
            "image_sha256": item.get("sha256"),
            "width": item.get("width"),
            "height": item.get("height"),
            "image_format": item.get("format"),
            "status_flags": sorted(status_flags),
            "image_error": None,
            "metadata_error": metadata_error,
        }
        results.append(row)

    return results


def process_harrybaines(
    repo_root: Path,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Process Harry Baines images, assigning Sciurus carolinensis per user confirmed basis."""
    source_root = repo_root / "data" / "squirrel_sources" / "harrybaines_squirrels"
    raw_dir = source_root / "raw"
    acq_path = source_root / "acquisition.json"

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Harry Baines raw directory not found: {raw_dir}")
    if not acq_path.is_file():
        raise FileNotFoundError(f"Harry Baines acquisition metadata not found: {acq_path}")

    with acq_path.open("r", encoding="utf-8") as stream:
        acq_data = json.load(stream)

    source_url = acq_data.get("source_url", "https://www.kaggle.com/datasets/harrybaines/squirrels")
    license_str = acq_data.get("provider_license", "CC0: Public Domain")
    attribution_str = "Harry Baines"

    image_files = sorted(
        [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name,
    )

    burst_map = cluster_burst_groups(image_files)

    report["harrybaines"] = {
        "total_images": len(image_files),
        "burst_groups": len(set(burst_map.values())),
        "corrupt_images": 0,
    }

    results: list[dict[str, Any]] = []
    for f in image_files:
        width, height, img_fmt, img_err = inspect_image(f)
        status_flags = ["image_level_label_provisional_for_boxes", "user_confirmed_species"]
        image_error = None
        if img_err:
            report["harrybaines"]["corrupt_images"] += 1
            status_flags.append("image_read_error")
            image_error = img_err
        sha256_val = compute_sha256(f)

        source_rel_path = f.relative_to(source_root).as_posix()
        repo_rel_path = f.relative_to(repo_root).as_posix()
        group_id = burst_map[f]

        row: dict[str, Any] = {
            "image_id": f"harrybaines:{source_rel_path}",
            "image_path": repo_rel_path,
            "source": "harrybaines",
            "species_label": "Sciurus carolinensis",
            "scientific_name": "Sciurus carolinensis",
            "label_basis": "user_confirmed",
            "species_label_provisional": True,
            "observation_id": None,
            "photo_id": None,
            "source_url": source_url,
            "photo_url": None,
            "license": license_str,
            "attribution": attribution_str,
            "group_id": group_id,
            "image_sha256": sha256_val,
            "width": width,
            "height": height,
            "image_format": img_fmt,
            "status_flags": sorted(status_flags),
            "image_error": image_error,
            "metadata_error": None,
        }
        results.append(row)

    return results


def process_meyer_trailcam(
    repo_root: Path,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Process Meyer Trailcam images, strictly including DatasetSamples/Squirrel only."""
    source_root = repo_root / "data" / "squirrel_sources" / "meyer_trailcam"
    dataset_samples_dir = source_root / "raw" / "DatasetSamples"
    squirrel_dir = dataset_samples_dir / "Squirrel"
    acq_path = source_root / "acquisition.json"

    if not squirrel_dir.is_dir():
        raise FileNotFoundError(f"Meyer Trailcam squirrel directory not found: {squirrel_dir}")
    if not acq_path.is_file():
        raise FileNotFoundError(f"Meyer Trailcam acquisition metadata not found: {acq_path}")

    with acq_path.open("r", encoding="utf-8") as stream:
        acq_data = json.load(stream)

    source_url = acq_data.get(
        "source_url",
        "https://www.kaggle.com/datasets/jimmeyer645/trailcam-dataset-samples-from-larger-dataset",
    )
    license_str = acq_data.get("provider_license", "CDLA-Permissive-1.0")
    attribution_str = "Jim Meyer"

    all_files = sorted([p for p in squirrel_dir.rglob("*") if p.is_file()])
    valid_original_images: list[Path] = []
    skipped_rendered: list[Path] = []
    skipped_annotations: list[Path] = []
    skipped_other: list[Path] = []

    for f in all_files:
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            if "-rendered" in f.name:
                skipped_rendered.append(f)
            else:
                valid_original_images.append(f)
        elif f.suffix.lower() == ".xml":
            skipped_annotations.append(f)
        else:
            skipped_other.append(f)

    other_animal_folders = [
        d.name for d in dataset_samples_dir.iterdir()
        if d.is_dir() and d.name != "Squirrel"
    ]

    report["meyer_trailcam"] = {
        "valid_original_images": len(valid_original_images),
        "skipped_rendered_derivatives": len(skipped_rendered),
        "skipped_xml_annotations": len(skipped_annotations),
        "skipped_non_media_files": len(skipped_other),
        "excluded_other_animal_folders": sorted(other_animal_folders),
        "corrupt_images": 0,
    }

    results: list[dict[str, Any]] = []
    for f in valid_original_images:
        width, height, img_fmt, img_err = inspect_image(f)
        status_flags = ["image_level_label_provisional_for_boxes", "generic_squirrel_source"]
        image_error = None
        if img_err:
            report["meyer_trailcam"]["corrupt_images"] += 1
            status_flags.append("image_read_error")
            image_error = img_err
        sha256_val = compute_sha256(f)

        source_rel_path = f.relative_to(source_root).as_posix()
        repo_rel_path = f.relative_to(repo_root).as_posix()

        rel_to_squirrel = f.relative_to(squirrel_dir)
        parent_parts = rel_to_squirrel.parent.parts
        if parent_parts:
            group_id = f"meyer_trailcam_{'_'.join(parent_parts)}"
        else:
            group_id = "meyer_trailcam_root"

        row: dict[str, Any] = {
            "image_id": f"meyer_trailcam:{source_rel_path}",
            "image_path": repo_rel_path,
            "source": "meyer_trailcam",
            "species_label": "generic_squirrel",
            "scientific_name": "generic_squirrel",
            "label_basis": "generic_source",
            "species_label_provisional": True,
            "observation_id": None,
            "photo_id": None,
            "source_url": source_url,
            "photo_url": None,
            "license": license_str,
            "attribution": attribution_str,
            "group_id": group_id,
            "image_sha256": sha256_val,
            "width": width,
            "height": height,
            "image_format": img_fmt,
            "status_flags": sorted(status_flags),
            "image_error": image_error,
            "metadata_error": None,
        }
        results.append(row)

    return results


def build_manifest(
    repo_root: Path,
    output_path: Path | None = None,
    spot_check_sample_size: int = 50,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic manifest and verification report across all squirrel sources."""
    repo_root = repo_root.resolve()
    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "sources": {},
    }

    seen_keys: set[tuple[int, int]] = set()

    inat_rows = process_inaturalist(repo_root, report, seen_keys)
    bulk_rows = process_inaturalist_bulk(repo_root, report, seen_keys, spot_check_sample_size)
    hb_rows = process_harrybaines(repo_root, report)
    meyer_rows = process_meyer_trailcam(repo_root, report)

    all_rows = inat_rows + bulk_rows + hb_rows + meyer_rows

    # Deterministic sorting
    all_rows.sort(key=lambda r: (r["source"], r["image_path"]))

    # Validation: uniqueness of image_id and image_path
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for row in all_rows:
        img_id = row["image_id"]
        img_path = row["image_path"]
        if img_id in seen_ids:
            raise ValueError(f"Duplicate image_id detected: {img_id}")
        if img_path in seen_paths:
            raise ValueError(f"Duplicate image_path detected: {img_path}")
        seen_ids.add(img_id)
        seen_paths.add(img_path)

    # Per-source and per-species statistics
    per_source_counts: dict[str, int] = {}
    per_species_counts: dict[str, int] = {}
    for row in all_rows:
        src = row["source"]
        per_source_counts[src] = per_source_counts.get(src, 0) + 1
        spec = row["species_label"] or "unassigned"
        per_species_counts[spec] = per_species_counts.get(spec, 0) + 1

    report["totals"] = {
        "total_images": len(all_rows),
        "per_source": per_source_counts,
        "per_species": per_species_counts,
        "unique_group_ids": len(set(r["group_id"] for r in all_rows)),
    }

    if output_path is not None:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            for row in all_rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        report["output_manifest"] = str(output_path)

    return all_rows, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to repository root (defaults to parent of scripts/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for output JSONL manifest (defaults to output/v2_annotation/source_manifest.jsonl)",
    )
    parser.add_argument(
        "--spot-check-size",
        type=int,
        default=50,
        help="Deterministic sample size for bulk Pillow verification",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    output_path = args.output
    if output_path is None:
        output_path = repo_root / "output" / "v2_annotation" / "source_manifest.jsonl"
    else:
        output_path = output_path.resolve()

    try:
        rows, report = build_manifest(repo_root, output_path, spot_check_sample_size=args.spot_check_size)
    except Exception as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1

    print(f"Manifest written successfully to: {output_path}")
    print(f"Total images: {report['totals']['total_images']}")
    print("Per-source counts:")
    for src, count in sorted(report["totals"]["per_source"].items()):
        print(f"  - {src}: {count}")
    print("Per-species counts (normalized):")
    for spec, count in sorted(report["totals"]["per_species"].items()):
        print(f"  - {spec}: {count}")
    print(f"Unique splitting groups: {report['totals']['unique_group_ids']}")

    # Report exclusions and missing items
    bulk_rep = report.get("inaturalist_bulk", {})
    if bulk_rep:
        print(f"Bulk skipped pilot overlap: {bulk_rep.get('skipped_pilot_overlap')}")
        print(f"Bulk skipped GIF animations: {bulk_rep.get('skipped_gif_animations')}")
        missing_list = bulk_rep.get("missing_on_disk", [])
        print(f"Bulk missing paths on disk: {len(missing_list)}")
        for m in missing_list:
            print(f"  * {m['path']} (obs_id={m['observation_id']}, photo_id={m['photo_id']}, class={m['class']})")
        print(f"Bulk spot check: {bulk_rep.get('spot_check_count')} verified, {len(bulk_rep.get('spot_check_failures', []))} failures")

    return 0


if __name__ == "__main__":
    sys.exit(main())
