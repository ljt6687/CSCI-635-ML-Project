"""Immutable audited COCO input and a fingerprinted YOLO label export."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

SPLITS = ("train", "valid", "test")
ULTRALYTICS_VERSION = "8.4.164"
YOLOV5_COMMIT = "402e17ddf820996f51a191cbb798376e1144f069"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def dataset_identity(manifest: dict) -> dict:
    return {k: manifest[k] for k in (
        "source_fingerprint", "source_image_fingerprint", "audit_decisions_sha256",
        "categories", "annotation_hashes", "split_counts", "annotation_counts",
    )}


def dataset_fingerprint(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(dataset_identity(manifest), sort_keys=True).encode()).hexdigest()


def image_path(data: Path, split: str, image: dict) -> Path:
    parent = (data / split).resolve()
    path = (parent / image["file_name"]).resolve()
    if not path.is_relative_to(parent) or not path.is_file():
        raise ValueError(f"Missing or unsafe image path: {split}/{image['file_name']}")
    return path


def verify_dataset(data: str | Path) -> tuple[dict, dict]:
    data = Path(data)
    manifest = json.loads((data / ".complete.json").read_text())
    if manifest.get("version") != 3 or manifest.get("qa_status") != "passed":
        raise ValueError("A passing version-3 audited export is required")
    if manifest.get("num_classes") != 11 or len(manifest.get("categories", [])) != 11:
        raise ValueError("Expected the audited 11-class v2 dataset")
    for key in ("source_fingerprint", "source_image_fingerprint", "audit_decisions_sha256"):
        if not manifest.get(key):
            raise ValueError(f"Missing audit provenance: {key}")
    classes = {i + 1: name for i, name in enumerate(manifest["categories"])}
    docs = {}
    for split in SPLITS:
        ann = data / split / "_annotations.coco.json"
        if sha256(ann) != manifest["annotation_hashes"][split]:
            raise ValueError(f"Stale {split} annotation fingerprint")
        doc = json.loads(ann.read_text())
        if {c["id"]: c["name"] for c in doc["categories"]} != classes:
            raise ValueError(f"{split}: category mapping differs from the audited export")
        if len(doc["images"]) != manifest["split_counts"][split] or len(doc["annotations"]) != manifest["annotation_counts"][split]:
            raise ValueError(f"{split}: manifest counts differ")
        images = {im["id"]: im for im in doc["images"]}
        if len(images) != len(doc["images"]) or not images:
            raise ValueError(f"{split}: missing or duplicate image IDs")
        if len({a["id"] for a in doc["annotations"]}) != len(doc["annotations"]):
            raise ValueError(f"{split}: duplicate annotation IDs")
        stems = set()
        for im in images.values():
            image_path(data, split, im)
            stem = Path(im["file_name"]).stem
            if stem in stems or im["width"] <= 0 or im["height"] <= 0:
                raise ValueError(f"{split}: colliding filenames or invalid dimensions")
            stems.add(stem)
        for a in doc["annotations"]:
            if a["image_id"] not in images or a["category_id"] not in classes or a.get("iscrowd", 0):
                raise ValueError(f"{split}: invalid annotation references or unsupported crowd box")
            coco_to_yolo(a["bbox"], images[a["image_id"]]["width"], images[a["image_id"]]["height"])
        if {a["category_id"] for a in doc["annotations"]} != set(classes):
            raise ValueError(f"{split}: all eleven classes must be represented")
        docs[split] = doc
    return manifest, docs


def coco_to_yolo(box, width: int, height: int) -> list[float]:
    x, y, w, h = map(float, box)
    if width <= 0 or height <= 0 or not all(math.isfinite(v) for v in (x, y, w, h)):
        raise ValueError("Nonfinite box or invalid image dimensions")
    if w <= 0 or h <= 0 or min(x, y) < 0 or x + w > width + 1e-6 or y + h > height + 1e-6:
        raise ValueError("Box lies outside its image or has nonpositive area")
    return [(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]


def export_yolo(data: str | Path, destination: str | Path | None = None) -> Path:
    """Build once; subsequent calls validate image and label hashes, never rewrite."""
    from PIL import Image
    import yaml

    data = Path(data).resolve()
    manifest, docs = verify_dataset(data)
    fingerprint = dataset_fingerprint(manifest)
    destination = Path(destination or data.parent / "squirrel-v2-audited-yolo" / fingerprint[:16]).resolve()
    if destination == data or destination.is_relative_to(data):
        raise ValueError("YOLO export must be outside the immutable COCO source")
    marker = destination / "export_manifest.json"
    if destination.exists():
        if not marker.is_file():
            raise ValueError("Incomplete export exists; choose another destination or remove it after inspection")
        saved = json.loads(marker.read_text())
        if saved["dataset_fingerprint"] != fingerprint:
            raise ValueError("Existing YOLO export belongs to different data")
        for row in saved["images"]:
            if (sha256(destination / row["image"]) != row["sha256"]
                    or sha256(data / row["split"] / row["source_file"]) != row["sha256"]
                    or sha256(destination / row["label"]) != row["label_sha256"]):
                raise ValueError(f"YOLO export changed: {row['image']}")
        if sha256(destination / "data.yaml") != saved["yaml_sha256"]:
            raise ValueError("YOLO dataset YAML changed")
        return destination / "data.yaml"

    temporary = destination.with_name(destination.name + f".building-{os.getpid()}")
    if temporary.exists():
        raise ValueError(f"Incomplete conversion at {temporary}")
    temporary.mkdir(parents=True)
    rows, seen = [], {}
    try:
        for split, doc in docs.items():
            annotations = {}
            for a in doc["annotations"]:
                annotations.setdefault(a["image_id"], []).append(a)
            for folder in ("images", "labels"):
                (temporary / folder / split).mkdir(parents=True)
            for im in doc["images"]:
                source = image_path(data, split, im)
                with Image.open(source) as opened:
                    if opened.size != (im["width"], im["height"]) or opened.mode != "RGB":
                        raise ValueError(f"Unexpected image dimensions/mode: {source.name}")
                digest = sha256(source)
                if digest in seen and seen[digest] != split:
                    raise ValueError("Confirmed exact cross-split duplicate; training blocked")
                seen[digest] = split
                relative_image = Path("images") / split / source.name
                relative_label = Path("labels") / split / (source.stem + ".txt")
                # Copy bytes rather than link: native loaders/caches cannot modify the source.
                shutil.copyfile(source, temporary / relative_image)
                lines = []
                for a in sorted(annotations.get(im["id"], []), key=lambda a: a["id"]):
                    values = coco_to_yolo(a["bbox"], im["width"], im["height"])
                    lines.append(f"{a['category_id'] - 1} " + " ".join(f"{v:.12f}" for v in values))
                (temporary / relative_label).write_text("\n".join(lines) + ("\n" if lines else ""))
                rows.append(dict(split=split, image_id=im["id"], source_file=im["file_name"],
                                 image=str(relative_image), label=str(relative_label), sha256=digest,
                                 label_sha256=sha256(temporary / relative_label)))
        settings = dict(path=str(destination), train="images/train", val="images/valid", test="images/test",
                        nc=11, names={i: name for i, name in enumerate(manifest["categories"])})
        (temporary / "data.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
        save_json(temporary / "export_manifest.json", dict(
            schema_version=1, source=str(data), dataset_fingerprint=fingerprint,
            dataset_manifest=manifest, images=rows, yaml_sha256=sha256(temporary / "data.yaml")))
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return destination / "data.yaml"


def records_for_split(data: str | Path, split: str, doc: dict) -> list[dict]:
    names = {c["id"]: c["name"] for c in doc["categories"]}
    boxes = {}
    for a in doc["annotations"]:
        x, y, w, h = a["bbox"]
        boxes.setdefault(a["image_id"], []).append(dict(
            box=[x, y, x + w, y + h], category_id=a["category_id"], class_name=names[a["category_id"]]))
    return [dict(image_id=im["id"], path=str(image_path(Path(data), split, im)),
                 gt=boxes.get(im["id"], [])) for im in doc["images"]]
