"""Focused tests for the immutable, audited COCO exporter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from PIL import Image

from scripts.export_squirrel_v2_audited_coco import (
    Entry,
    assign_splits,
    export_dataset,
    risk_components,
    source_fingerprint,
    transformed_box,
)


class GeometryTests(unittest.TestCase):
    def test_exif_rotation_six_transforms_bbox(self):
        # 100x60 rotated 90 degrees clockwise becomes 60x100.
        self.assertEqual(transformed_box([10, 20, 30, 20], 100, 60, 6, 60, 100),
                         [20.0, 10.0, 20.0, 30.0])

    def test_scaling_and_invalid_geometry(self):
        self.assertEqual(transformed_box([100, 50, 200, 100], 1000, 500, 1, 500, 250),
                         [50.0, 25.0, 100.0, 50.0])
        self.assertIsNone(transformed_box([990, 50, 20, 100], 1000, 500, 1, 500, 250))


class GroupingTests(unittest.TestCase):
    def entry(self, split: str, name: str, phash: int, group: str = "") -> Entry:
        return Entry(split, name, {"id": 1}, 1, [{"category_id": 1}], group or name,
                     Path(name), 100, 100, name, name, phash, [(1, [1, 1, 10, 10])], 1)

    def test_tight_perceptual_component_is_split_isolated(self):
        entries = [self.entry("train", "a.jpg", 0),
                   self.entry("test", "b.jpg", 1),
                   self.entry("valid", "c.jpg", 2)]
        components, edges = risk_components(entries, {"confirmed_duplicate_components": []})
        self.assertEqual(edges, 3)
        self.assertEqual(len(components), 1)
        assign_splits(entries, components)
        self.assertEqual({e.assigned_split for e in entries}, {"train"})

    def test_four_bit_similarity_across_chunks_is_grouped(self):
        # One changed bit in each of four 13-bit chunks defeats the old
        # four-chunk index, but the five-chunk index must still find it.
        base = 0
        four_bit_variant = sum(1 << bit for bit in (0, 13, 26, 39))
        entries = [self.entry("train", "a.jpg", base),
                   self.entry("test", "b.jpg", four_bit_variant)]
        components, edges = risk_components(entries, {})
        self.assertEqual(edges, 1)
        self.assertEqual(len(components), 1)
        assign_splits(entries, components)
        self.assertEqual({e.assigned_split for e in entries}, {"train"})

    def test_visually_related_scene_is_grouped_beyond_phash_threshold(self):
        entries = [self.entry("train", "a.jpg", 0),
                   self.entry("valid", "b.jpg", (1 << 64) - 1)]
        decisions = {"visually_related_components": [[
            {"split": "train", "file_name": "a.jpg"},
            {"split": "valid", "file_name": "b.jpg"},
        ]]}
        components, _ = risk_components(entries, decisions)
        self.assertEqual(len(components), 1)
        assign_splits(entries, components)
        self.assertEqual({e.assigned_split for e in entries}, {"train"})

    def test_cross_class_group_is_not_split_across_partitions(self):
        entries = [self.entry("train", "a.jpg", 0x0101010101010101, "shared"),
                   self.entry("test", "b.jpg", 0xF0F0F0F0F0F0F0F0, "shared")]
        entries[1].category_id = 2
        components, _ = risk_components(entries, {})
        assign_splits(entries, components)
        self.assertEqual({e.assigned_split for e in entries}, {"train"})
        self.assertEqual([e.category_id for e in entries], [1, 2])


class FullExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.target = self.root / "audited"
        self.decisions = self.root / "decisions.json"
        categories = [{"id": i, "name": f"species_{i}", "supercategory": "squirrel"}
                      for i in range(1, 12)]
        rng = random.Random(42)
        for split in ("train", "valid", "test"):
            folder = self.source / split
            folder.mkdir(parents=True)
            images, annotations = [], []
            for cid in range(1, 12):
                name = f"{split}_{cid}.png"
                pixels = bytes(rng.randrange(256) for _ in range(64 * 64 * 3))
                image = Image.frombytes("RGB", (64, 64), pixels)
                image.save(folder / name)
                images.append({"id": cid, "file_name": name, "width": 64, "height": 64,
                               "group_id": f"{split}_{cid}", "source_image_id": name})
                annotations.append({"id": cid, "image_id": cid, "category_id": cid,
                                    "bbox": [5, 6, 20, 18], "area": 360, "iscrowd": 0})
            if split == "test":
                # The additional record is an exact train/test leak; class 1
                # still has a distinct test image after deduplication.
                duplicate = Image.open(self.source / "train/train_1.png")
                duplicate.save(folder / "duplicate.png")
                images.append({"id": 12, "file_name": "duplicate.png", "width": 64, "height": 64,
                               "group_id": "duplicate_group", "source_image_id": "duplicate"})
                annotations.append({"id": 12, "image_id": 12, "category_id": 1,
                                    "bbox": [5, 6, 20, 18], "area": 360, "iscrowd": 0})
            document = {"categories": categories, "images": images, "annotations": annotations}
            (folder / "_annotations.coco.json").write_text(json.dumps(document))
        lines = []
        for split in ("train", "valid", "test"):
            doc = json.loads((self.source / split / "_annotations.coco.json").read_text())
            for item in doc["images"]:
                with Image.open(self.source / split / item["file_name"]) as image:
                    rgb = image.convert("RGB")
                    pixel = hashlib.sha256(f"{rgb.width}x{rgb.height}".encode() + rgb.tobytes()).hexdigest()
                lines.append(f"{split}/{item['file_name']}|{pixel}\n")
        image_fingerprint = hashlib.sha256("".join(sorted(lines)).encode()).hexdigest()
        self.decisions.write_text(json.dumps({"source_fingerprint": source_fingerprint(self.source),
                                              "source_image_fingerprint": image_fingerprint,
                                              "qa_passed": True, "exclude": [], "relabel": [],
                                              "confirmed_duplicate_components": [],
                                              "false_positive_pairs": [], "unresolved": []}))

    def test_export_dedupes_without_modifying_input(self):
        before = source_fingerprint(self.source)
        summary = export_dataset(self.source, self.decisions, self.target)
        self.assertEqual(before, source_fingerprint(self.source))
        self.assertEqual(summary["output_images"], 33)
        self.assertEqual(summary["output_annotations"], 33)
        marker = json.loads((self.target / ".complete.json").read_text())
        self.assertEqual(marker["qa_status"], "passed")
        self.assertEqual(marker["source_fingerprint"], before)
        for split in ("train", "valid", "test"):
            doc = json.loads((self.target / split / "_annotations.coco.json").read_text())
            self.assertEqual(len(doc["images"]), 11)
            self.assertEqual(len(doc["annotations"]), 11)
            self.assertEqual({a["category_id"] for a in doc["annotations"]}, set(range(1, 12)))
            self.assertTrue(all((self.target / split / i["file_name"]).is_file() for i in doc["images"]))
            self.assertTrue(all(i["risk_group_id"] for i in doc["images"]))

    def test_excludes_only_bad_annotation_in_multi_box_image(self):
        ann_file = self.source / "train" / "_annotations.coco.json"
        doc = json.loads(ann_file.read_text())
        doc["annotations"].append({"id": 99, "image_id": 1, "category_id": 1,
                                   "bbox": [35, 35, 10, 10], "area": 100, "iscrowd": 0})
        ann_file.write_text(json.dumps(doc))
        decisions = json.loads(self.decisions.read_text())
        decisions["source_fingerprint"] = source_fingerprint(self.source)
        decisions["exclude_annotations"] = [{"split": "train", "file_name": "train_1.png",
                                             "annotation_id": 99, "reason": "visually empty box",
                                             "evidence": "manual full-frame inspection"}]
        self.decisions.write_text(json.dumps(decisions))
        summary = export_dataset(self.source, self.decisions, self.target)
        self.assertEqual(summary["output_images"], 33)
        self.assertEqual(summary["output_annotations"], 33)
        removed = json.loads((self.target / "decisions_applied.json").read_text())["removed"]
        self.assertIn({"key": "train/train_1.png#annotation:99", "reason": "visually empty box"}, removed)

    def test_rejects_stale_or_failed_audit(self):
        doc = json.loads(self.decisions.read_text())
        doc["source_fingerprint"] = "stale"
        self.decisions.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            export_dataset(self.source, self.decisions, self.target)
        doc["source_fingerprint"] = source_fingerprint(self.source)
        doc["qa_passed"] = False
        self.decisions.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "Audit gate"):
            export_dataset(self.source, self.decisions, self.target)


if __name__ == "__main__":
    unittest.main()
