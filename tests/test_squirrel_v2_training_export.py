"""Tests for Squirrel Dataset v2 multiclass training export and leak-free splits."""

import json
from pathlib import Path
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data/squirrel-v2-clean"
SUMMARY_FILE = ROOT / "output/v2_annotation/training_export_summary.json"
MANIFEST_MARKER = DATASET_DIR / ".complete.json"

EXPECTED_CATEGORIES = [
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
    "generic_squirrel",
]


class TestSquirrelV2TrainingExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = {}
        for split_name in ("train", "valid", "test"):
            ann_path = DATASET_DIR / split_name / "_annotations.coco.json"
            if not ann_path.is_file():
                raise FileNotFoundError(f"Missing {ann_path}")
            cls.splits[split_name] = json.loads(ann_path.read_text(encoding="utf-8"))

    def test_directory_structure_and_files(self):
        self.assertTrue(DATASET_DIR.is_dir(), "Dataset directory missing")
        self.assertTrue(MANIFEST_MARKER.is_file(), ".complete.json marker missing")
        self.assertTrue(SUMMARY_FILE.is_file(), "training_export_summary.json missing")

        for split_name in ("train", "valid", "test"):
            split_dir = DATASET_DIR / split_name
            self.assertTrue(split_dir.is_dir(), f"Split directory missing: {split_dir}")
            ann_file = split_dir / "_annotations.coco.json"
            self.assertTrue(ann_file.is_file(), f"Annotation file missing: {ann_file}")

    def test_zero_file_name_overlap(self):
        train_files = {img["file_name"] for img in self.splits["train"]["images"]}
        val_files = {img["file_name"] for img in self.splits["valid"]["images"]}
        test_files = {img["file_name"] for img in self.splits["test"]["images"]}

        self.assertEqual(len(train_files & val_files), 0, "Train and Val share image files")
        self.assertEqual(len(train_files & test_files), 0, "Train and Test share image files")
        self.assertEqual(len(val_files & test_files), 0, "Val and Test share image files")
        self.assertEqual(len(train_files | val_files | test_files), 8584, "Total unique images must be 8584")

    def test_zero_group_leakage(self):
        train_groups = {img["group_id"] for img in self.splits["train"]["images"]}
        val_groups = {img["group_id"] for img in self.splits["valid"]["images"]}
        test_groups = {img["group_id"] for img in self.splits["test"]["images"]}

        self.assertEqual(len(train_groups & val_groups), 0, f"Group leakage Train/Val: {train_groups & val_groups}")
        self.assertEqual(len(train_groups & test_groups), 0, f"Group leakage Train/Test: {train_groups & test_groups}")
        self.assertEqual(len(val_groups & test_groups), 0, f"Group leakage Val/Test: {val_groups & test_groups}")

    def test_large_burst_protection(self):
        """Bursts with > 4 images must be isolated in train."""
        val_groups = {}
        for img in self.splits["valid"]["images"]:
            val_groups[img["group_id"]] = val_groups.get(img["group_id"], 0) + 1

        test_groups = {}
        for img in self.splits["test"]["images"]:
            test_groups[img["group_id"]] = test_groups.get(img["group_id"], 0) + 1

        for gid, count in val_groups.items():
            self.assertLessEqual(count, 4, f"Large burst in valid split: {gid} has {count} images")

        for gid, count in test_groups.items():
            self.assertLessEqual(count, 4, f"Large burst in test split: {gid} has {count} images")

    def test_categories_schema_and_alignment(self):
        cat_names = [c["name"] for c in self.splits["train"]["categories"]]
        self.assertEqual(cat_names, EXPECTED_CATEGORIES)

        for split_name in ("valid", "test"):
            split_cats = [c["name"] for c in self.splits[split_name]["categories"]]
            self.assertEqual(split_cats, cat_names, f"{split_name} categories do not match train")

    def test_split_proportions_and_per_class_representation(self):
        marker = json.loads(MANIFEST_MARKER.read_text(encoding="utf-8"))
        counts = marker["split_counts"]
        total = sum(counts.values())

        self.assertEqual(total, 8584)
        self.assertEqual(counts["train"], 6006)
        self.assertEqual(counts["valid"], 1288)
        self.assertEqual(counts["test"], 1290)

        # Total split percentages: 70.0%, 15.0%, 15.0%
        self.assertAlmostEqual(counts["train"] / total, 0.70, places=2)
        self.assertAlmostEqual(counts["valid"] / total, 0.15, places=2)
        self.assertAlmostEqual(counts["test"] / total, 0.15, places=2)

        # Every class represented in all 3 splits with ~70:15:15
        dist = marker["species_distribution"]
        for sp in EXPECTED_CATEGORIES:
            row = dist[sp]
            self.assertGreater(row["train"], 0, f"No train images for {sp}")
            self.assertGreater(row["valid"], 0, f"No valid images for {sp}")
            self.assertGreater(row["test"], 0, f"No test images for {sp}")
            sp_tot = row["total"]
            self.assertTrue(0.68 <= row["train"] / sp_tot <= 0.72, f"{sp} train ratio out of range")
            self.assertTrue(0.13 <= row["valid"] / sp_tot <= 0.17, f"{sp} valid ratio out of range")
            self.assertTrue(0.13 <= row["test"] / sp_tot <= 0.17, f"{sp} test ratio out of range")

    def test_no_empty_images_and_box_validity(self):
        for split_name in ("train", "valid", "test"):
            coco = self.splits[split_name]
            split_dir = DATASET_DIR / split_name
            images_by_id = {img["id"]: img for img in coco["images"]}
            ann_counts = {}

            for ann in coco["annotations"]:
                iid = ann["image_id"]
                ann_counts[iid] = ann_counts.get(iid, 0) + 1
                img = images_by_id[iid]
                iw, ih = img["width"], img["height"]
                x, y, w, h = ann["bbox"]

                self.assertGreaterEqual(x, 0.0)
                self.assertGreaterEqual(y, 0.0)
                self.assertGreater(w, 0.0)
                self.assertGreater(h, 0.0)
                self.assertLessEqual(round(x + w, 2), iw)
                self.assertLessEqual(round(y + h, 2), ih)
                self.assertAlmostEqual(ann["area"], round(w * h, 2), places=1)
                self.assertIn(ann["category_id"], range(1, 12))

            # Every image in the split must have at least one annotation
            for img in coco["images"]:
                self.assertGreater(ann_counts.get(img["id"], 0), 0, f"Empty image in {split_name}: {img['file_name']}")
                # Check file exists and PIL opens it
                img_path = split_dir / img["file_name"]
                self.assertTrue(img_path.is_file(), f"Missing linked image file: {img_path}")


if __name__ == "__main__":
    unittest.main()
