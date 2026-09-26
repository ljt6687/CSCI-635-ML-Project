import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.audit_squirrel_v2_purity import audit, components, find_pairs, source_fingerprint, source_image_fingerprint, visual_pair_evidence


class PurityAuditTests(unittest.TestCase):
    def test_full_audit_refuses_to_overwrite_reviewed_decisions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); out = root / "audit"; out.mkdir()
            saved = '{"qa_passed": false, "exclude": [{"file_name": "reviewed.jpg"}]}'
            (out / "decisions.json").write_text(saved)
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite reviewed decisions"):
                audit(root / "missing_source", root / "missing_manifest", out)
            self.assertEqual((out / "decisions.json").read_text(), saved)

    def test_fingerprint_is_concatenated_coco_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = b""
            for split, data in (("train", b"a"), ("valid", b"b"), ("test", b"c")):
                folder = root / split
                folder.mkdir()
                (folder / "_annotations.coco.json").write_bytes(data)
                expected += data
            self.assertEqual(source_fingerprint(root), hashlib.sha256(expected).hexdigest())

    def test_image_fingerprint_excludes_only_named_image(self):
        rows = [
            {"split": "valid", "file_name": "b.jpg", "pixel_hash": "bb"},
            {"split": "train", "file_name": "a.jpg", "pixel_hash": "aa"},
        ]
        expected = hashlib.sha256(b"train/a.jpg|aa\nvalid/b.jpg|bb\n").hexdigest()
        self.assertEqual(source_image_fingerprint(rows), expected)
        excluded = [{"split": "valid", "file_name": "b.jpg", "reason": "broken"}]
        self.assertEqual(source_image_fingerprint(rows, excluded), hashlib.sha256(b"train/a.jpg|aa\n").hexdigest())

    def test_pairs_capture_exact_pixels_beyond_phash_limit(self):
        rows = [
            dict(split="train", file_name="a.jpg", phash="0000000000000000", pixel_hash="same", group_id="", species_label="A"),
            dict(split="valid", file_name="b.jpg", phash="ffffffffffffffff", pixel_hash="same", group_id="", species_label="A"),
            dict(split="train", file_name="c.jpg", phash="0000000000000001", pixel_hash="other", group_id="", species_label="B"),
        ]
        pairs = find_pairs(rows)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["same_pixels"])
        self.assertEqual(pairs[0]["distance"], 64)
        self.assertEqual(components(pairs), [[{"split": "train", "file_name": "a.jpg"}, {"split": "valid", "file_name": "b.jpg"}]])

    def test_visual_pair_evidence_confirms_identical_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ("train", "test"):
                (root / split).mkdir()
                Image.new("RGB", (32, 32), "brown").save(root / split / "same.png")
            pairs = [{"pair_id": "p", "left": {"split": "train", "file_name": "same.png"},
                      "right": {"split": "test", "file_name": "same.png"},
                      "same_pixels": True, "distance": 0, "cross_class": False}]
            result = visual_pair_evidence(root, pairs)
            self.assertTrue(result[0]["visual_match"])
            self.assertEqual(result[0]["evidence"], "identical decoded pixels")

    def test_exif_orientation_keeps_raw_coco_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / "data"
            for split in ("train", "valid", "test"):
                folder = data / split; folder.mkdir(parents=True)
                image = Image.new("RGB", (20, 30), "brown")
                exif = image.getexif(); exif[274] = 6
                image.save(folder / "oriented.jpg", exif=exif)
                doc = {"images": [{"id": 1, "file_name": "oriented.jpg", "width": 20, "height": 30}],
                       "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 20]}],
                       "categories": [{"id": 1, "name": "A"}]}
                (folder / "_annotations.coco.json").write_text(json.dumps(doc))
            out = root / "audit"
            audit(data, root / "missing_manifest.jsonl", out)
            problems = json.loads((out / "integrity_problems.json").read_text())
            self.assertFalse(any(p["reason"] == "dimension mismatch" for p in problems))

    def test_audit_blocks_missing_media_and_writes_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            for split in ("train", "valid", "test"):
                folder = data / split
                folder.mkdir(parents=True)
                Image.new("RGB", (32, 32), "brown").save(folder / "present.png")
                images = [{"id": 1, "file_name": "present.png", "width": 32, "height": 32}]
                if split == "valid":
                    images.append({"id": 2, "file_name": "missing.png", "width": 32, "height": 32})
                doc = {"images": images, "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10]}], "categories": [{"id": 1, "name": "A"}]}
                (folder / "_annotations.coco.json").write_text(json.dumps(doc))
            out = root / "audit"
            summary = audit(data, root / "no_metadata.jsonl", out)
            decisions = json.loads((out / "decisions.json").read_text())
            self.assertGreaterEqual(summary["integrity_problems"], 1)
            self.assertFalse(decisions["qa_passed"])
            self.assertEqual(decisions["source_fingerprint"], source_fingerprint(data))
            self.assertEqual(len(decisions["confirmed_duplicate_components"]), 1)
            self.assertTrue(any(p["reason"] == "missing image" for p in json.loads((out / "integrity_problems.json").read_text())))


if __name__ == "__main__":
    unittest.main()
