"""Focused contract tests for provisional squirrel inference."""

import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image

from scripts import run_squirrel_v2_inference as inference


class FakePrediction:
    def __init__(self, boxes, scores, classes=None):
        self.xyxy = boxes
        self.confidence = scores
        self.class_id = classes if classes is not None else [0] * len(scores)


class FakeModel:
    def __init__(self, predictions):
        self.predictions = iter(predictions)
        self.thresholds = []

    def predict(self, image, *, threshold, include_source_image):
        self.thresholds.append((threshold, include_source_image, image.size))
        return next(self.predictions)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "output" / "model_weights").mkdir(parents=True)
        (self.root / "images").mkdir()
        self.weights = self.root / "output" / "model_weights" / "best.pth"
        self.weights.write_bytes(b"fake checkpoint")
        self.sha = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        (self.root / "output" / "model_weights" / "model_metadata.json").write_text(json.dumps({
            "model": "RFDETRMedium", "class_mapping": {"0": "SQUIRREL"},
            "resolution": 576, "sha256": self.sha,
        }))
        Image.new("RGB", (100, 80), "white").save(self.root / "images" / "sample.png")
        self.manifest = self.root / "manifest.jsonl"
        self.output = self.root / "output" / "v2_annotation" / "results.jsonl"

    def rows(self, count=1):
        rows = [{"image_id": f"img-{i}", "image_path": "images/sample.png",
                 "source": "local", "species_label": "Sciurus carolinensis",
                 "label_basis": "image_level", "license": "CC0"} for i in range(count)]
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return rows

    def results(self):
        return [json.loads(line) for line in self.output.read_text().splitlines()]

    def test_exact_bucket_boundaries(self):
        expected = [(None, "recheck_needed"), (0.20, "recheck_needed"),
                    (0.21, "recheck_needed"), (0.50, "probably_passed"),
                    (0.80, "probably_passed"), (0.81, "passed")]
        for score, bucket in expected:
            with self.subTest(score=score):
                self.assertEqual(inference.bucket_for_score(score)[0], bucket)
        self.assertEqual(inference.bucket_for_score(None, "missing")[1], "error")

    def test_result_preserves_provenance_and_all_valid_candidates(self):
        self.rows()
        model = FakeModel([FakePrediction([[10, 10, 30, 30], [0, 5, 4, 20]], [0.81, 0.20])])
        summary = inference.run(self.manifest, self.output, self.root, self.weights, "cpu",
                                model_factory=lambda *_: model)
        row = self.results()[0]
        self.assertEqual(summary["buckets"], {"passed": 1})
        self.assertEqual(row["source"], "local")
        self.assertEqual(row["license"], "CC0")
        self.assertEqual(row["image_size"], {"width": 100, "height": 80})
        self.assertEqual([box["score"] for box in row["detections"]], [0.81, 0.20])
        self.assertEqual(row["top_score"], 0.81)
        self.assertEqual(row["model_sha256"], self.sha)
        self.assertTrue(all(box["species_label_provisional"] for box in row["detections"]))
        self.assertIn("multiple_detections", row["review_flags"])
        self.assertIn("possible_poor_localization", row["review_flags"])
        self.assertEqual(model.thresholds, [(0.05, False, (100, 80))])
        self.assertIn("single RF-DETR detection score", row["model_settings"]["score_semantics"])

    def test_no_detection_and_missing_image_are_reviewed(self):
        rows = self.rows(2)
        rows[1]["image_path"] = "images/missing.png"
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        model = FakeModel([FakePrediction([], [])])
        inference.run(self.manifest, self.output, self.root, self.weights, "cpu",
                      model_factory=lambda *_: model)
        empty, missing = self.results()
        self.assertEqual(empty["bucket_reason"], "no_detection_above_floor")
        self.assertIsNone(empty["top_score"])
        self.assertIn("no_detection_review", empty["review_flags"])
        self.assertNotIn("empty", empty["review_flags"])
        self.assertEqual(missing["bucket_reason"], "error")
        self.assertIn("FileNotFoundError", missing["error"])
        self.assertIn("no_detection_review", missing["review_flags"])

    def test_invalid_boxes_cannot_pass_and_output_boxes_are_bounded(self):
        self.rows()
        model = FakeModel([FakePrediction([[-2, 5, 20, 30], [150, 1, 160, 20]], [0.9, 0.95])])
        inference.run(self.manifest, self.output, self.root, self.weights, "cpu",
                      model_factory=lambda *_: model)
        result = self.results()[0]
        self.assertEqual(result["bucket"], "recheck_needed")
        self.assertEqual(result["bucket_reason"], "invalid_model_detection")
        self.assertEqual(result["rejected_detection_count"], 1)
        self.assertEqual(result["detections"][0]["xyxy"], [0.0, 5.0, 20.0, 30.0])
        self.assertIn("out_of_bounds_clipped", result["detections"][0]["localization_flags"])

    def test_hash_mismatch_fails_before_model_load(self):
        self.rows()
        self.weights.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            inference.run(self.manifest, self.output, self.root, self.weights, "cpu",
                          model_factory=lambda *_: self.fail("model should not load"))
        self.assertFalse(self.output.exists())

    def test_resume_skips_completed_prefix_and_rejects_changed_manifest(self):
        self.rows(2)
        first = FakeModel([FakePrediction([], [])])
        inference.run(self.manifest, self.output, self.root, self.weights, "cpu", limit=1,
                      model_factory=lambda *_: first)
        second = FakeModel([FakePrediction([[5, 5, 30, 30]], [0.5])])
        summary = inference.run(self.manifest, self.output, self.root, self.weights, "cpu", resume=True,
                                model_factory=lambda *_: second)
        self.assertEqual(summary["resumed_rows"], 1)
        self.assertEqual([r["image_id"] for r in self.results()], ["img-0", "img-1"])
        self.assertEqual(self.results()[1]["bucket"], "probably_passed")
        self.assertFalse(self.output.with_name("results.jsonl.partial").exists())
        rows = self.rows(2)
        rows[0]["species_label"] = "changed"
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "different input"):
            inference.run(self.manifest, self.output, self.root, self.weights, "cpu", resume=True,
                          model_factory=lambda *_: self.fail("model should not load"))

    def test_resume_truncates_only_interrupted_partial_tail(self):
        self.rows(2)
        inference.run(self.manifest, self.output, self.root, self.weights, "cpu", limit=1,
                      model_factory=lambda *_: FakeModel([FakePrediction([], [])]))
        partial = self.output.with_name(self.output.name + ".partial")
        self.output.rename(partial)
        with partial.open("ab") as stream:
            stream.write(b'{"image_id": "interrupted"')
        summary = inference.run(self.manifest, self.output, self.root, self.weights, "cpu", resume=True,
                                model_factory=lambda *_: FakeModel([FakePrediction([], [])]))
        self.assertEqual(summary["resumed_rows"], 1)
        self.assertEqual(len(self.results()), 2)
        self.assertFalse(partial.exists())

    def test_optimization_api_uses_device_dtype(self):
        instances = []
        class StubRFDETR:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                instances.append(self)
            def inference(self, **kwargs):
                self.optimization = kwargs
        fake_torch = types.SimpleNamespace(float16="half", float32="full")
        fake_rfdetr = types.SimpleNamespace(RFDETRMedium=StubRFDETR)
        with patch.dict("sys.modules", {"torch": fake_torch, "rfdetr": fake_rfdetr}):
            inference.load_model(self.weights, "cuda")
            inference.load_model(self.weights, "cpu")
        self.assertEqual(instances[0].optimization, {"compile": True, "dtype": "half"})
        self.assertEqual(instances[1].optimization, {"compile": False, "dtype": "full"})
        self.assertEqual(instances[0].kwargs["device"], "cuda")


if __name__ == "__main__":
    unittest.main()
