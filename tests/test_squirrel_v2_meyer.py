"""Focused tests for Meyer VOC reference parsing and detector comparison."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts import evaluate_squirrel_v2_meyer as meyer


class MeyerEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / meyer.SOURCE_RELATIVE
        self.source.mkdir(parents=True)
        self.predictions = self.root / "predictions.jsonl"
        self.output = self.root / "comparison"
        self.references = []
        for index in range(30):
            folder = self.source / f"camera_{index // 10}"
            folder.mkdir(exist_ok=True)
            image = folder / f"squirrel_{index:02d}.JPG"
            Image.new("RGB", (100, 80), "white").save(image)
            Image.new("RGB", (100, 80), "white").save(folder / f"squirrel_{index:02d}-rendered.JPG")
            self.write_xml(image, truncated=index == 2)
            self.references.append(image)

    def write_xml(self, image, *, filename=None, width=100, box=(10, 10, 30, 30), truncated=False):
        x1, y1, x2, y2 = box
        image.with_suffix(".xml").write_text(
            f"<annotation><filename>{filename or image.name}</filename>"
            f"<size><width>{width}</width><height>80</height></size>"
            f"<object><name>Squirrel</name><difficult>0</difficult>"
            f"<truncated>{int(truncated)}</truncated><bndbox>"
            f"<xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax>"
            f"</bndbox></object></annotation>", encoding="utf-8")

    def row(self, image, detections=None):
        return {"image_id": f"meyer_trailcam:{image.relative_to(self.root / 'data/squirrel_sources/meyer_trailcam').as_posix()}",
                "image_path": image.relative_to(self.root).as_posix(), "source": "meyer_trailcam",
                "species_label": "generic_squirrel", "bucket": "probably_passed", "error": None,
                "detections": detections if detections is not None else [{"xyxy": [10, 10, 30, 30], "score": 0.9}]}

    def write_predictions(self, rows):
        self.predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_cli_matches_one_to_one_and_reports_all_thresholds(self):
        rows = [self.row(image) for image in self.references]
        rows[0]["detections"] = [{"xyxy": [10, 10, 30, 30], "score": 0.45},
                                  {"xyxy": [10, 10, 30, 30], "score": 0.45}]
        rows[1]["detections"] = []
        rows[2]["detections"] = [{"xyxy": [10, 10, 20, 30], "score": 0.50}]
        self.write_predictions([{"source": "inaturalist", "image_id": "other"}, *rows])
        self.assertEqual(meyer.main(["--predictions", str(self.predictions), "--output-dir", str(self.output),
                                     "--repo-root", str(self.root)]), 0)
        report = json.loads((self.output / "meyer_report.json").read_text())
        per_image = [json.loads(line) for line in (self.output / "meyer_per_image.jsonl").read_text().splitlines()]
        self.assertEqual(report["source_inventory"], {"original_jpgs": 30, "rendered_jpgs_excluded": 30, "xml_files": 30})
        self.assertEqual(report["reference_object_count"], 30)
        self.assertEqual(report["metrics"]["0.45"]["tp"], 29)
        self.assertEqual(report["metrics"]["0.45"]["fp"], 1)
        self.assertEqual(report["metrics"]["0.45"]["fn"], 1)
        self.assertAlmostEqual(report["metrics"]["0.45"]["precision"], 29 / 30)
        self.assertAlmostEqual(report["metrics"]["0.45"]["f1"], 29 / 30)
        self.assertAlmostEqual(report["metrics"]["0.45"]["mean_matched_iou"], 28.5 / 29)
        self.assertEqual([(report["metrics"][key]["tp"], report["metrics"][key]["fp"], report["metrics"][key]["fn"])
                          for key in ("0.20", "0.50", "0.80")], [(29, 1, 1), (28, 0, 2), (27, 0, 3)])
        self.assertEqual(per_image[0]["thresholds"]["0.45"]["false_positive_detection_indices"], [1])
        self.assertEqual(per_image[1]["thresholds"]["0.45"]["missed_reference_indices"], [0])
        self.assertEqual(per_image[2]["thresholds"]["0.50"]["matches"][0]["iou"], 0.5)
        self.assertTrue(per_image[2]["reference_objects"][0]["truncated"])
        self.assertEqual(len(report["per_image_misses_and_false_positives_at_0.45"]), 2)
        self.assertTrue(all("-rendered" not in row["image_path"] for row in per_image))

    def test_rejects_missing_and_duplicate_prediction_rows(self):
        rows = [self.row(image) for image in self.references]
        self.write_predictions(rows[:-1])
        with self.assertRaisesRegex(ValueError, "missing 1 Meyer prediction rows"):
            meyer.evaluate(self.predictions, self.output, self.root)
        self.write_predictions(rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate Meyer image_id"):
            meyer.evaluate(self.predictions, self.output, self.root)
        self.assertFalse(self.output.exists())

    def test_rejects_rendered_prediction_and_bad_xml(self):
        rows = [self.row(image) for image in self.references]
        rows[0] = self.row(self.references[0].with_name("squirrel_00-rendered.JPG"))
        self.write_predictions(rows)
        with self.assertRaisesRegex(ValueError, "rendered JPGs are excluded"):
            meyer.evaluate(self.predictions, self.output, self.root)
        self.write_xml(self.references[0], filename="wrong.JPG")
        with self.assertRaisesRegex(ValueError, "XML filename does not match"):
            meyer.load_references(self.root)
        self.write_xml(self.references[0], box=(10, 10, 101, 30))
        with self.assertRaisesRegex(ValueError, "outside"):
            meyer.load_references(self.root)
        self.write_xml(self.references[0], width=101)
        with self.assertRaisesRegex(ValueError, "differs from image size"):
            meyer.load_references(self.root)

    def test_manifest_audit_reports_rendered_rows(self):
        path = self.root / "output/v2_annotation/source_manifest.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(self.row(self.references[0].with_name("squirrel_00-rendered.JPG"))) + "\n")
        audit = meyer.manifest_audit(self.root)
        self.assertEqual(audit["meyer_rows"], 1)
        self.assertEqual(len(audit["rendered_rows"]), 1)


if __name__ == "__main__":
    unittest.main()
