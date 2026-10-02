"""Contracts for immutable conversion, common metrics, and notebook workflows."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import nbformat
from PIL import Image

from csci_635_ml_project.baseline import create_run, default_config, redact
from csci_635_ml_project.baseline_data import (coco_to_yolo, dataset_fingerprint, export_yolo,
                                             save_json, sha256, verify_dataset)
from csci_635_ml_project.baseline_reports import (coco_metrics, detection_metrics,
                                                evaluate_report, load_predictions, match_detections, select_threshold)
from csci_635_ml_project.baseline_comparison import validate_benchmarks, validate_runs

ROOT = Path(__file__).resolve().parents[1]


def fixture_data(parent):
    data = parent / "coco"
    categories = [dict(id=i + 1, name=f"class_{i}") for i in range(11)]
    manifest = dict(version=3, qa_status="passed", num_classes=11,
                    source_fingerprint="source", source_image_fingerprint="images", audit_decisions_sha256="audit",
                    categories=[c["name"] for c in categories], annotation_hashes={}, split_counts={}, annotation_counts={})
    for s, split in enumerate(("train", "valid", "test")):
        folder = data / split; folder.mkdir(parents=True)
        images = [dict(id=i + 1, file_name=f"image_{i}.jpg", width=20, height=12) for i in range(11)]
        for i, im in enumerate(images): Image.new("RGB", (20, 12), (s * 70 + i * 3, i * 13, s * 40)).save(folder / im["file_name"])
        doc = dict(images=images, categories=categories,
                   annotations=[dict(id=i + 1, image_id=i + 1, category_id=i + 1, bbox=[1, 2, 5, 6], area=30, iscrowd=0) for i in range(11)])
        save_json(folder / "_annotations.coco.json", doc)
        manifest["annotation_hashes"][split] = sha256(folder / "_annotations.coco.json")
        manifest["split_counts"][split] = 11; manifest["annotation_counts"][split] = 11
    save_json(data / ".complete.json", manifest)
    return data, manifest


class DatasetTests(unittest.TestCase):
    def test_round_trip_mapping_splits_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp); data, manifest = fixture_data(parent)
            before = {str(p): sha256(p) for p in data.rglob("*") if p.is_file()}
            yaml = export_yolo(data, parent / "yolo")
            converted = json.loads((yaml.parent / "export_manifest.json").read_text())
            self.assertEqual(len(converted["images"]), 33)
            self.assertEqual(converted["dataset_fingerprint"], dataset_fingerprint(manifest))
            for row in converted["images"]:
                values = (yaml.parent / row["label"]).read_text().split()
                self.assertEqual(int(values[0]), row["image_id"] - 1)
                cx, cy, w, h = map(float, values[1:])
                self.assertAlmostEqual((cx - w / 2) * 20, 1)
                self.assertAlmostEqual((cy - h / 2) * 12, 2)
                self.assertAlmostEqual(w * 20, 5)
                self.assertAlmostEqual(h * 12, 6)
                self.assertEqual(sha256(yaml.parent / row["image"]), before[str(data / row["split"] / row["source_file"])])
            self.assertEqual(export_yolo(data, parent / "yolo"), yaml)
            self.assertEqual(before, {str(p): sha256(p) for p in data.rglob("*") if p.is_file()})
            (yaml.parent / converted["images"][0]["label"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "export changed"): export_yolo(data, parent / "yolo")

    def test_bad_boxes_are_not_silently_clipped(self):
        for box in ([0, 0, 0, 1], [-1, 0, 2, 2], [19, 0, 2, 2], [0, 0, float("nan"), 2]):
            with self.assertRaises(ValueError): coco_to_yolo(box, 20, 12)

    def test_stale_manifest_and_incomplete_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp); data, _ = fixture_data(parent)
            destination = parent / "yolo"; destination.mkdir()
            with self.assertRaisesRegex(ValueError, "Incomplete"): export_yolo(data, destination)
            with (data / "train/_annotations.coco.json").open("a") as f: f.write(" ")
            with self.assertRaisesRegex(ValueError, "Stale"): verify_dataset(data)

    def test_identical_moved_source_reuses_export_and_image_changes_fail(self):
        import shutil
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp); data, _ = fixture_data(parent)
            yaml = export_yolo(data, parent / "yolo")
            other = parent / "moved"; shutil.copytree(data, other)
            self.assertEqual(export_yolo(other, parent / "yolo"), yaml)
            Image.new("RGB", (20, 12), (255, 1, 128)).save(other / "train/image_0.jpg")
            with self.assertRaisesRegex(ValueError, "export changed"):
                export_yolo(other, parent / "yolo")

    def test_empty_images_survive_conversion(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp); data, manifest = fixture_data(parent)
            p = data / "train/_annotations.coco.json"; doc = json.loads(p.read_text())
            doc["images"].append(dict(id=99, file_name="empty.jpg", width=20, height=12))
            Image.new("RGB", (20, 12), (255, 255, 255)).save(data / "train/empty.jpg")
            save_json(p, doc); manifest["annotation_hashes"]["train"] = sha256(p); manifest["split_counts"]["train"] += 1
            save_json(data / ".complete.json", manifest)
            yaml = export_yolo(data, parent / "yolo")
            self.assertEqual((yaml.parent / "labels/train/empty.txt").read_text(), "")

    def test_exact_cross_split_duplicate_blocks_export(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp); data, _ = fixture_data(parent)
            (data / "valid/image_0.jpg").write_bytes((data / "train/image_0.jpg").read_bytes())
            with self.assertRaisesRegex(ValueError, "cross-split duplicate"): export_yolo(data, parent / "yolo")
            self.assertFalse((parent / "yolo").exists())


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.gt = dict(box=[0, 0, 10, 10], category_id=1, class_name="a")
        self.pred = dict(**self.gt, score=.8)
        self.rows = [dict(image_id=1, gt=[self.gt])]

    def test_correct_wrong_duplicate_missing_and_background(self):
        self.assertEqual(match_detections([self.gt], [self.pred], .5)[:3], (1, 0, 0))
        wrong = {**self.pred, "category_id": 2, "class_name": "b"}
        match = match_detections([self.gt], [wrong], .5)
        self.assertEqual(match[:3], (0, 1, 1)); self.assertEqual(match[4], [("a", "b")])
        self.assertEqual(match_detections([self.gt], [self.pred, self.pred], .5)[:3], (1, 1, 0))
        self.assertEqual(match_detections([self.gt], [], .5)[4], [("a", "background")])
        self.assertEqual(match_detections([], [self.pred], .5)[4], [("background", "a")])
        self.assertEqual(detection_metrics([], {}, .5)["f1"], 0)

    def test_threshold_uses_validation_and_highest_tie(self):
        predictions = {1: [self.pred]}
        self.assertAlmostEqual(select_threshold(self.rows, predictions), .8)
        self.assertEqual(detection_metrics(self.rows, predictions, .8)["f1"], 1)
        self.assertEqual(select_threshold(self.rows, {1: []}), 1)

    def test_empty_coco_predictions_have_zero_ap_and_ar(self):
        with tempfile.TemporaryDirectory() as temp:
            doc = dict(images=[dict(id=1, width=10, height=10)], categories=[dict(id=1, name="a")],
                       annotations=[dict(id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10], area=100, iscrowd=0)])
            metrics, per_class = coco_metrics(doc, {1: []}, Path(temp))
            self.assertEqual(metrics["mAP_50_95"], 0)
            self.assertEqual(metrics["AR100"], 0)
            self.assertEqual(per_class["a"]["AP50"], 0)

    def test_prediction_ids_mapping_and_finiteness(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "pred.json"
            doc = dict(images=[dict(id=1)], categories=[dict(id=1, name="a")])
            save_json(p, {"1": [self.pred]}); self.assertEqual(load_predictions(p, doc), {1: [self.pred]})
            for raw in ({"2": []}, {"1": [{**self.pred, "class_name": "wrong"}]}, {"1": [{**self.pred, "score": 2}]}):
                save_json(p, raw)
                with self.assertRaises(ValueError): load_predictions(p, doc)

    def test_report_wrong_class_metrics_and_background_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            image = p / "im.jpg"; Image.new("RGB", (10, 10)).save(image)
            doc = dict(images=[dict(id=1, width=10, height=10)], categories=[dict(id=1, name="a"), dict(id=2, name="b")],
                       annotations=[dict(id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10], area=100, iscrowd=0)])
            rows = [{**self.rows[0], "path": str(image)}]
            metrics = evaluate_report(doc, rows, {1: [{**self.pred, "category_id": 2, "class_name": "b"}]}, .5, p / "report")
            self.assertEqual(metrics["operating"]["f1"], 0)
            self.assertEqual(metrics["per_class"][0]["fn"], 1)
            self.assertEqual(metrics["per_class"][1]["fp"], 1)
            import pandas as pd
            cm = pd.read_csv(p / "report/confusion_counts.csv", index_col=0)
            self.assertEqual(cm.loc["a", "b"], 1)
            self.assertIn("background", cm.columns)


class WorkflowTests(unittest.TestCase):
    def test_defaults_and_clean_compilable_notebooks(self):
        for model in ("yolo26", "yolov5"):
            config = default_config(model)
            self.assertEqual((config["epochs"], config["patience"], config["imgsz"], config["seed"]), (200, 30, 640, 42))
        for path in ROOT.glob("squirrel_detection_*v2.ipynb"):
            notebook = nbformat.read(path, as_version=4); nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code": compile(cell.source, str(path), "exec")
                if "rfdetr" not in path.name:
                    self.assertFalse(cell.get("outputs", []))
                    self.assertIsNone(cell.get("execution_count"))

    def test_resume_config_and_dataset_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data, _ = fixture_data(root); config = default_config("yolov5")
            run = create_run(root, config, data=data)
            self.assertEqual(create_run(root, config, resume_run=run, data=data), run)
            with self.assertRaisesRegex(ValueError, "configuration"):
                create_run(root, {**config, "patience": 50}, resume_run=run, data=data)
            saved = json.loads((run / "dataset_manifest.json").read_text()); saved["categories"].reverse()
            save_json(run / "dataset_manifest.json", saved)
            with self.assertRaisesRegex(ValueError, "dataset changed"):
                create_run(root, config, resume_run=run, data=data)

    def test_comparison_rejects_dataset_checkpoint_and_threshold_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data, manifest = fixture_data(root)
            run = root / "run"; run.mkdir()
            checkpoint = run / "best.pt"; checkpoint.write_bytes(b"test checkpoint")
            save_json(run / "training_complete.json", dict(best_checkpoint="best.pt", sha256=sha256(checkpoint)))
            save_json(run / "dataset_manifest.json", manifest)
            save_json(run / "config.json", dict(model="yolov5", heads=["accuracy"]))
            save_json(run / "decision_threshold.json", dict(threshold=.5, checkpoint_sha256=sha256(checkpoint),
                       dataset_fingerprint=dataset_fingerprint(manifest), inference_head="accuracy"))
            for split in ("valid", "test"):
                folder = run / "reports" / split; folder.mkdir(parents=True)
                save_json(folder / "predictions.json", {})
                save_json(folder / "metrics.json", dict(confidence_threshold=.5,
                    metadata=dict(dataset_fingerprint=dataset_fingerprint(manifest), checkpoint_sha256=sha256(checkpoint), inference_head="accuracy")))
            validate_runs({"yolov5": run}, manifest)
            changed = copy.deepcopy(manifest); changed["split_counts"]["test"] += 1
            with self.assertRaisesRegex(ValueError, "incompatible dataset"): validate_runs({"yolov5": run}, changed)
            p = run / "reports/test/metrics.json"; metrics = json.loads(p.read_text()); metrics["confidence_threshold"] = .6; save_json(p, metrics)
            with self.assertRaisesRegex(ValueError, "threshold/head"): validate_runs({"yolov5": run}, manifest)
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint hash"): validate_runs({"yolov5": run}, manifest)

    def test_reference_pixel_guard(self):
        import csv,hashlib
        from csci_635_ml_project.baseline_comparison import verify_reference_pixels
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data, _ = fixture_data(root); _, docs = verify_dataset(data)
            report = root / "reference/reports"; report.mkdir(parents=True)
            with (report / "image_inventory.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=["split", "image_id", "file_name", "pixel_hash"]); writer.writeheader()
                for split, doc in docs.items():
                    for im in doc["images"]:
                        with Image.open(data / split / im["file_name"]) as opened:
                            rgb = opened.convert("RGB")
                        digest = hashlib.sha256(f"{rgb.width}x{rgb.height}".encode() + rgb.tobytes()).hexdigest()
                        writer.writerow(dict(split=split, image_id=im["id"], file_name=im["file_name"], pixel_hash=digest))
            self.assertEqual(verify_reference_pixels(root / "reference", data, docs)["images_verified"], 33)
            Image.new("RGB", (20, 12), (255, 255, 0)).save(data / "test/image_0.jpg")
            with self.assertRaisesRegex(ValueError, "pixels changed"):
                verify_reference_pixels(root / "reference", data, docs)

    def test_latency_protocol_mismatch(self):
        a = dict(protocol=dict(precision="FP32", image_ids=[1], hardware="GPU-A"))
        validate_benchmarks([a, copy.deepcopy(a)])
        for key, value in [("precision", "FP16"), ("image_ids", [2]), ("hardware", "GPU-B")]:
            b = copy.deepcopy(a); b["protocol"][key] = value
            with self.assertRaisesRegex(ValueError, "different hardware"): validate_benchmarks([a, b])

    def test_checkpoint_class_order_guard(self):
        from csci_635_ml_project.baseline_worker import check_class_names
        check_class_names({0: "a", 1: "b"}, ["a", "b"])
        with self.assertRaisesRegex(ValueError, "class order"):
            check_class_names(["b", "a"], ["a", "b"])

    def test_dual_head_loss_trace_retains_both_branches(self):
        from csci_635_ml_project.baseline_worker import flatten_losses
        values = flatten_losses({"one2many": [1, 2, 3], "one2one": [4, 5, 6]})
        self.assertEqual(len(values), 6)
        self.assertEqual(values["loss/one2many/0"], 1)
        self.assertEqual(values["loss/one2one/2"], 6)

    def test_redaction(self):
        from unittest.mock import patch
        with patch.dict("os.environ", CLEARML_API_SECRET_KEY="sample-private-value"):
            self.assertNotIn("sample-private-value", redact("Failure: sample-private-value"))
        self.assertNotIn("private", redact("https://example.test/file?signature=private&other=1"))


if __name__ == "__main__": unittest.main()
