"""Regression checks for squirrel_detection_rfdetr_medium_v2.ipynb extracting helpers directly."""

import ast
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import nbformat
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "squirrel_detection_rfdetr_medium_v2.ipynb"
NB = nbformat.read(NB_PATH, as_version=4)


def definitions(names, namespace):
    nodes = []
    for cell in NB.cells:
        if cell.cell_type == "code":
            for node in ast.parse(cell.source).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                    nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "notebook-helpers", "exec"), namespace)
    return namespace


class NotebookV2Tests(unittest.TestCase):
    def setUp(self):
        self.ns = definitions(
            {"box_iou", "match_multiclass_detections", "detection_metrics"},
            dict(np=np, math=math, CFG={"iou": 0.5})
        )
        self.gt = [dict(box=[0, 0, 10, 10], category_id=1, class_name="Sciurus carolinensis")]
        self.pred = [dict(box=[0, 0, 10, 10], score=0.9, category_id=1, class_name="Sciurus carolinensis")]

    def test_all_cells_compile(self):
        nbformat.validate(NB)
        for i, cell in enumerate(NB.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"notebook_cell_{i}", "exec")

    def test_multiclass_matching_same_and_different_classes(self):
        f = self.ns["match_multiclass_detections"]

        # Exact match
        tp, fp, fn, ious, conf_pairs = f(self.gt, self.pred, 0.5)
        self.assertEqual((tp, fp, fn), (1, 0, 0))
        self.assertEqual(conf_pairs, [("Sciurus carolinensis", "Sciurus carolinensis")])

        # Wrong class prediction (same box, different class)
        pred_wrong_class = [dict(box=[0, 0, 10, 10], score=0.9, category_id=2, class_name="Sciurus niger")]
        tp, fp, fn, ious, conf_pairs = f(self.gt, pred_wrong_class, 0.5)
        self.assertEqual((tp, fp, fn), (0, 1, 1))  # IoU match, wrong species
        self.assertEqual(conf_pairs, [("Sciurus carolinensis", "Sciurus niger")])
        metrics = self.ns["detection_metrics"](
            [dict(image_id=1, gt=self.gt)], {1: pred_wrong_class}, 0.5
        )
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["f1"]), (0, 1, 1, 0.0))

        # Duplicate prediction -> 1 TP, 1 FP
        tp, fp, fn, ious, conf_pairs = f(self.gt, self.pred * 2, 0.5)
        self.assertEqual((tp, fp, fn), (1, 1, 0))

        # Miss -> 1 FN
        tp, fp, fn, ious, conf_pairs = f(self.gt, [], 0.5)
        self.assertEqual((tp, fp, fn), (0, 0, 1))
        self.assertEqual(conf_pairs, [("Sciurus carolinensis", "background")])

        # False positive on empty image
        tp, fp, fn, ious, conf_pairs = f([], self.pred, 0.5)
        self.assertEqual((tp, fp, fn), (0, 1, 0))
        self.assertEqual(conf_pairs, [("background", "Sciurus carolinensis")])

    def test_seeded_capped_train_sampler(self):
        import torch
        ns = definitions(
            {"build_train_sampling_plan", "EpochWeightedSampler"},
            dict(math=math, torch=torch),
        )
        doc = {
            "categories": [{"id": 1}, {"id": 2}],
            "images": [{"id": i} for i in range(1, 10)],
            "annotations": [
                {"image_id": i, "category_id": 1 if i <= 8 else 2} for i in range(1, 10)
            ],
        }
        weights, length, counts = ns["build_train_sampling_plan"](doc, list(range(1, 10)), 4)
        self.assertEqual(counts, {1: 8, 2: 1})
        self.assertEqual(length, 12)  # full batch x accumulation windows
        self.assertAlmostEqual(max(weights) / min(weights), 1.5)
        epoch = [0]
        sampler = ns["EpochWeightedSampler"](weights, length, 42, lambda: epoch[0])
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        epoch[0] = 1
        self.assertNotEqual(first, list(sampler))
        with self.assertRaises(ValueError):
            ns["build_train_sampling_plan"](doc, list(range(2, 11)), 4)

    def test_train_loader_uses_weighted_sampler_only(self):
        import torch
        from types import SimpleNamespace

        class FakeBase:
            @staticmethod
            def _require_dataset(dataset, split):
                return dataset
            def _resolve_batch_size(self):
                return 2
            def val_dataloader(self):
                return "natural validation loader"
            def test_dataloader(self):
                return "natural test loader"

        class FakeDataset:
            ids = list(range(1, 12))
            def __len__(self):
                return len(self.ids)
            def __getitem__(self, index):
                return self.ids[index]

        doc = {
            "categories": [{"id": i} for i in range(1, 12)],
            "images": [{"id": i} for i in range(1, 12)],
            "annotations": [{"image_id": i, "category_id": i} for i in range(1, 12)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            ns = definitions(
                {"build_train_sampling_plan", "EpochWeightedSampler", "AuditedRFDETRDataModule"},
                dict(math=math, torch=torch, RFDETRDataModule=FakeBase,
                    splits={"train": doc}, CFG={"seed": 42}, RUN=Path(tmp),
                    save_json=lambda path, value: Path(path).write_text(json.dumps(value))),
            )
            dm = ns["AuditedRFDETRDataModule"]()
            dm._dataset_train = FakeDataset()
            dm.train_config = SimpleNamespace(grad_accum_steps=2)
            dm.trainer = SimpleNamespace(world_size=1, current_epoch=0)
            dm._collate_fn = lambda batch: batch
            dm._num_workers = 0
            dm._pin_memory = False
            dm._persistent_workers = False
            dm._prefetch_factor = None
            loader = dm.train_dataloader()
            self.assertIsInstance(loader.sampler, ns["EpochWeightedSampler"])
            self.assertEqual(len(list(loader)), 6)  # 12 draws, 2 per microbatch
            self.assertEqual(dm.val_dataloader(), "natural validation loader")
            self.assertEqual(dm.test_dataloader(), "natural test loader")
            self.assertTrue((Path(tmp) / "train_sampling.json").is_file())

    def test_audited_dataset_gate_and_class_mapping(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            categories = [{"id": i + 1, "name": f"class_{i}"} for i in range(11)]
            ann_hashes = {}
            for split in ("train", "valid", "test"):
                folder = data / split
                folder.mkdir()
                doc = {
                    "categories": categories,
                    "images": [{"id": i + 1} for i in range(11)],
                    "annotations": [
                        {"id": i + 1, "image_id": i + 1, "category_id": i + 1}
                        for i in range(11)
                    ],
                }
                ann_file = folder / "_annotations.coco.json"
                ann_file.write_text(json.dumps(doc))
                ann_hashes[split] = hashlib.sha256(ann_file.read_bytes()).hexdigest()
            manifest = dict(version=3, num_classes=11, qa_status="passed",
                source_fingerprint="source-sha", source_image_fingerprint="image-sha",
                audit_decisions_sha256="decisions-sha",
                categories=[c["name"] for c in categories], annotation_hashes=ann_hashes)
            marker = data / ".complete.json"
            ns = definitions({"verify_dataset"}, dict(DATA=data, CFG={"num_classes": 11},
                json=json, sha256=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()))
            with self.assertRaises(AssertionError):
                ns["verify_dataset"]()  # no marker: never auto-export
            marker.write_text(json.dumps(manifest))
            self.assertEqual(ns["verify_dataset"]()["qa_status"], "passed")
            manifest["qa_status"] = "failed"
            marker.write_text(json.dumps(manifest))
            with self.assertRaises(AssertionError):
                ns["verify_dataset"]()
            manifest["qa_status"] = "passed"
            marker.write_text(json.dumps(manifest))
            ann_file = data / "valid" / "_annotations.coco.json"
            bad = json.loads(ann_file.read_text())
            bad["categories"][0]["id"] = 99
            ann_file.write_text(json.dumps(bad))
            manifest["annotation_hashes"]["valid"] = hashlib.sha256(ann_file.read_bytes()).hexdigest()
            marker.write_text(json.dumps(manifest))
            with self.assertRaises(AssertionError):
                ns["verify_dataset"]()

    def test_redaction_and_safe_writer(self):
        import re
        with tempfile.TemporaryDirectory() as tmp:
            ns = definitions(
                {"redact", "assert_secret_free", "SafeWriter"},
                dict(re=re, io=io, json=json, _SECRETS=["FAKE_SECRET_123"], RUN=Path(tmp))
            )
            target = io.StringIO()
            writer = ns["SafeWriter"](target)
            writer.write("FAKE_SEC")
            writer.write("RET_123 https://example.test/file?api_key=unknown\n")
            writer.finish()
            self.assertNotIn("FAKE_SECRET_123", target.getvalue())
            self.assertNotIn("unknown", target.getvalue())
            self.assertIn("[REDACTED]", target.getvalue())
            with self.assertRaises(RuntimeError):
                ns["assert_secret_free"]({"key": "FAKE_SECRET_123"})

    def test_coco_multiclass_evaluation(self):
        doc = {
            "images": [{"id": 1, "width": 100, "height": 100}],
            "categories": [
                {"id": 1, "name": "Sciurus carolinensis"},
                {"id": 2, "name": "Sciurus niger"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}
            ]
        }
        ns = definitions({"coco_metrics"}, dict(COCO=COCO, COCOeval=COCOeval, splits={"test": doc}))
        predictions = {1: [dict(box=[0, 0, 10, 10], score=0.95, category_id=1, class_name="Sciurus carolinensis")]}

        with contextlib.redirect_stdout(io.StringIO()):
            metrics = ns["coco_metrics"]("test", [dict(image_id=1)], predictions)
        self.assertAlmostEqual(metrics["AP50"], 1.0)
        self.assertAlmostEqual(metrics["mAP_50_95"], 1.0)

    def test_group_leakage_detection(self):
        import hashlib
        records = [
            dict(split="train", file_name="a.jpg", group_id="burst_1", pixel_hash="same", phash="0000000000000000"),
            dict(split="valid", file_name="b.jpg", group_id="burst_1", pixel_hash="same", phash="0000000000000000"),
            dict(split="test", file_name="c.jpg", group_id="burst_2", pixel_hash="other", phash="0000000000000001"),
            dict(split="train", file_name="d.jpg", group_id="burst_3", pixel_hash="diff", phash="ffffffffffffffff"),
        ]
        ns = definitions({"find_leakage_pairs"}, dict(np=np, hashlib=hashlib, CFG={"leakage_max_distance": 4}))
        pairs = ns["find_leakage_pairs"](records)

        confirmed = [p for p in pairs if p["confirmed"]]
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(records[confirmed[0]["left"]]["group_id"], records[confirmed[0]["right"]]["group_id"])


if __name__ == "__main__":
    unittest.main()
