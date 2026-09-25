"""Meaningful coverage for review selection and provisional COCO conversion."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_squirrel_v2_review.py"
SPEC = importlib.util.spec_from_file_location("prepare_squirrel_v2_review", SCRIPT)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def pair(number, bucket, *, source="inaturalist", species="Sciurus niger", flags=None):
    image_id = f"{source}:{number}"
    manifest = {"image_id": image_id, "image_path": f"images/{number}.jpg",
                "source": source, "species_label": species}
    prediction = dict(manifest, bucket=bucket, model_sha256="abc", review_flags=flags or [],
                      top_score=0.9, image_size={"width": 100, "height": 80},
                      detections=[{"xyxy": [10, 12, 40, 50], "score": 0.9}])
    return manifest, prediction


class ReviewSelectionTests(unittest.TestCase):
    def test_all_recheck_and_flags_plus_stratified_samples(self):
        pairs = [pair(i, "passed") for i in range(20)]
        pairs += [pair(i, "probably_passed", source="harrybaines") for i in range(10)]
        pairs += [pair(i, "recheck_needed", source="meyer_trailcam", species="generic_squirrel")
                  for i in range(4)]
        pairs[10][1]["review_flags"] = ["species_label_provisional_per_box", "possible_poor_localization"]
        pairs[10][1]["detections"][0]["localization_flags"] = ["edge_box"]
        manifest, predictions = map(list, zip(*pairs))
        selected = review.select_review(manifest, predictions)
        counts = {bucket: sum(row["bucket"] == bucket for row in selected)
                  for bucket in ("passed", "probably_passed", "recheck_needed")}
        self.assertGreaterEqual(counts["passed"], 2)
        self.assertEqual(counts["probably_passed"], 5)
        self.assertEqual(counts["recheck_needed"], 4)
        self.assertIn(pairs[10][0]["image_id"], {row["image_id"] for row in selected})
        self.assertEqual(selected, review.select_review(manifest, predictions))

    def test_full_coverage_required(self):
        manifest, prediction = pair(1, "passed")
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            review.select_review([manifest], [])
        prediction["species_label"] = "Sciurus carolinensis"
        with self.assertRaisesRegex(ValueError, "species_label"):
            review.select_review([manifest], [prediction])

    def test_small_material_box_requires_review_even_without_model_localization_flag(self):
        pairs = [pair(i, "passed") for i in range(20)]
        manifest, predictions = map(list, zip(*pairs))
        predictions[13]["detections"] = [{"xyxy": [10, 12, 12, 14], "score": 0.91}]
        queue = review.select_review(manifest, predictions, passed_rate=0, probable_rate=0)
        selected = {row["image_id"]: row for row in queue}
        self.assertIn(pairs[13][0]["image_id"], selected)
        self.assertIn("geometry:any_material_box_below_0.5pct",
                      selected[pairs[13][0]["image_id"]]["selection_reasons"])

    def test_low_score_proposals_are_not_editable_draft_boxes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            (root / "images" / "1.jpg").write_bytes(b"image placeholder")
            manifest, prediction = pair(1, "recheck_needed")
            prediction["detections"].append({"xyxy": [1, 1, 5, 5], "score": 0.10})
            row = review.select_review([manifest], [prediction])[0]
            coco, _ = review.build_review_coco([row], root / "review", root)
            self.assertEqual(len(coco["annotations"]), 1)
            self.assertEqual(len(row["detections"]), 2)

    def test_all_meyer_images_use_xml_boxes_in_review_draft(self):
        pairs = [pair(i, "passed", source="meyer_trailcam", species="generic_squirrel")
                 for i in range(3)]
        manifest, predictions = map(list, zip(*pairs))
        queue = review.select_review(manifest, predictions, passed_rate=0, probable_rate=0)
        self.assertEqual(len(queue), 3)
        self.assertTrue(all("all_meyer_xml_comparison" in row["selection_reasons"] for row in queue))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            for i in range(3):
                (root / "images" / f"{i}.jpg").write_bytes(b"image placeholder")
                (root / "images" / f"{i}.xml").write_text(
                    f"<annotation><filename>{i}.jpg</filename><size><width>100</width>"
                    "<height>80</height></size><object><name>Squirrel</name><bndbox>"
                    "<xmin>2</xmin><ymin>3</ymin><xmax>20</xmax><ymax>30</ymax>"
                    "</bndbox></object></annotation>"
                )
            coco, ids = review.build_review_coco(queue, root / "review", root)
            self.assertEqual(len(coco["annotations"]), 3)
            self.assertTrue(all(a["bbox"] == [2, 3, 18, 27] for a in coco["annotations"]))
            self.assertTrue(all(im["annotation_basis"] == "meyer_voc_reference" for im in coco["images"]))
            self.assertEqual(len(ids), 3)

    def test_review_coco_uses_original_image_and_box(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            (root / "images" / "1.jpg").write_bytes(b"image placeholder")
            (root / "review").mkdir()
            manifest, prediction = pair(1, "recheck_needed")
            row = review.select_review([manifest], [prediction])[0]
            coco, ids = review.build_review_coco([row], root / "review", root)
            self.assertEqual(ids[row["image_id"]], 1)
            self.assertEqual(coco["annotations"][0]["bbox"], [10, 12, 30, 38])
            self.assertEqual(coco["images"][0]["folder"], "../images")
            self.assertEqual(list((root / "review").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
