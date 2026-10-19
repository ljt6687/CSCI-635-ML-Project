"""Regressions for PyLabel blank rows and >0.80 IoU box union."""

import unittest

from scripts.squirrel_v2_review_export import export_rows
from scripts.build_squirrel_v2_reviewed_combined import box_iou, merge_heavy_overlaps


def template():
    return {
        "info": {},
        "categories": [{"id": 1, "name": "Sciurus carolinensis"}, {"id": 2, "name": "generic_squirrel"}],
        "images": [
            {"id": 1, "file_name": "a.jpg", "folder": "images", "width": 100, "height": 100,
             "annotation_basis": "v1_model_provisional"},
            {"id": 2, "file_name": "b.jpg", "folder": "images", "width": 100, "height": 100,
             "annotation_basis": "v1_model_provisional"},
        ],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 5, 20, 20],
                         "area": 400, "iscrowd": 0}],
    }


def row(i, name, box=None):
    result = {"img_id": str(i), "img_filename": name, "img_folder": "images",
              "img_width": "100", "img_height": "100", "cat_id": "", "cat_name": "",
              "ann_bbox_xmin": "", "ann_bbox_ymin": "", "ann_bbox_xmax": "", "ann_bbox_ymax": ""}
    if box:
        result.update({"cat_id": "1", "cat_name": "Sciurus carolinensis",
                       "ann_bbox_xmin": str(box[0]), "ann_bbox_ymin": str(box[1]),
                       "ann_bbox_xmax": str(box[2]), "ann_bbox_ymax": str(box[3])})
    return result



class ReconcileTests(unittest.TestCase):
    def test_blank_category_retains_empty_image_without_export_error(self):
        coco, audit = export_rows([row(1, "a.jpg", [5, 5, 25, 25]), row(2, "b.jpg")], template())
        assert len(coco["images"]) == 2
        assert len(coco["annotations"]) == 1
        assert coco["annotations"][0]["id"] == 1
        assert audit["manually_removed_image_ids"] == []


    def test_removed_image_and_changed_box_are_preserved(self):
        coco, audit = export_rows([row(1, "a.jpg", [3, 3, 30, 30])], template())
        assert [image["id"] for image in coco["images"]] == [1]
        assert audit["manually_removed_image_ids"] == [2]
        assert audit["changed_image_ids"] == [1]
        assert coco["annotations"][0]["bbox"] == [3, 3, 27, 27]
        assert coco["annotations"][0]["annotation_basis"] == "human_pylabel_csv"


    def test_iou_union_merges_same_class_above_threshold_only(self):
        coco = template()
        coco["annotations"] = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 5, 20, 20], "area": 400},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [6, 6, 20, 20], "area": 400},
            {"id": 3, "image_id": 1, "category_id": 2, "bbox": [5, 5, 20, 20], "area": 400},
            {"id": 4, "image_id": 1, "category_id": 1, "bbox": [70, 70, 20, 20], "area": 400},
        ]
        assert box_iou(coco["annotations"][0]["bbox"], coco["annotations"][1]["bbox"]) > 0.8
        result, audit = merge_heavy_overlaps(coco, set())
        assert len(result["annotations"]) == 3
        assert len(audit) == 1
        merged = next(a for a in result["annotations"] if a["id"] == 1)
        assert merged["bbox"] == [5, 5, 21, 21]
        assert merged["merged_from_annotation_ids"] == [1, 2]
        assert {a["id"] for a in result["annotations"]} == {1, 3, 4}


    def test_reversed_pylabel_drag_is_normalized_and_audited(self):
        coco, audit = export_rows([row(1, "a.jpg", [25, 25, 5, 5])], template())
        self.assertEqual(coco["annotations"][0]["bbox"], [5, 5, 20, 20])
        self.assertEqual(len(audit["normalized_reversed_boxes"]), 1)
        self.assertEqual(audit["normalized_reversed_boxes"][0]["original_xyxy"], [25, 25, 5, 5])

    def test_human_changed_images_are_not_union_merged(self):
        coco = template()
        coco["annotations"].append({"id": 2, "image_id": 1, "category_id": 1,
                                    "bbox": [6, 6, 20, 20], "area": 400})
        result, audit = merge_heavy_overlaps(coco, {1})
        assert len(result["annotations"]) == 2
        assert audit == []

class FullInferenceUnionTests(unittest.TestCase):
    def test_union_preserves_top_score_and_distinct_classes(self):
        from scripts.merge_squirrel_v2_inference_boxes import union_detections
        detections = [
            {"xyxy": [0, 0, 20, 20], "score": 0.85, "class_id": 0,
             "species_label": "Sciurus niger", "localization_flags": []},
            {"xyxy": [1, 1, 21, 21], "score": 0.45, "class_id": 0,
             "species_label": "Sciurus niger", "localization_flags": ["edge_box"]},
            {"xyxy": [0, 0, 20, 20], "score": 0.8, "class_id": 1,
             "species_label": "Sciurus niger", "localization_flags": []},
        ]
        merged, groups = union_detections(detections)
        self.assertEqual(groups, [[0, 1]])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["score"], 0.85)
        self.assertEqual(merged[0]["xyxy"], [0, 0, 21, 21])
        self.assertEqual(merged[0]["localization_flags"], ["edge_box"])
