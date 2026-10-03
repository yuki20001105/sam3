from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from annotation_manager import load_annotations, validate_annotations
from app import (
    annotations_from_draft,
    image_state_key,
    pop_undo,
    processed_classes_from_draft,
    push_undo,
    run_prompt_for_image,
)
from exporters import save_annotations
from image_utils import display_to_original_box, merge_same_class_annotations, original_to_display_box
from state_manager import load_draft, save_draft


CLASSES = [
    {
        "id": 0,
        "name": "TV",
        "display_name": "TV",
        "prompt": "television",
        "enabled": True,
    }
]


def annotation(box: list[float] | None = None) -> dict:
    return {
        "annotation_id": "a1",
        "class_id": 0,
        "class_name": "TV",
        "bbox_xyxy": box or [10.0, 10.0, 50.0, 60.0],
        "confidence": 0.9,
        "source": "test",
        "human_verified": False,
    }


class FakePredictor:
    def predict_text(self, image, prompt, confidence):
        return [annotation()]

    def predict_points(self, image, points, labels, confidence):
        return [annotation([15.0, 15.0, 55.0, 65.0])]

    def predict_box(self, image, box, confidence):
        return [annotation(list(map(float, box)))]


class CoreWorkflowTests(unittest.TestCase):
    def test_legacy_null_prompt_is_safe(self):
        self.assertEqual(processed_classes_from_draft({"prompt": None}), set())
        self.assertEqual(processed_classes_from_draft({}), set())
        self.assertEqual(
            processed_classes_from_draft({"prompt": {"classes": ["TV"]}}),
            {"TV"},
        )

    def test_null_annotations_are_safe(self):
        self.assertEqual(annotations_from_draft({"annotations": None}), [])
        self.assertEqual(annotations_from_draft(None), [])

    def test_session_state_is_isolated_per_dataset(self):
        image_path = Path("/workspace/images/same-name.jpg")
        self.assertNotEqual(
            image_state_key(Path("/workspace/dataset-a"), image_path),
            image_state_key(Path("/workspace/dataset-b"), image_path),
        )

    def test_delete_snapshot_can_be_restored_by_undo(self):
        import streamlit as st

        st.session_state["undo_stack"] = []
        original = [annotation(), {**annotation([20.0, 20.0, 70.0, 70.0]), "annotation_id": "a2"}]
        push_undo("dataset::image", original)
        deleted = original[1:]
        self.assertEqual(len(deleted), 1)
        self.assertEqual(pop_undo("dataset::image"), original)
        self.assertIsNone(pop_undo("dataset::image"))

    def test_text_point_and_box_routes(self):
        predictor = FakePredictor()
        image = Image.new("RGB", (100, 80))
        text = run_prompt_for_image(
            predictor, image, {"mode": "Text", "classes": CLASSES}, 0.5
        )
        point = run_prompt_for_image(
            predictor,
            image,
            {"mode": "Point", "class": CLASSES[0], "points": [[20, 20]], "labels": [1]},
            0.5,
        )
        box = run_prompt_for_image(
            predictor,
            image,
            {"mode": "Box", "class": CLASSES[0], "box": [10, 10, 60, 70]},
            0.5,
        )
        self.assertEqual(text[0]["source"], "sam3.1_text")
        self.assertEqual(point[0]["source"], "sam3.1_point")
        self.assertEqual(box[0]["source"], "sam3.1_box")
        self.assertFalse(point[0]["human_verified"])

    def test_draft_revision_and_null_prompt_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_path = root / "source.jpg"
            Image.new("RGB", (100, 80)).save(image_path)
            saved = [annotation()]
            save_draft(root, image_path.name, 100, 80, saved, prompt=None)
            loaded = load_annotations(root, image_path, 100, 80, CLASSES)
            self.assertEqual(len(loaded), 1)
            save_draft(root, image_path.name, 100, 80, saved, status="approved")
            draft = load_draft(root, image_path.name)
            self.assertEqual(draft["revision"], 2)
            self.assertEqual(draft["status"], "approved")

    def test_export_and_yolo_restore(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jpg"
            Image.new("RGB", (100, 80)).save(source)
            save_annotations(root, source, 100, 80, [annotation()], CLASSES)
            self.assertTrue((root / "labels" / "source.txt").is_file())
            self.assertTrue((root / "annotations" / "source.json").is_file())
            self.assertTrue((root / "classes.txt").is_file())
            (root / "annotations" / "source.json").unlink()
            restored = load_annotations(root, source, 100, 80, CLASSES)
            self.assertEqual(len(restored), 1)
            self.assertTrue(restored[0]["human_verified"])

    def test_validation_merge_and_coordinate_round_trip(self):
        first = annotation()
        replacement = annotation([11.0, 11.0, 51.0, 61.0])
        replacement["annotation_id"] = "a2"
        merged = merge_same_class_annotations([first], [replacement])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["annotation_id"], "a2")
        display_box = original_to_display_box([10, 20, 110, 220], (200, 400), (100, 200))
        self.assertEqual(
            display_to_original_box(display_box, (200, 400), (100, 200)),
            [10.0, 20.0, 110.0, 220.0],
        )
        self.assertEqual(validate_annotations([annotation()], 100, 80, CLASSES), [])
        self.assertTrue(validate_annotations([annotation([-1, 0, 5, 5])], 100, 80, CLASSES))


if __name__ == "__main__":
    unittest.main()
