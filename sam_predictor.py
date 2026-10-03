from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


class Sam31Predictor:
    def __init__(self, checkpoint_path: Path):
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint_path),
            use_fa3=False,
            use_rope_real=False,
            compile=False,
            async_loading_frames=False,
        )

    def predict_text(self, image: Image.Image, prompt: str, min_confidence: float = 0.0) -> list[dict[str, Any]]:
        return self._run_prompt(image, prompt, min_confidence)

    def predict_points(self, image: Image.Image, points: list[list[float]], labels: list[int], min_confidence: float = 0.0) -> list[dict[str, Any]]:
        # The multiplex model creates a brand-new tracked object directly from
        # points + a fresh obj_id; no preceding text detection is required.
        # The object's mask is only materialized on the frame after an explicit
        # propagate_in_video step (the add_prompt response itself is empty).
        relative_points = [[x / image.width, y / image.height] for x, y in points]
        scratch = Path(tempfile.mkdtemp(prefix="sam31_ui_"))
        image.convert("RGB").save(scratch / "00000.jpg", quality=95)
        session_id = None
        try:
            session_id = self.predictor.handle_request({"type": "start_session", "resource_path": str(scratch)})["session_id"]
            self.predictor.handle_request({
                "type": "add_prompt", "session_id": session_id, "frame_index": 0,
                "points": relative_points, "point_labels": labels, "obj_id": 0,
                "rel_coordinates": True,
            })
            outputs = None
            for event in self.predictor.handle_stream_request({
                "type": "propagate_in_video", "session_id": session_id,
                "propagation_direction": "forward", "start_frame_index": 0,
                "max_frame_num_to_track": 1,
            }):
                if event["frame_index"] == 0:
                    outputs = event["outputs"]
            annotations = self._outputs_to_annotations(outputs, image.width, image.height, None, only_obj_id=0)
            return [item for item in annotations if item["confidence"] >= min_confidence]
        finally:
            if session_id is not None:
                self.predictor.handle_request({"type": "close_session", "session_id": session_id})
            shutil.rmtree(scratch, ignore_errors=True)

    def predict_box(self, image: Image.Image, box_xyxy: list[float], min_confidence: float = 0.0) -> list[dict[str, Any]]:
        # Box-only geometric prompt (no text): the box itself specifies the object.
        x1, y1, x2, y2 = box_xyxy
        relative_xywh = [[x1 / image.width, y1 / image.height, (x2 - x1) / image.width, (y2 - y1) / image.height]]
        return self._run_prompt(
            image, None, min_confidence,
            bounding_boxes=relative_xywh, bounding_box_labels=[1],
        )

    def _run_prompt(self, image: Image.Image, prompt: str | None, min_confidence: float, **prompt_kwargs: Any) -> list[dict[str, Any]]:
        scratch = Path(tempfile.mkdtemp(prefix="sam31_ui_"))
        frame_path = scratch / "00000.jpg"
        image.convert("RGB").save(frame_path, quality=95)
        session_id = None
        try:
            response = self.predictor.handle_request({
                "type": "start_session",
                "resource_path": str(scratch),
            })
            session_id = response["session_id"]
            if prompt is not None:
                response = self.predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt,
                    "rel_coordinates": True,
                    **prompt_kwargs,
                })
            else:
                response = self.predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "rel_coordinates": True,
                    **prompt_kwargs,
                })
            outputs = response["outputs"]
            annotations = self._outputs_to_annotations(outputs, image.width, image.height, prompt)
            return [item for item in annotations if item["confidence"] >= min_confidence]
        finally:
            if session_id is not None:
                self.predictor.handle_request({"type": "close_session", "session_id": session_id})
            shutil.rmtree(scratch, ignore_errors=True)

    @staticmethod
    def _outputs_to_annotations(outputs: dict[str, Any] | None, width: int, height: int, prompt: str | None, only_obj_id: int | None = None) -> list[dict[str, Any]]:
        if outputs is None:
            return []
        masks = outputs.get("out_binary_masks")
        if masks is None:
            return []
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        masks = np.asarray(masks)
        masks = np.squeeze(masks)
        if masks.ndim == 2:
            masks = masks[None, ...]
        scores = outputs.get("out_probs", outputs.get("scores"))
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        scores = np.asarray(scores).reshape(-1) if scores is not None else np.ones(len(masks))
        annotations = []
        object_ids = np.asarray(outputs.get("out_obj_ids", []))
        for index, mask in enumerate(masks):
            if only_obj_id is not None and index < len(object_ids) and int(object_ids[index]) != only_obj_id:
                continue
            ys, xs = np.where(mask.astype(bool))
            if len(xs) == 0:
                continue
            annotations.append({
                "annotation_id": uuid.uuid4().hex,
                "class_id": 0,
                "class_name": prompt or "geometry_prompt",
                "bbox_xyxy": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                "confidence": float(scores[index]) if index < len(scores) else 1.0,
                "source": "sam3.1",
                "human_verified": False,
            })
        return annotations
