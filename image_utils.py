from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def display_size(image: Image.Image, max_width: int = 900) -> tuple[int, int]:
    width, height = image.size
    scale = min(1.0, max_width / width)
    return round(width * scale), round(height * scale)


def display_to_original_point(point: tuple[float, float], original: tuple[int, int], display: tuple[int, int]) -> list[float]:
    original_width, original_height = original
    display_width, display_height = display
    return [point[0] * original_width / display_width, point[1] * original_height / display_height]


def original_to_display_point(point: tuple[float, float], original: tuple[int, int], display: tuple[int, int]) -> list[float]:
    original_width, original_height = original
    display_width, display_height = display
    return [point[0] * display_width / original_width, point[1] * display_height / original_height]


def display_to_original_box(box: list[float], original: tuple[int, int], display: tuple[int, int]) -> list[float]:
    x1, y1 = display_to_original_point((box[0], box[1]), original, display)
    x2, y2 = display_to_original_point((box[2], box[3]), original, display)
    return [x1, y1, x2, y2]


def original_to_display_box(box: list[float], original: tuple[int, int], display: tuple[int, int]) -> list[float]:
    x1, y1 = original_to_display_point((box[0], box[1]), original, display)
    x2, y2 = original_to_display_point((box[2], box[3]), original, display)
    return [x1, y1, x2, y2]


def bbox_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def merge_same_class_annotations(existing: list[dict], additions: list[dict], threshold: float = 0.8) -> list[dict]:
    merged = list(existing)
    for addition in additions:
        replacement = next(
            (
                index for index, current in enumerate(merged)
                if current["class_id"] == addition["class_id"]
                and bbox_iou(current["bbox_xyxy"], addition["bbox_xyxy"]) >= threshold
            ),
            None,
        )
        if replacement is None:
            merged.append(addition)
        else:
            merged[replacement] = addition
    return merged


def canvas_prompt_objects(points: list[list[float]], labels: list[int], box: list[float] | None, original: tuple[int, int], display: tuple[int, int]) -> dict:
    objects = []
    for point, label in zip(points, labels):
        x, y = original_to_display_point((point[0], point[1]), original, display)
        objects.append({"type": "circle", "left": x - 6, "top": y - 6, "radius": 6,
                        "fill": "#22c55e" if label else "#ef4444", "stroke": "#ffffff", "strokeWidth": 2})
    if box:
        x1, y1, x2, y2 = original_to_display_box(box, original, display)
        objects.append({"type": "rect", "left": x1, "top": y1, "width": x2 - x1,
                        "height": y2 - y1, "fill": "rgba(0,0,0,0)", "stroke": "#facc15",
                        "strokeWidth": 3, "strokeDashArray": [8, 5]})
    return {"version": "4.4.0", "objects": objects}


def draw_annotations(image: Image.Image, annotations: list[dict[str, Any]], selected_index: int | None = None) -> Image.Image:
    canvas = np.array(image.convert("RGB"))
    for index, annotation in enumerate(annotations):
        x1, y1, x2, y2 = map(int, annotation["bbox_xyxy"])
        color = (255, 70, 40) if index == selected_index else (30, 180, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        label = f"{annotation['class_name']} {float(annotation.get('confidence', 1.0)):.2f}"
        cv2.putText(canvas, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return Image.fromarray(canvas)
