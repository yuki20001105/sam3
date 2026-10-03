from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from class_config import get_class_by_id
from config import IMAGE_EXTENSIONS
from state_manager import annotation_id, load_draft


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ) if image_dir.is_dir() else []


def load_annotations(dataset_dir: Path, image_path: Path, width: int, height: int, classes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    draft = load_draft(dataset_dir, image_path.name)
    if draft is not None:
        annotations = draft.get("annotations")
        return annotations if isinstance(annotations, list) else []
    json_path = dataset_dir / "annotations" / f"{image_path.stem}.json"
    if json_path.is_file():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        annotations = data.get("annotations")
        return annotations if isinstance(annotations, list) else []

    label_path = dataset_dir / "labels" / f"{image_path.stem}.txt"
    if not label_path.is_file():
        return []
    annotations = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id, center_x, center_y, box_width, box_height = parts[:5]
        class_index = int(class_id)
        class_config = get_class_by_id(classes, class_index)
        if class_config is None:
            continue
        cx, cy = float(center_x) * width, float(center_y) * height
        bw, bh = float(box_width) * width, float(box_height) * height
        annotations.append({
            "annotation_id": annotation_id(),
            "class_id": class_index,
            "class_name": class_config["name"],
            "bbox_xyxy": [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
            "confidence": 1.0,
            "source": "saved",
            "human_verified": True,
        })
    return annotations


def validate_annotations(annotations: list[dict[str, Any]], width: int, height: int, classes: list[dict[str, Any]]) -> list[str]:
    errors = []
    class_ids = {item["id"] for item in classes}
    class_names = {item["name"] for item in classes}
    for index, annotation in enumerate(annotations):
        bbox = annotation.get("bbox_xyxy", [])
        class_name = annotation.get("class_name", "")
        if class_name not in class_names or annotation.get("class_id") not in class_ids:
            errors.append(f"BBox {index + 1}: unknown class '{class_name}'")
        if len(bbox) != 4:
            errors.append(f"BBox {index + 1}: bbox must have four values")
            continue
        x1, y1, x2, y2 = map(float, bbox)
        if not all(value == value and abs(value) != float("inf") for value in (x1, y1, x2, y2)):
            errors.append(f"BBox {index + 1}: NaN or Infinity")
        if x2 <= x1 or y2 <= y1:
            errors.append(f"BBox {index + 1}: width and height must be positive")
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            errors.append(f"BBox {index + 1}: outside image bounds")
    return errors
