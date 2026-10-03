from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_annotations(dataset_dir: Path, image_path: Path, width: int, height: int, annotations: list[dict[str, Any]], classes: list[dict[str, Any]]) -> None:
    labels_dir = dataset_dir / "labels"
    annotations_dir = dataset_dir / "annotations"
    images_dir = dataset_dir / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    destination = images_dir / image_path.name
    if image_path.resolve() != destination.resolve():
        destination.write_bytes(image_path.read_bytes())

    classes = sorted(classes, key=lambda item: item["id"])
    class_ids = {item["name"]: item["id"] for item in classes}
    label_lines = []
    normalized = []
    for annotation in annotations:
        x1, y1, x2, y2 = map(float, annotation["bbox_xyxy"])
        class_name = annotation["class_name"]
        class_id = annotation.get("class_id")
        if class_ids.get(class_name) != class_id:
            raise ValueError(f"Class mapping mismatch for {class_name}")
        center_x = ((x1 + x2) / 2) / width
        center_y = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        label_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
        normalized.append({**annotation, "class_id": class_id, "bbox_xyxy": [x1, y1, x2, y2]})

    (labels_dir / f"{image_path.stem}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    max_id = max((item["id"] for item in classes), default=-1)
    class_lines = [f"__unused_id_{index}__" for index in range(max_id + 1)]
    for item in classes:
        class_lines[item["id"]] = item["name"]
    (dataset_dir / "classes.txt").write_text("\n".join(class_lines) + "\n", encoding="utf-8")
    (dataset_dir / "class_mapping.json").write_text(
        json.dumps({"classes": classes}, indent=2), encoding="utf-8"
    )
    metadata = {"image": image_path.name, "width": width, "height": height, "annotations": normalized}
    (annotations_dir / f"{image_path.stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
