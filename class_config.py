from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ClassConfigError(ValueError):
    pass


def load_classes(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ClassConfigError(f"Class configuration not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    classes = data.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ClassConfigError("classes must be a non-empty list")

    ids: set[int] = set()
    names: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(classes, start=1):
        if not isinstance(item, dict):
            raise ClassConfigError(f"Class {index} must be a mapping")
        class_id = item.get("id")
        name = item.get("name")
        enabled = item.get("enabled")
        if not isinstance(class_id, int) or class_id < 0:
            raise ClassConfigError(f"Class {index}: id must be a non-negative integer")
        if not isinstance(name, str) or not name.strip():
            raise ClassConfigError(f"Class {index}: name must be a non-empty string")
        if not isinstance(enabled, bool):
            raise ClassConfigError(f"Class {index}: enabled must be true or false")
        if class_id in ids:
            raise ClassConfigError(f"Duplicate class id: {class_id}")
        if name in names:
            raise ClassConfigError(f"Duplicate class name: {name}")
        ids.add(class_id)
        names.add(name)
        normalized.append({**item, "id": class_id, "name": name, "enabled": enabled,
                           "prompt": item.get("prompt", name),
                           "display_name": item.get("display_name", name)})
    return sorted(normalized, key=lambda item: item["id"])


def get_enabled_classes(classes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in classes if item["enabled"]]


def get_class_by_id(classes: list[dict[str, Any]], class_id: int) -> dict[str, Any] | None:
    return next((item for item in classes if item["id"] == class_id), None)


def get_class_by_name(classes: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((item for item in classes if item["name"] == name), None)


def class_mapping(classes: list[dict[str, Any]]) -> dict[int, str]:
    return {item["id"]: item["name"] for item in classes}


def add_class(path: Path, name: str, enabled: bool = True) -> dict[str, Any]:
    classes = load_classes(path)
    name = name.strip()
    if not name:
        raise ClassConfigError("Class name must be a non-empty string")
    if get_class_by_name(classes, name) is not None:
        raise ClassConfigError(f"Duplicate class name: {name}")
    new_class = {"id": max(item["id"] for item in classes) + 1, "name": name, "enabled": enabled}
    raw_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_data.setdefault("classes", []).append(new_class)
    path.write_text(yaml.safe_dump(raw_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return new_class
