from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATES = {"unprocessed", "needs_review", "approved"}


def image_id_for(path: Path) -> str:
    return path.name


def annotation_id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_path(dataset_dir: Path, image_id: str) -> Path:
    return dataset_dir / "drafts" / f"{Path(image_id).stem}.json"


def load_draft(dataset_dir: Path, image_id: str) -> dict[str, Any] | None:
    path = state_path(dataset_dir, image_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_draft(dataset_dir: Path, image_id: str, width: int, height: int, annotations: list[dict[str, Any]], prompt: dict[str, Any] | None = None, status: str = "needs_review") -> Path:
    if status not in STATES:
        raise ValueError(f"Unknown image status: {status}")
    path = state_path(dataset_dir, image_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_id": image_id,
        "width": width,
        "height": height,
        "status": status,
        "revision": (load_draft(dataset_dir, image_id) or {}).get("revision", 0) + 1,
        "updated_at": now_iso(),
        "prompt": prompt,
        "annotations": annotations,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
