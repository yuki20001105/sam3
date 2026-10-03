from pathlib import Path

import torch
from PIL import Image

from sam_predictor import Sam31Predictor

image = Image.open("/workspace/data/coco/val2017/000000000139.jpg").convert("RGB")
predictor = Sam31Predictor(Path("/workspace/weights/sam3.1/sam3.1_multiplex.pt"))
original = predictor.predictor.handle_request({"type": "start_session", "resource_path": "/workspace/data/coco/val2017"})
session_id = original["session_id"]
try:
    response = predictor.predictor.handle_request({
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "points": torch.tensor([[0.684375, 0.528169]], dtype=torch.float32),
        "point_labels": torch.tensor([1], dtype=torch.int32),
        "obj_id": 1,
        "rel_coordinates": True,
    })
    outputs = response["outputs"]
    lines = [f"keys {list(outputs.keys())}"]
    for key, value in outputs.items():
        if hasattr(value, "shape"):
            lines.append(f"{key} {tuple(value.shape)} {value.dtype}")
        else:
            lines.append(f"{key} {type(value)} {value}")
    Path("/tmp/point_output.txt").write_text("\n".join(lines), encoding="utf-8")
finally:
    predictor.predictor.handle_request({"type": "close_session", "session_id": session_id})
