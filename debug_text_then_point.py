from pathlib import Path

import torch

from sam_predictor import Sam31Predictor

predictor = Sam31Predictor(Path("/workspace/weights/sam3.1/sam3.1_multiplex.pt"))
started = predictor.predictor.handle_request({"type": "start_session", "resource_path": "/workspace/data/coco/val2017"})
session_id = started["session_id"]
try:
    text = predictor.predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": "person"})["outputs"]
    lines = [f"text_ids={text['out_obj_ids']}", f"text_boxes={text['out_boxes_xywh']}", f"text_masks={text['out_binary_masks'].shape}"]
    if len(text["out_obj_ids"]):
        point = predictor.predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "points": torch.tensor([[0.684375, 0.528169]], dtype=torch.float32),
            "point_labels": torch.tensor([1], dtype=torch.int32),
            "obj_id": int(text["out_obj_ids"][0]),
            "rel_coordinates": True,
        })["outputs"]
        lines += [f"point_ids={point['out_obj_ids']}", f"point_masks={point['out_binary_masks'].shape}"]
    Path("/tmp/text_point_output.txt").write_text("\n".join(lines), encoding="utf-8")
finally:
    predictor.predictor.handle_request({"type": "close_session", "session_id": session_id})
