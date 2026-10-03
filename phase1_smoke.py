from pathlib import Path

from PIL import Image

from class_config import get_class_by_name, load_classes
from config import DEFAULT_CLASSES_PATH
from exporters import save_annotations
from sam_predictor import Sam31Predictor

image_path = Path("/workspace/data/coco/val2017/000000532481.jpg")
dataset_dir = Path("/workspace/dataset_phase1_smoke")
checkpoint = Path("/workspace/weights/sam3.1/sam3.1_multiplex.pt")

image = Image.open(image_path).convert("RGB")
predictor = Sam31Predictor(checkpoint)
classes = load_classes(DEFAULT_CLASSES_PATH)
person = get_class_by_name(classes, "person")
annotations = predictor.predict_text(image, person["prompt"])
if not annotations:
    raise RuntimeError("SAM3.1 returned no person candidates for the smoke image")

for annotation in annotations:
    annotation["class_id"] = person["id"]
    annotation["class_name"] = person["name"]
annotations[0]["human_verified"] = True
save_annotations(dataset_dir, image_path, image.width, image.height, annotations, classes)
label_path = dataset_dir / "labels" / f"{image_path.stem}.txt"
json_path = dataset_dir / "annotations" / f"{image_path.stem}.json"
if not label_path.is_file() or not json_path.is_file():
    raise RuntimeError("YOLO label or annotation JSON was not written")

print(f"candidates={len(annotations)}")
print(f"label={label_path}")
print(f"json={json_path}")
print(label_path.read_text(encoding="utf-8").strip())
