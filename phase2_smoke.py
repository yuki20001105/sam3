from pathlib import Path

from PIL import Image

from class_config import get_class_by_name, load_classes
from config import DEFAULT_CLASSES_PATH
from exporters import save_annotations
from sam_predictor import Sam31Predictor

image_path = Path("/workspace/data/coco/val2017/000000000139.jpg")
checkpoint = Path("/workspace/weights/sam3.1/sam3.1_multiplex.pt")
classes = load_classes(DEFAULT_CLASSES_PATH)
person = get_class_by_name(classes, "person")
image = Image.open(image_path).convert("RGB")
predictor = Sam31Predictor(checkpoint)

positive = predictor.predict_points(image, person["prompt"], [[438.0, 225.0]], [1], 0.5)
mixed = predictor.predict_points(image, person["prompt"], [[438.0, 225.0], [30.0, 30.0]], [1, 0], 0.5)
box = predictor.predict_box(image, person["prompt"], [400.0, 150.0, 480.0, 310.0], 0.5)

for source, annotations in (("sam3.1_point", positive), ("sam3.1_point", mixed), ("sam3.1_box", box)):
    if not annotations:
        raise RuntimeError(f"No annotations returned for {source}")
    for annotation in annotations:
        annotation["class_id"] = person["id"]
        annotation["class_name"] = person["name"]
        annotation["source"] = source
        annotation["human_verified"] = True

save_annotations(Path("/workspace/dataset_phase2_point"), image_path, image.width, image.height, positive, classes)
save_annotations(Path("/workspace/dataset_phase2_box"), image_path, image.width, image.height, box, classes)
print(f"positive={len(positive)} mixed={len(mixed)} box={len(box)}")
print("phase2 prompt smoke: OK")
