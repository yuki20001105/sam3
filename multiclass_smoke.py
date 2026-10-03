from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO

from class_config import get_class_by_name, load_classes
from config import DEFAULT_CLASSES_PATH
from exporters import save_annotations
from sam_predictor import Sam31Predictor

image_dir = Path("/workspace/data/coco/val2017")
annotation_path = Path("/workspace/data/coco/annotations/instances_val2017.json")
dataset_dir = Path("/workspace/dataset_multiclass_smoke")
checkpoint = Path("/workspace/weights/sam3.1/sam3.1_multiplex.pt")
classes = load_classes(DEFAULT_CLASSES_PATH)
coco = COCO(str(annotation_path))
person_id = coco.getCatIds(catNms=["person"])[0]
chair_id = coco.getCatIds(catNms=["chair"])[0]
candidate_ids = set(coco.getImgIds(catIds=[person_id])) & set(coco.getImgIds(catIds=[chair_id]))
image_info = next(
    info for info in coco.loadImgs(list(candidate_ids))
    if (image_dir / info["file_name"]).is_file()
)
image_path = image_dir / image_info["file_name"]
image = Image.open(image_path).convert("RGB")
predictor = Sam31Predictor(checkpoint)
annotations = []

for name in ["person", "chair"]:
    class_config = get_class_by_name(classes, name)
    for annotation in predictor.predict_text(image, class_config["prompt"], 0.5):
        annotation["class_id"] = class_config["id"]
        annotation["class_name"] = class_config["name"]
        annotations.append(annotation)

if not annotations:
    raise RuntimeError("No multi-class candidates were returned")
save_annotations(dataset_dir, image_path, image.width, image.height, annotations, classes)
print([(item["class_id"], item["class_name"]) for item in annotations])
print((dataset_dir / "labels" / f"{image_path.stem}.txt").read_text(encoding="utf-8").strip())
print((dataset_dir / "classes.txt").read_text(encoding="utf-8").strip())
