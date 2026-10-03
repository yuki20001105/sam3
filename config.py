from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_IMAGE_DIR = Path("/workspace/data/coco/val2017")
DEFAULT_DATASET_DIR = Path("/workspace/dataset")
DEFAULT_CHECKPOINT = Path("/workspace/weights/sam3.1/sam3.1_multiplex.pt")
DEFAULT_CLASSES_PATH = Path("/workspace/sam31_annotation_tool/config/classes.yaml")
