from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from annotation_manager import list_images
from class_config import get_enabled_classes, load_classes
from config import DEFAULT_CHECKPOINT, DEFAULT_CLASSES_PATH, DEFAULT_DATASET_DIR, DEFAULT_IMAGE_DIR
from exporters import save_annotations
from sam_predictor import Sam31Predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch SAM3.1 auto-annotation with YAML classes.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=Path("/workspace/dataset_auto"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES_PATH)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="0 means all images.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    classes = load_classes(args.classes)
    selected_classes = get_enabled_classes(classes)
    images = list_images(args.image_dir)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise RuntimeError(f"No supported images in {args.image_dir}")
    if not selected_classes:
        raise RuntimeError("No enabled classes in YAML")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.dataset_dir / "batch_progress.json"
    predictor = Sam31Predictor(args.checkpoint)
    completed = 0
    skipped = 0
    failures: list[dict[str, str]] = []

    for index, image_path in enumerate(tqdm(images, desc="SAM3.1 images"), start=1):
        label_path = args.dataset_dir / "labels" / f"{image_path.stem}.txt"
        if label_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            annotations = []
            for class_config in selected_classes:
                for annotation in predictor.predict_text(image, class_config["prompt"], args.confidence):
                    annotation["class_id"] = class_config["id"]
                    annotation["class_name"] = class_config["name"]
                    annotations.append(annotation)
            save_annotations(args.dataset_dir, image_path, image.width, image.height, annotations, classes)
            completed += 1
        except Exception as error:
            failures.append({"image": image_path.name, "error": str(error)})
        finally:
            progress_path.write_text(
                json.dumps(
                    {
                        "total": len(images),
                        "processed": index,
                        "completed": completed,
                        "skipped": skipped,
                        "failures": failures,
                        "selected_classes": [item["name"] for item in selected_classes],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    print(f"completed={completed} skipped={skipped} failures={len(failures)}")
    print(f"output={args.dataset_dir}")


if __name__ == "__main__":
    main()
