from pathlib import Path
import json
import random

# Constants
MATERIAL = "NbSe2" #can be "WS2" or "NbSe2" or ... (other materials - later on)
BASE_DIR = Path(__file__).resolve().parent / MATERIAL

# Setup directories
def setup_directories(base_path, subdirs):
    dirs = {}
    for subdir in subdirs:
        dir_path = base_path / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        dirs[subdir] = dir_path
    return dirs
dirs = setup_directories(BASE_DIR, ["logs", "checkpoints", "configs", "data"])

DATASET_DIR = BASE_DIR / "data" / "2025_01_13_newImages" #for last insert name-of-dataset-folder-to-split
# Paths to the COCO annotation JSON file
ANNOTATION_JSON = "labels_my-project-name_2025-01-10-04-51-39.json"



# Main script
if __name__ == "__main__":

    # Define split ratio (e.g., 80% train, 20% validation)
    train_ratio = 0.8

    # Load the COCO dataset annotations
    with open(DATASET_DIR / ANNOTATION_JSON, "r") as f:
        coco_data = json.load(f)

    # Shuffle the image entries randomly
    random.shuffle(coco_data["images"])

    # Calculate the split index
    split_idx = int(len(coco_data["images"]) * train_ratio)

    # Split the data
    train_images = coco_data["images"][:split_idx]
    val_images = coco_data["images"][split_idx:]

    # Map image IDs to separate annotations for train and val sets
    train_image_ids = {img["id"] for img in train_images}
    train_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in train_image_ids]

    val_image_ids = {img["id"] for img in val_images}
    val_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in val_image_ids]

    # Create train and validation datasets
    train_data = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_data["categories"]
    }

    val_data = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco_data["categories"]
    }

    # Save the split datasets into new JSON files
    with open(DATASET_DIR / "train_annotations.json", "w") as f:
        json.dump(train_data, f)

    with open(DATASET_DIR / "val_annotations.json", "w") as f:
        json.dump(val_data, f)

    print("Dataset splitting completed!")