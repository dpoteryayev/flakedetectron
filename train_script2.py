import torch
# import os
import logging
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from pathlib import Path
from detectron2.data.datasets import register_coco_instances

# Constants
MATERIAL = "WS2" #can be "WS2" or "NbSe2" or ... (other materials - later on)
BASE_DIR = Path(__file__).resolve().parent / MATERIAL
TRAIN_JSON = "labels_my-project-name_2024-12-10-04-46-10.json"
VAL_JSON = "labels_my-project-name_2024-12-10-04-47-18.json"
MODEL_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
MODEL_WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG)

# Setup directories
def setup_directories(base_path, subdirs):
    dirs = {}
    for subdir in subdirs:
        dir_path = base_path / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        dirs[subdir] = dir_path
    return dirs
dirs = setup_directories(BASE_DIR, ["logs", "checkpoints", "configs", "data"])

if not (dirs["data"] / "train" / TRAIN_JSON).exists():
    raise FileNotFoundError(f"Training dataset JSON not found: {TRAIN_JSON}")
if not (dirs["data"] / "val" / VAL_JSON).exists():
    raise FileNotFoundError(f"Validation dataset JSON not found: {VAL_JSON}")


# Dataset registration
def register_datasets():
    register_coco_instances("dataset_train", {}, dirs["data"] / "train" / TRAIN_JSON, dirs["data"] / "train")
    register_coco_instances("dataset_val", {}, dirs["data"] / "val" / VAL_JSON, dirs["data"] / "val")

# Model configuration
def setup_cfg():
    cfg = get_cfg()
    cfg.OUTPUT_DIR = str(dirs["configs"])
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))
    cfg.DATASETS.TRAIN = ("dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 200    # 1000 iterations seems good enough for this dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # Default is 512, using 256 for this dataset.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # We have 3 classes for WS2: ML, BL, FL.
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # if torch.cuda.is_available():
    #     cfg.MODEL.DEVICE = "cuda"
    # else:
    #     print("Warning: GPU not available. Using CPU.")

    # NOTE: this config means the number of classes, without the background. Do not use num_classes+1 here.
    return cfg



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# logger.info(f"Training on {len(train_dataset_dicts)} images")

# Main script
if __name__ == "__main__":
    register_datasets()
    cfg = setup_cfg()
    trainer = DefaultTrainer(cfg)
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
    raise
