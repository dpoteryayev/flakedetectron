import torch
# import os
import logging
from detectron2 import model_zoo
# from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from pathlib import Path
from detectron2.data.datasets import register_coco_instances

from detectron2.engine import DefaultTrainer, hooks
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import TensorboardXWriter



# Constants
MATERIAL = "WS2" #can be "WS2" or "NbSe2" or ... (other materials - later on)
BASE_DIR = Path(__file__).resolve().parent / MATERIAL
TRAIN_JSON = "labels_my-project-name_2024-12-10-04-46-10.json"
VAL_JSON = "labels_my-project-name_2024-12-10-04-47-18.json"
TEST_JSON = ""
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
# if not (dirs["data"] / "val" / VAL_JSON).exists():
#     raise FileNotFoundError(f"Validation dataset JSON not found: {VAL_JSON}")
# if not (dirs["data"] / "test" / TEST_JSON).exists():
#     raise FileNotFoundError(f"Test dataset JSON not found: {TEST_JSON}")


# Dataset registration
def register_datasets():
    register_coco_instances("dataset_train", {}, dirs["data"] / "train" / TRAIN_JSON, dirs["data"] / "train")
    register_coco_instances("dataset_val", {}, dirs["data"] / "val" / VAL_JSON, dirs["data"] / "val")
    # register_coco_instances("dataset_test", {}, dirs["data"] / "test" / TEST_JSON, dirs["data"] / "test")

# Model configuration
def setup_cfg():
    cfg = get_cfg()
    cfg.OUTPUT_DIR = str(dirs["configs"])
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))
    cfg.DATASETS.TRAIN = ("dataset_train",)
    cfg.DATASETS.TEST = ()
    # cfg.DATASETS.TEST = ("dataset_test",)
    cfg.DATALOADER.NUM_WORKERS = 1 # was 2
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
    cfg.SOLVER.IMS_PER_BATCH = 1  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00001  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000    # 1000 iterations seems good enough for this dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # Default is 512, using 256 for this dataset.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # We have 3 classes for WS2: ML, BL, FL.
    # NOTE: this config means the number of classes, without the background. Do not use num_classes+1 here.
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0  # Clip gradient values to prevent explosion
    return cfg



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# logger.info(f"Training on {len(train_dataset_dicts)} images")







# Custom Trainer with TensorBoard support
class TrainerWithTensorboard(DefaultTrainer):
    def build_writers(self):
        # Build default writers (JSON and log file) and add TensorBoard
        writers = super().build_writers()
        writers.append(TensorboardXWriter(self.cfg.OUTPUT_DIR))
        return writers








# Main script
if __name__ == "__main__":
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)  # Limit to 50% of GPU memory
    register_datasets()
    cfg = setup_cfg()
    # trainer = DefaultTrainer(cfg)
    trainer = TrainerWithTensorboard(cfg)
    # try:
    #     trainer.train()
    # except Exception as e:
    #     logger.error(f"Training failed: {e}")
    # raise