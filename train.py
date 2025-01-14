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

from detectron2.evaluation import COCOEvaluator


# Constants
MATERIAL = "NbSe2" #can be "WS2" or "NbSe2" or ... (other materials - later on)
TODAY_DATASET = "2025_01_13_newImages"

BASE_DIR = Path(__file__).resolve().parent / MATERIAL
TODAY_DATASET_FOLDER = BASE_DIR / "data" / TODAY_DATASET
TRAIN_JSON = "train_annotations.json"
TEST_JSON = "val_annotations.json"

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

if not (TODAY_DATASET_FOLDER / TRAIN_JSON).exists():
    raise FileNotFoundError(f"Training dataset JSON not found: {TRAIN_JSON}")
# if not (TODAY_DATASET_FOLDER / VAL_JSON).exists():
#     raise FileNotFoundError(f"Validation dataset JSON not found: {VAL_JSON}")
if not (TODAY_DATASET_FOLDER / TEST_JSON).exists():
    raise FileNotFoundError(f"Test dataset JSON not found: {TEST_JSON}")


# Dataset registration
def register_datasets():
    register_coco_instances("dataset_train", {}, TODAY_DATASET_FOLDER / TRAIN_JSON, TODAY_DATASET_FOLDER)
    # register_coco_instances("dataset_val", {}, TODAY_DATASET_FOLDER / VAL_JSON, TODAY_DATASET_FOLDER)
    register_coco_instances("dataset_test", {}, TODAY_DATASET_FOLDER / TEST_JSON, TODAY_DATASET_FOLDER)

# Model configuration
def setup_cfg():
    cfg = get_cfg()
    cfg.OUTPUT_DIR = str(dirs["configs"])
    cfg.LOG_DIR = str(dirs["logs"])
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))
    cfg.DATASETS.TRAIN = ("dataset_train",)
    # cfg.DATASETS.TEST = ()
    cfg.DATASETS.TEST = ("dataset_test",)
    cfg.TEST.EVAL_PERIOD = 20
    cfg.DATALOADER.NUM_WORKERS = 1 # was 2
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
    cfg.SOLVER.IMS_PER_BATCH = 1  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00001  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000    # 1000 iterations seems good enough for this dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # Default is 512, using 256 for this dataset.
    
    if MATERIAL == "WS2":
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # We have 3 classes for WS2: ML, BL, FL
    elif MATERIAL == "NbSe2":
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # We had 6 classes for NbSe2: 1L, 2L, 3L, 4L, 5L, >=5L; now 4 classes: 3L, 4L, 5L, >5L.
    else: print("You have chosen the unsupported material. Choose one from the list: WS2, NbSe2.") 
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
        writers.append(TensorboardXWriter(self.cfg.LOG_DIR))
        return writers
    
    @staticmethod
    def build_evaluator(cfg, dataset_name, output_folder=None):
        """
        Builds the evaluator for evaluation during training.
        """
        if output_folder is None:
            output_folder = Path(cfg.OUTPUT_DIR) / "eval"
        return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)








# Main script
if __name__ == "__main__":
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)  # Limit to 50% of GPU memory
    register_datasets()
    cfg = setup_cfg()
    # trainer = DefaultTrainer(cfg)
    trainer = TrainerWithTensorboard(cfg)
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
    raise