# --------------------------------------------------IMPORT section--------------------------------------------------

import torch, detectron2

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# !nvcc --version
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    print(f"CUDA Version: {cuda_version}")
else:
    print("CUDA is not available.")

from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random





# import some common detectron2 utilities

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog








# --------------------------------------------------SCRIPT section--------------------------------------------------

# Creation of detectron2 config and detectron2 DefaultPredictor

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo.  https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# Import the necessary function to register datasets in the COCO format. 
# Let us register both the training and validation datasets. 
# Please note that we are working with training (and validation) data that is is the coco format where we have a 
#   single JSON file that describes all the annotations from all training images.

# Here, we are naming our training data as 'dataset_train' and the validation data as 'dataset_val'.
from detectron2.data.datasets import register_coco_instances
register_coco_instances("dataset_train", {}, "WS2/data/train/labels_my-project-name_2024-12-10-04-46-10.json", "WS2/data/train")
register_coco_instances("dataset_val", {}, "WS2/data/val/labels_my-project-name_2024-12-10-04-47-18.json", "WS2/data/val")

# Metadata and dictionaries extraction
train_metadata = MetadataCatalog.get("dataset_train")
train_dataset_dicts = DatasetCatalog.get("dataset_train")
val_metadata = MetadataCatalog.get("dataset_val")
val_dataset_dicts = DatasetCatalog.get("dataset_val")


