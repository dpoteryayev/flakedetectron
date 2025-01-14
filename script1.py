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

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"




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

# Let's play a bit with a model using normal picture
# Read the image
image = cv2.imread(r"C:/Users/dpoteryayev/1. PhD/FlakeDetectron/BackyardGardenMicroWedding-27-2199992322.jpg")

# Rescaling
scale_percent = 50  # Scale down to 50% of the original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
new_dimensions = (width, height)

# Resize the image
resized_image = cv2.resize(image, new_dimensions)
# Making window itself scalable
cv2.namedWindow("Normal image", cv2.WINDOW_NORMAL)
# Showing the image
cv2.imshow("Normal image",resized_image)
# k = cv2.waitKey(0)

# Model predicts
outputs = predictor(image)


# look at the outputs - tensors and bounding boxes.
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# Making window itself scalable
cv2.namedWindow("Predicted image", cv2.WINDOW_NORMAL)
# Showing the image
cv2.imshow("Predicted image",out.get_image()[:, :, ::-1])
k = cv2.waitKey(0)

























print(f'Program script1 has run successfully')