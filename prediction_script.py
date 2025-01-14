import torch
import os
import logging
from detectron2 import model_zoo
from detectron2.config import get_cfg
from pathlib import Path
from detectron2.data.datasets import register_coco_instances

from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger

import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.utils.visualizer import ColorMode


# Supported file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Setup directories
def setup_directories(base_path, subdirs):
    dirs = {}
    for subdir in subdirs:
        dir_path = base_path / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        dirs[subdir] = dir_path
    return dirs



# Dataset registration
def register_datasets():
    register_coco_instances("dataset_train", {}, TODAY_DATASET_FOLDER / TRAIN_JSON, TODAY_DATASET_FOLDER)
    # register_coco_instances("dataset_val", {}, dirs["data"] / "val" / VAL_JSON, dirs["data"] / "val")
    # register_coco_instances("dataset_test", {}, dirs["data"] / "test" / TEST_JSON, dirs["data"] / "test")





# Model configuration
def setup_cfg():
    cfg = get_cfg()
    cfg.OUTPUT_DIR = str(dirs["configs"])
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))
    cfg.DATASETS.TRAIN = ("dataset_train",)
    cfg.DATASETS.TEST = ()
    # cfg.DATASETS.TEST = ("dataset_test",)
    # cfg.DATALOADER.NUM_WORKERS = 1 # was 2
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
    # cfg.SOLVER.IMS_PER_BATCH = 1  # This is the real "batch size" commonly known to deep learning people
    # cfg.SOLVER.BASE_LR = 0.00001  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 1000    # 1000 iterations seems good enough for this dataset
    # cfg.SOLVER.STEPS = []        # do not decay learning rate
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # Default is 512, using 256 for this dataset.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # We have 3 classes for WS2: ML, BL, FL.
    # NOTE: this config means the number of classes, without the background. Do not use num_classes+1 here.
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    # cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    # cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0  # Clip gradient values to prevent explosion
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set a custom testing threshold
    
    return cfg



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# logger.info(f"Training on {len(train_dataset_dicts)} images")



# Main script
if __name__ == "__main__":
    MATERIAL = "NbSe2" #can be "WS2" or "NbSe2" or ... (other materials - later on)
    TODAY_DATASET = "2025_01_13_newImages"
    BASE_DIR = Path(__file__).resolve().parent / MATERIAL
    TODAY_DATASET_FOLDER = BASE_DIR / "data" / TODAY_DATASET
    TRAIN_JSON = "train_annotations.json" 
    # VAL_JSON = "val_annotations.json"
    # TEST_JSON = ""
    
    


    input_images_directory = BASE_DIR / "input_images_directory/2025_01_13"
    output_directory = BASE_DIR / "output_directory/2025_01_13_2"
    
    dirs = setup_directories(BASE_DIR, ["logs", "checkpoints", "configs", "data", input_images_directory, output_directory])
    
    
    
    
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)  # Limit to 50% of GPU memory
    register_datasets()
    train_metadata = MetadataCatalog.get("dataset_train")
    train_dataset_dicts = DatasetCatalog.get("dataset_train")
    # val_metadata = MetadataCatalog.get("dataset_val")
    # val_dataset_dicts = DatasetCatalog.get("dataset_val")
    
    
    MODEL_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" 
    MODEL_WEIGHTS = os.path.join(str(dirs["configs"]), "model_final.pth")
    cfg = setup_cfg()
    
    
    
    
    
    
    
    
    
    predictor = DefaultPredictor(cfg)
    
    # Loop over the images in the input folder
    for image_filename in os.listdir(input_images_directory):
        # Get full path of the input image
        image_path = os.path.join(input_images_directory, image_filename)
        
        # Read the image using OpenCV
        new_im = cv2.imread(image_path)
        
        # Check if the image was loaded successfully
        if new_im is None:
            print(f"Failed to load image: {image_filename}. Skipping...")
            continue
        
        # Ensure the image is in RGB format
        new_im_rgb = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)

        # Perform prediction on the RGB image
        outputs = predictor(new_im_rgb)
        
        # Check if any objects were detected
        instances = outputs["instances"]
        if len(instances) == 0:
            print(f"No objects found in image: {image_filename}. Skipping...")
            continue  # Skip this image if no objects are detected

        # Visualize the predictions
        v = Visualizer(new_im_rgb, metadata=train_metadata, scale=1.0)
        out = v.draw_instance_predictions(instances.to("cpu"))

        # Convert back to BGR for saving (OpenCV saves in BGR format)
        result_image_bgr = cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR)

        # Create the output filename with "_result" appended
        result_filename = os.path.splitext(image_filename)[0] + "_result.png"
        output_path = os.path.join(output_directory, result_filename)

        # Save the segmented image
        cv2.imwrite(output_path, result_image_bgr)
        print(f"Objects found and saved for image: {image_filename}")

    print("Segmentation completed. Only images with detected objects were saved.")
