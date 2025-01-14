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
from pathlib import Path




# import some common detectron2 utilities

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances







# --------------------------------------------------SCRIPT section--------------------------------------------------

# Creation of detectron2 config and detectron2 DefaultPredictor
# it is required to prevent recursive spawning of processes. 
if __name__ == '__main__':
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


    # Choose one of the following lines - WS2 or NbSe2 - and comment the other
    material = "WS2" 
    # material = "NbSe2" 







    # Directory of a material, containing all data (both train and val), logs, checkpoints, configs.
    maindir = Path(__file__).resolve().parent / material

    logdir = maindir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    ckptdir = maindir / "checkpoints"
    ckptdir.mkdir(parents=True, exist_ok=True)

    cfgdir = maindir / "configs"
    cfgdir.mkdir(parents=True, exist_ok=True)

    datadir = maindir / "data"
    datadir.mkdir(parents=True, exist_ok=True)


    # Here, we are naming our training data as 'dataset_train' and the validation data as 'dataset_val'.
    register_coco_instances("dataset_train", {},    datadir / "train"   /     "labels_my-project-name_2024-12-10-04-46-10.json",  datadir / "train")
    register_coco_instances("dataset_val", {},      datadir / "val"     /     "labels_my-project-name_2024-12-10-04-47-18.json",  datadir / "val")

    # Metadata and dictionaries extraction
    train_metadata = MetadataCatalog.get("dataset_train")
    train_dataset_dicts = DatasetCatalog.get("dataset_train")
    val_metadata = MetadataCatalog.get("dataset_val")
    val_dataset_dicts = DatasetCatalog.get("dataset_val")




    # # Visualize some random samples
    # for d in random.sample(train_dataset_dicts, 3):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
    #     vis = visualizer.draw_dataset_dict(d)

    #     # Display the visualized image
    #     cv2.imshow("Predicted Image", vis.get_image()[:, :, ::-1])
    #     print(f'You may see image {d["file_name"]} from train dataset on the separate window. Press any button to close current image and see next one.')
    # # Wait for a key press for each image
    #     k = cv2.waitKey(0)  # 0 means wait indefinitely
    #     if k == 27:  # Press 'Esc' to exit
    #         break

    # # Close all OpenCV windows
    # cv2.destroyAllWindows()



    # --------------------------------------------------TRAIN section--------------------------------------------------


    cfg = get_cfg()
    cfg.OUTPUT_DIR = str(cfgdir)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 200    # 1000 iterations seems good enough for this dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # Default is 512, using 256 for this dataset.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # We have 3 classes for WS2: ML, BL, FL.
    # NOTE: this config means the number of classes, without the background. Do not use num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) #Create an instance of of DefaultTrainer with the given congiguration
    trainer.resume_or_load(resume=False) #Load a pretrained model if available (resume training) or start training from scratch if no pretrained model is available


    trainer.train() #Start the training process
    
    # # Look at training curves in tensorboard:
    # %load_ext tensorboard
    # %tensorboard --logdir output


    # import yaml
    # # Save the configuration to a config.yaml file
    # # Save the configuration to a config.yaml file
    # config_yaml_path = cfgdir / "yaml"
    # config_yaml_path.mkdir(parents=True, exist_ok=True)
    # # config_yaml_path = "/content/drive/MyDrive/ColabNotebooks/models/Detectron2_Models/config.yaml"
    # with open(config_yaml_path, 'w') as file:
    #     yaml.dump(cfg, file)








# --------------------------------------------------END section--------------------------------------------------
print(f'Program train.py has run successfully')