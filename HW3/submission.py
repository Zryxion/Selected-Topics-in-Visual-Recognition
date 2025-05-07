import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
# import mrcnn.model as modellib
from mrcnn import model2 as modellib
from mrcnn.model import log

import train as nucleus

# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0


# Inference Configuration
config = nucleus.NucleusInferenceConfig()
# config.display()
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)
# weights_path = model.find_last()
weights_path = "/mnt/HDD3/home/owen/Homework/Deep_Learning/logs/nucleus20250505T1341/mask_rcnn_nucleus_0038.h5"
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

print(model.keras_model.count_params())

DATASET_DIR = "/mnt/HDD3/home/owen/Homework/Deep_Learning/HW3/Mask_RCNN/data/test_release/"
MAP_DIR = "/mnt/HDD3/home/owen/Homework/Deep_Learning/HW3/Mask_RCNN/data/test_image_name_to_ids.json"
dataset = nucleus.NucleusDataset()
dataset.load_nucleus(DATASET_DIR, "test")
dataset.prepare()

nucleus.detect(model, DATASET_DIR, MAP_DIR, "test")