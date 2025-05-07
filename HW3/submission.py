# import mrcnn.model as modellib
from mrcnn import model2 as modellib
import train as nucleus
import os
import sys
import tensorflow as tf
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


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
weights_path = model.find_last()
# weights_path = "mask_rcnn_nucleus_0038.h5"
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

print(model.keras_model.count_params())

DATASET_DIR = "./data/test_release/"
MAP_DIR = "./data/test_image_name_to_ids.json"
dataset = nucleus.NucleusDataset()
dataset.load_nucleus(DATASET_DIR, "test")
dataset.prepare()

nucleus.detect(model, DATASET_DIR, MAP_DIR, "test")
