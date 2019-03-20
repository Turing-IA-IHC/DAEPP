"""
DAEPP: detection of anomalous events and prejudice for people.

This code tests specific cases in infarct detection
Written by Gabriel Rojas - 2019
Copyright (c) 2019 G0 S.A.S.
Licensed under the MIT License (see LICENSE for details)
"""

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import os
from os import scandir, getcwd
from distutils.dir_util import copy_tree
import shutil

# === Configuration vars ===
# Version to validate
VERSION = 'Basic' # Basic, Skeleton
# Path of image folder (use slash at the end)
INPUT_PATH = "./" + VERSION + "/Dataset/10Data/test/"
# Size image
WIDTH, HEIGHT = 256, 256
# === ===== ===== ===== ===


model = './' + VERSION + '/Model/model.h5'
weights = './' + VERSION + '/Model/weights.h5'
print("Loading model from:", model)

cnn = load_model(model)
cnn.load_weights(weights)

def predict(file):
    x = load_img(file, target_size=(WIDTH, HEIGHT))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    clase = 'Infarto' if answer == 0 else 'NoInfarto'
    print("Detected class:", clase, "\tLast layer:", result)

# Cases
INPUT_PATH = "./Basic/Dataset/10Data/test/Infarto/Augmented__93_4218078.png" # Infarto
predict(INPUT_PATH)
INPUT_PATH = "./Basic/Dataset/10Data/test/NoInfarto/Augmented__744_2160416.png" # NoInfarto
predict(INPUT_PATH)
INPUT_PATH = "M:/IA/Infarto/img/Cortadas/4.jpg.Cropped.png" # Infarto
predict(INPUT_PATH)
INPUT_PATH = "M:/IA/Infarto/img/Cortadas/5.jpg.Cropped.png" # No Infarto
predict(INPUT_PATH)

