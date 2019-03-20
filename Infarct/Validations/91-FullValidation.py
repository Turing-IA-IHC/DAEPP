"""
DAEPP: detection of anomalous events and prejudice for people.

This code tests the success rate in infarct detection
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


def lsFiles(ruta = getcwd()):
    """
    Returns files names in a folder
    """
    files = [arch.name for arch in scandir(ruta) if arch.is_file()]
    return files

def lsFolders(ruta = getcwd()):
    """
    Returns folders names in a folder parent
    """
    folders = [arch.name for arch in scandir(ruta) if arch.is_file() == False]
    return folders

def predict(file):
    """
    Returns values predicted
    """
    x = load_img(file, target_size=(WIDTH, HEIGHT))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    clase = 'Infarto' if answer == 0 else 'NoInfarto'
    return clase

folders = lsFolders(INPUT_PATH)

generalSuccess = 0
generalCases = 0

print("\n======= ======== ========")
for f in folders:
    files = lsFiles(INPUT_PATH + f)
    clase = f.replace(INPUT_PATH, '')
    print("Class: ", clase)
    indivSuccess = 0
    indivCases = 0
    for a in files:
        p = predict(INPUT_PATH + f + "/" + a)
        if p == clase:
            indivSuccess = indivSuccess + 1
        indivCases = indivCases + 1

    print("\tCases", indivCases, "Success", indivSuccess, "Rate", indivSuccess/indivCases)
    
    generalSuccess = generalSuccess + indivSuccess
    generalCases = generalCases + indivCases

print("Totals: ")
print("\tCases", generalCases, "Success", generalSuccess, "Rate", generalSuccess/generalCases)
print("======= ======== ========")
