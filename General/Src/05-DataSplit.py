"""
DAEPP: detection of anomalous events and prejudice for people.

This code split image in Train, Validate and Test
Written by Gabriel Rojas - 2019
Copyright (c) 2019 G0 S.A.S.
Licensed under the MIT License (see LICENSE for details)
"""

import os
from os import scandir, getcwd
from distutils.dir_util import copy_tree
import shutil

# === Configuration vars ===
# Path of image folder (use slash at the end)
INPUT_PATH = "M:/tmp/02Aumentado/"
# Path where new images goes to put (use slash at the end)
OUTPUT_PATH = "M:/tmp/03Data/"

# Percent to take for each folder
TRAIN = .8
VAL = .15
TEST = .5
# === ===== ===== ===== ===


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

# Creates base folders
if not os.path.exists(OUTPUT_PATH + "/train/"):
    os.mkdir(OUTPUT_PATH + "/train/")
if not os.path.exists(OUTPUT_PATH + "/val/"):
    os.mkdir(OUTPUT_PATH + "/val/")
if not os.path.exists(OUTPUT_PATH + "/test/"):
    os.mkdir(OUTPUT_PATH + "/test/")

# The class are indentify for each sub folder
folders = lsFolders(INPUT_PATH)

for f in folders:
    # Make final folders
    if not os.path.exists(OUTPUT_PATH + "/train/" + f):
        os.mkdir(OUTPUT_PATH + "/train/" + f)
    if not os.path.exists(OUTPUT_PATH + "/val/" + f):
        os.mkdir(OUTPUT_PATH + "/val/" + f)
    if not os.path.exists(OUTPUT_PATH + "/test/" + f):
        os.mkdir(OUTPUT_PATH + "/test/" + f)
    
    files = lsFiles(INPUT_PATH + f)

    copy = 0
    deTrain = int(len(files) * TRAIN)
    deVal = int(len(files) * VAL)
    deTest = int(len(files) * TEST)
    dest = "/train/"
    for a in files:
        if copy <= deTrain:
            dest = "/train/"
        elif copy <= deTrain + deVal:
            dest = "/val/"
        else:
            dest = "/test/"            

        shutil.copyfile(INPUT_PATH + f + "/" + a, OUTPUT_PATH + dest + f + "/" + a)
        copy = copy + 1