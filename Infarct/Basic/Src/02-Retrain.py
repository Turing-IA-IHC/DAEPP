"""
DAEPP: detection of anomalous events and prejudice for people.

This code take a model to detect infarcts and retrain it
Written by Gabriel Rojas - 2019
Copyright (c) 2019 G0 S.A.S.
Licensed under the MIT License (see LICENSE for details)
"""

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers import  Convolution2D, MaxPooling2D
from keras import backend as K
import os
from os import scandir, getcwd
from distutils.dir_util import copy_tree
import shutil

# === Configuration vars ===
# Version to validate
VERSION = 'Basic'
# Path of image folder (use slash at the end)
INPUT_PATH_TRAIN = "./" + VERSION + "/Dataset/10Data/train/"
INPUT_PATH_VAL = "./" + VERSION + "/Dataset/10Data/val/"

# Various
EPOCH_CHECK_POINT = 4
NUM_CHECK_POINT = 3
STEPS = 7000
VALIDATION_STEPS = 500
BATCH_SIZE = 4
LR = 0.1
CLASES = 2
KEEP_ONLY_LATEST = False
# Size image
WIDTH, HEIGHT = 256, 256
# === ===== ===== ===== ===

target_dir = './' + VERSION + '/Model/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

model = './' + VERSION + '/Model/model.h5'
weights = './' + VERSION + '/Model/weights.h5'
print("Loading model from:", model)

cnn = load_model(model)
cnn.load_weights(weights)

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    INPUT_PATH_TRAIN,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    INPUT_PATH_VAL,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')


cnn.compile(optimizer=optimizers.SGD(lr=LR),
              loss='binary_crossentropy',
              metrics=['acc', 'mse'])
              
cnn.summary()


modAnterior = ""
pesAnterior = ""
for i in range(5):        
    cnn.fit_generator(
        entrenamiento_generador,
        steps_per_epoch=STEPS,
        epochs=EPOCH_CHECK_POINT,
        validation_data=validacion_generador,
        validation_steps=VALIDATION_STEPS
        )
    
    print("Saving model", i)
    cnn.save(target_dir + str(i) + 'model_R.h5')
    cnn.save_weights(target_dir + str(i) + 'weights_R.h5') 
 
    if KEEP_ONLY_LATEST:
        if os.path.exists(modAnterior):
            print("Delete before", i - 1)
            os.remove(modAnterior)
            os.remove(pesAnterior)
        modAnterior = target_dir + str(i) + 'model_R.h5'
        pesAnterior = target_dir + str(i) + 'weights_R.h5'
