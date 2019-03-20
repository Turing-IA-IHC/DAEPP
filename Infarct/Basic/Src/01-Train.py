"""
DAEPP: detection of anomalous events and prejudice for people.

This code make a neural network to detec infarcts
Written by Gabriel Rojas - 2019
Copyright (c) 2019 G0 S.A.S.
Licensed under the MIT License (see LICENSE for details)
"""

import sys
import os
import tensorflow
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers import  Convolution2D, MaxPooling2D
from keras import backend as K

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
LR = 0.2
CLASES = 2
KEEP_ONLY_LATEST = True
# Size image
WIDTH, HEIGHT = 256, 256
# === ===== ===== ===== ===


target_dir = './' + VERSION + '/Model/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

K.clear_session()

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

cnn = Sequential()

cnn.add(Convolution2D(64, (3 ,3), padding ="same", input_shape=(256, 256, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

cnn.add(Convolution2D(128, kernel_size=(3, 3), activation='relu'))
cnn.add(Convolution2D(256, kernel_size=(3, 3), activation='relu'))
cnn.add(Convolution2D(512, kernel_size=(3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Convolution2D(1024, kernel_size=(3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Convolution2D(512, kernel_size=(3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Dropout(0.1))
cnn.add(Flatten())

for _ in range(10):
    cnn.add(Dense(128, activation='relu'))

cnn.add(Dropout(0.5))
cnn.add(Dense(CLASES, activation='softmax'))

cnn.compile(optimizer=optimizers.SGD(lr=LR),
              loss='binary_crossentropy',
              metrics=['acc', 'mse'])

#cnn.summary()

"""
# Para usar tensorboard
import datetime
import keras
NAME = "Cascada_{}".format(datetime.datetime.now().isoformat(timespec='seconds')).replace(':', '-')

tensorboard = keras.callbacks.TensorBoard(
    log_dir="./logs/{}".format(NAME),
    histogram_freq=0,
    write_graph=True,
    write_images=True)
"""

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
    cnn.save(target_dir + str(i) + 'model.h5')
    cnn.save_weights(target_dir + str(i) + 'weights.h5') 
 
    if KEEP_ONLY_LATEST:
        if os.path.exists(modAnterior):
            print("Delete before", i - 1)
            os.remove(modAnterior)
            os.remove(pesAnterior)
        modAnterior = target_dir + str(i) + 'model.h5'
        pesAnterior = target_dir + str(i) + 'weights.h5'