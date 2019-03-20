"""
DAEPP: detection of anomalous events and prejudice for people.

This code allow to increse cuantity of image to process
Written by Gabriel Rojas - 2019
Copyright (c) 2019 G0 S.A.S.
Licensed under the MIT License (see LICENSE for details)
"""

# Code based in: http://sigmoidal.ai/reduzindo-overfitting-com-data-augmentation/

import numpy as np
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

# === Configuration vars ===
# Path of image folder (use slash at the end)
INPUT_PATH = "M:/tmp/01Cortadas/"
# Path where new images goes to put (use slash at the end)
OUTPUT_PATH = "M:/tmp/02Aumentado/"
 
# Quantity of images to generate concurrently
BATCH = 2
# Quantity of images to generate per each
QUANTITY = 10
# === ===== ===== ===== ===

imgAug = ImageDataGenerator(rotation_range=45, width_shift_range=0.1,
                            height_shift_range=0.1, zoom_range=0.25,
                            fill_mode='nearest', horizontal_flip=True)

imgGen = imgAug.flow_from_directory(INPUT_PATH, save_to_dir=OUTPUT_PATH,
                     save_format='png', save_prefix='Augmented_',
                     batch_size=BATCH, class_mode='categorical')

qty = len(imgGen.filenames)
print('Total images of source:', qty)

counter = 0
for (i, newImage) in enumerate(imgGen):
    counter += 1
    # Limits Quantity to generate
    if counter == qty * QUANTITY / BATCH:
        break