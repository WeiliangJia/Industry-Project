import tensorflow as tf
# print(tf.__version__)
from tensorflow import keras
# print(keras.__version__)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import nibabel as nib
import cv2
import mimetypes

img1 = nib.load(r'Industry-Project\image file\cbv.img')
hdr1 = nib.load(r'Industry-Project\image file\cbv.hdr')

# print(img1)
# print(hdr1)

# Specify the width, height, and number of channels of the image
width = 128  # replace with actual width
height = 128  # replace with actual height
channels = 1  # For grayscale images (use 3 for RGB)

# Load RAW file
with open(r'*.raw', 'rb') as f:
    raw_data = np.fromfile(f, dtype=np.uint8)

# Reshape the data according to the known dimensions
# image = raw_data.reshape((height, width, channels))



#main model
from keras import layers
from keras import models
from keras import optimizers


#this model is to detect the brain tumor, num_class = 2
def CNN_model1(input_shape=(128, 128, 35), num_classes=2):
    model = models.Sequential()
    # Convolution layer, activate function: relu, Convolution kernel: 3*3
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # Max pooling layer
    model.add(layers.MaxPooling2D((2, 2)))
    # Convolution layer, activate function: relu, Convolution kernel: 3*3
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # Max pooling layer
    model.add(layers.MaxPooling2D((2, 2)))
    # Convolution layer, activate function: relu, Convolution kernel: 3*3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # Max pooling layer
    model.add(layers.MaxPooling2D((2, 2)))
    # Convolution layer, activate function: relu, Convolution kernel: 3*3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # Max pooling layer
    model.add(layers.MaxPooling2D((2, 2)))
    
    # YOLO specific layers
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D((num_classes + 5) * 3, (1, 1)))  # (bbox_x, bbox_y, bbox_w, bbox_h, obj_conf, class_probs)
    grid_size = model.output_shape[1]  # Assuming a square grid
    model.add(layers.Reshape((grid_size, grid_size, 3, num_classes + 5)))

    # Compile with a YOLO loss function (you need to implement or use an existing YOLO loss)
    model.compile(loss='yolo_loss_function', optimizer=optimizers.Adam(lr=1e-4))

    return model



