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

def yolo_loss_function(y_true, y_pred, num_classes=2, lambda_coord=5, lambda_noobj=0.5):
    """
    Args:
    y_true: Ground truth tensor of shape (batch_size, grid_size, grid_size, 3, 5 + num_classes)
            Each box has 5 values: (x, y, w, h, object_confidence) + num_classes for class probabilities.
    y_pred: Predicted tensor of the same shape as y_true.
    num_classes: The number of object classes (in this case, 2 for tumor and non-tumor).
    lambda_coord: Weighting factor for the coordinate loss (to prioritize localization accuracy).
    lambda_noobj: Weighting factor for the no-object confidence loss (to suppress false positives).
    """
    # Split true and predicted values into components
    pred_box = y_pred[..., :4]  # (x, y, w, h)
    pred_confidence = y_pred[..., 4]  # object confidence
    pred_class_probs = y_pred[..., 5:]  # class probabilities
    
    true_box = y_true[..., :4]  # (x, y, w, h)
    true_confidence = y_true[..., 4]  # object confidence
    true_class_probs = y_true[..., 5:]  # class probabilities
    
    # --- 1. Localization Loss (Bounding Box Loss) ---
    # Coordinates (x, y) and dimensions (w, h) MSE Loss
    coord_loss = tf.reduce_sum(tf.square(true_box - pred_box), axis=-1)
    coord_loss = lambda_coord * tf.reduce_sum(true_confidence * coord_loss)  # Weighting only where object exists

    # --- 2. Objectness Loss ---
    # Binary cross-entropy between true and predicted object confidence
    objectness_loss = tf.keras.losses.binary_crossentropy(true_confidence, pred_confidence)
    objectness_loss = tf.reduce_sum(objectness_loss)

    # No-object loss (penalize high confidence in areas with no objects)
    no_object_loss = lambda_noobj * tf.reduce_sum((1 - true_confidence) * tf.square(pred_confidence))

    # --- 3. Classification Loss ---
    # Categorical cross-entropy for class probabilities
    class_loss = tf.keras.losses.categorical_crossentropy(true_class_probs, pred_class_probs)
    class_loss = tf.reduce_sum(true_confidence * class_loss)  # Only compute where objects exist

    # Total loss
    total_loss = coord_loss + objectness_loss + no_object_loss + class_loss

    return total_loss



