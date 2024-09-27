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

from sklearn.mixture import GaussianMixture
import numpy as np
import cv2
import glob
import pickle
from skimage import measure

def load_and_extract_features(image_folder):
    images = []
    for img_path in glob.glob(image_folder + "/*.jpg"):
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (128, 128))
        img_normalized = img_resized / 255.0
        images.append(img_normalized)
    
    features = [img.reshape(-1, 3) for img in images]
    features = np.vstack(features)
    return features

def train_gmm(features, n_components=2):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(features)
    return gmm

def detect_anomalies_gmm(gmm, img):
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    features = img_normalized.reshape(-1, 3)
    
    probabilities = gmm.score_samples(features)
    anomaly_map = probabilities.reshape(128, 128)
    return anomaly_map




#decesion boundary

def decision_boundary():
    image_folder = r"C:\Users\a\Desktop\3021 Industry Project\archive\Training\notumor"
    features = load_and_extract_features(image_folder)

# gmm = train_gmm(features, n_components=2)

# with open('gmm_model.pkl', 'wb') as file:
#     pickle.dump(gmm, file)

    with open('gmm_model.pkl', 'rb') as file:
        loaded_gmm = pickle.load(file)
    
    test_image_path = r"C:\Users\a\Desktop\3021 Industry Project\archive\Testing\meningioma\Te-me_0010.jpg"
    test_img = cv2.imread(test_image_path)

    anomaly_map = detect_anomalies_gmm(loaded_gmm, test_img)

#visualize
    threshold_value = np.percentile(anomaly_map, 5)
    _, thresholded_map = cv2.threshold(anomaly_map, threshold_value, 1, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours((thresholded_map * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    uniform_contours = []

    max_variance = 0.1  
    min_area = 100 

    for contour in contours:
        mask = np.zeros_like(anomaly_map, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
    
        region_temps = anomaly_map[mask == 1]
    
        variance = np.var(region_temps)
    
        if variance < max_variance:
            uniform_contours.append(contour)

    plt.imshow(anomaly_map, cmap='hot')

    for contour in uniform_contours:
        x, y, w, h = cv2.boundingRect(contour)
    
        area = w * h
        if area >= min_area:
        # 绘制矩形框
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='cyan', facecolor='none')
            plt.gca().add_patch(rect)

    plt.title("Low Temperature, Uniform Tumor Regions with Bounding Boxes")
    plt.colorbar()
    plt.show()



