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
import random
from sklearn.cluster import KMeans

# img1 = nib.load(r'C:\Users\a\Desktop\3021 Industry Project\Machine Learning Project-selected (1)\Case_1_old\cbv_reg.img')
# hdr1 = nib.load(r'C:\Users\a\Desktop\3021 Industry Project\Machine Learning Project-selected (1)\Case_1_old\cbv_reg.hdr')

# print(img1)
# print(hdr1)

#CNN model
from keras import layers
from keras import models
from keras import optimizers

def read_one_file(path):
    img = cv2.imread(path)
    print(img.shape)
    print(img.size)
    print(img.dtype)

# path1 = r'C:\Users\a\Desktop\3021 Industry Project\archive\Testing\meningioma\Te-me_0010.jpg'
# read_one_file(path1)        

#read file
def load_data(path, image_size=(128,128)):
    #numpy list
    images = []
    labels = []
    for file in os.listdir(path):
        if file.endswith(".jpg") or file.endswith(".jpeg"):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)  
            img = cv2.resize(img, image_size)  # resize
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB（OpenCV默认使用BGR）
            img = img / 255.0  # normalization
            if path[-1] == 'r':
                images.append(img)
                labels.append(0)
            else:
                images.append(img)
                labels.append(1)
    return images,labels

path_positive_train = r'C:\Users\a\Desktop\3021 Industry Project\archive\Training\meningioma'
path_negative_train = r'C:\Users\a\Desktop\3021 Industry Project\archive\Training\notumor'
train_set1, y1 = load_data(path_positive_train)
train_set2, y2 = load_data(path_negative_train)
train_set = train_set1 + train_set2
label_train = y1 + y2

# 将列表转换为 NumPy 数组
train_set = np.array(train_set)
label_train = np.array(label_train)

indices = np.arange(len(train_set))
np.random.shuffle(indices)
train_features = train_set[indices]
train_labels = label_train[indices]

path_positive_test = r'C:\Users\a\Desktop\3021 Industry Project\archive\Testing\meningioma'
path_negative_test = r'C:\Users\a\Desktop\3021 Industry Project\archive\Testing\notumor'
test_set1, y3 = load_data(path_positive_test)
test_set2, y4 = load_data(path_negative_test)
test_set = test_set1 + test_set2
label_test = y3 + y4

# 将列表转换为 NumPy 数组
test_set = np.array(test_set)
label_test = np.array(label_test)

indices2 = np.arange(len(test_set))
np.random.shuffle(indices2)
test_features = test_set[indices2]
test_labels = label_test[indices2]
# print(train_set.shape)
# print(len(train_set))

#this model is to detect the brain tumor, num_class = 2, based on CNN and KMeans
def CNN_model1(input_shape):
    model = models.Sequential()
    #Convolution layer, activate function: relu, Convolution kernel: 3*3
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(128, 128, 3)))
    #max pooling layer
    model.add(layers.MaxPooling2D((2, 2)))
    #Convolution layer, activate function: relu, Convolution kernel: 2*2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #max pooling layer
    model.add(layers.MaxPooling2D((2, 2)))
    #Convolution layer, activate function: relu, Convolution kernel: 3*3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    #max pooling layer
    model.add(layers.MaxPooling2D((2, 2)))
    #Convolution layer, activate function: relu, Convolution kernel: 3*3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    #max pooling layer
    model.add(layers.MaxPooling2D((2, 2)))
    #flatten
    model.add(layers.Flatten())
    #Fully connected layer, activate function relu
    model.add(layers.Dense(512, activation='relu'))
    
    return model
    
# input_shape = (128, 128, 3)
# feature_extractor = CNN_model1(input_shape)
# features = feature_extractor.predict(train_set)

# kmeans = KMeans(n_clusters = 2, random_state = 0).fit(features)
# print("Cluster labels:", kmeans.labels_)


#this model based on autoencoder
def CNN_model2(input_shape):
    # Encoder
    encoder = models.Sequential()
    encoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    encoder.add(layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    encoder.add(layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    encoder.add(layers.MaxPooling2D((2, 2), padding='same'))

    # Decoder
    decoder = models.Sequential()
    decoder.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(16, 16, 128)))
    decoder.add(layers.UpSampling2D((2, 2)))
    decoder.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    decoder.add(layers.UpSampling2D((2, 2)))
    decoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    decoder.add(layers.UpSampling2D((2, 2)))
    decoder.add(layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

    autoencoder = models.Sequential([encoder, decoder])
    return autoencoder, encoder

# input_shape = (128, 128, 3)
# autoencoder, encoder = CNN_model2(input_shape)
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
# autoencoder.fit(train_set, train_set, epochs=10, batch_size=64, validation_split=0.2)
# encoded_imgs = encoder.predict(train_set)

from keras.applications import VGG16
from sklearn.cluster import KMeans

# VGG16
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# features = base_model.predict(train_set)

# features_reshaped = features.reshape(len(features), -1)

# # Kmeans
# kmeans = KMeans(n_clusters=2, random_state=0).fit(features_reshaped)

# print("Cluster labels:", kmeans.labels_)

# #evaluate
# from sklearn.metrics import silhouette_score

# silhouette_avg = silhouette_score(features_reshaped, kmeans.labels_)
# print("Silhouette Score:", silhouette_avg)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

class OD:
    def __init__(self):
        pass
    
    def OD_function(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

        #use VGG16
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # brain tumor detection
        return model
    
    # def test(self):
    #     test_loss, test_accuracy = supervised_CNN.evaluate(test_set, label_test)
        
    def predict(self, path, model):
        pass
    
    def mask(self, model):
        pass
    
    def segmentation(self):
        pass
    
    def obervation(self):
        pass
    
        
    
def train(self):
        X_train, X_val, y_train, y_val = train_test_split(train_set, label_train, test_size=0.2, random_state=42)
        supervised_CNN = CNN_model4()
        supervised_CNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        supervised_CNN.fit(train_set, label_train, epochs=2, batch_size=32, validation_data=(X_val, y_val))
    



def CNN_model4():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    #use VGG16
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # brain tumor detection
    return model

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
# early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# X_train, X_val, y_train, y_val = train_test_split(train_set, label_train, test_size=0.2, random_state=42)
# supervised_CNN = CNN_model4()
# supervised_CNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# supervised_CNN.fit(train_set, 
#                    label_train, 
#                    epochs=20, 
#                    batch_size=32, 
#                    validation_data=(X_val, y_val),
#                    callbacks=[early_stop])
# supervised_CNN.save('brain_tumor_cnn_model.h5')
from tensorflow.keras.models import load_model
def predict_tumor(image_path):
    model = load_model('brain_tumor_cnn_model.h5')
# test_loss, test_accuracy = supervised_CNN.evaluate(test_set, label_test)
# print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
    img = cv2.imread(r'C:\Users\a\Desktop\3021 Industry Project\archive\Testing\notumor\Te-no_0014.jpg')
    img_re = cv2.resize(img, (128,128))
    img_resized = np.array(img_re, dtype=np.float32)
    img_resized /= 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    pred = model.predict(img_resized)
    class1 = (pred > 0.5).astype(int)
    print(pred)
    return pred, class1
