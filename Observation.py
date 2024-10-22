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


#observation
def observation_tumour(masked_area):
    #先看轮廓
    contours, _ = cv2.findContours(masked_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        #最大的轮廓是肿瘤
        largest_contour = max(contours, key=cv2.contourArea)        
        x, y, w, h = cv2.boundingRect(largest_contour)

        #返回坐标大小，可有可不有
        print(f"Tumor Coordinates (X, Y): ({x}, {y})")
        print(f"Tumor Size (Width, Height): ({w}, {h})")

        #X轴
        cv2.line(masked_area, (0, y), (masked_area.shape[1], y), (0, 255, 0), 1)
        
        #Y轴
        cv2.line(masked_area, (x, 0), (x, masked_area.shape[0]), (0, 255, 0), 1)
        return masked_area
    else:
        #没有轮廓就返回原版图片
        return masked_area
