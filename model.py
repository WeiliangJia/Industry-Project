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

img1 = nib.load(r'C:\Users\a\Desktop\3021 Industry Project\github folder\Industry-Project\image file\cbv.img')
hdr1 = nib.load(r'C:\Users\a\Desktop\3021 Industry Project\github folder\Industry-Project\image file\cbv.hdr')

print(img1)
print(hdr1)



