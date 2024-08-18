import numpy as np
import cv2

# Specify the width, height, and number of channels of the image
width = 128  # replace with actual width
height = 128  # replace with actual height
channels = 1  # For grayscale images (use 3 for RGB)

# Load RAW file
with open(r'*.raw', 'rb') as f:
    raw_data = np.fromfile(f, dtype=np.uint8)

# Reshape the data according to the known dimensions
image = raw_data.reshape((height, width, channels))

# Display the image using OpenCV
cv2.imshow('RAW Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()