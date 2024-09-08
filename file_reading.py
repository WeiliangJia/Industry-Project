import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import cv2
import struct
import pyreadstat



def read_hdr():
    img1 = nib.load(r'C:\Users\a\Desktop\3021 Industry Project\Machine Learning Project-selected (1)\Case_1_old\cbv_reg.hdr')
    hdr1 = nib.load(r'C:\Users\a\Desktop\3021 Industry Project\github folder\Industry-Project\image file\cbv.hdr')
    print(img1)
    data = img1.get_fdata()
    slice_index = data.shape[2] // 2  # Middle slice, adjust as needed
    slice_data = data[:, :, slice_index]

    # Plot the slice
    plt.imshow(slice_data, cmap='gray')
    plt.title('CBV Image Slice')
    plt.axis('off')
    plt.show()
        
#?error
def read_raw():
    depth = 40
    w = 128
    h = 128
    num_gradients = 31
    file_path = r'C:\Users\a\Desktop\3021 Industry Project\github folder\Industry-Project\image file\DTI_img1.raw'
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint16)
        data = data.reshape((num_gradients, depth, h, w))
    slice_index = 20
    gradient_index = 0
    plt.imshow(data[gradient_index, slice_index], cmap='gray')
    plt.title(f'Slice {slice_index}')
    plt.show()
    
def read_SAV():
    # Load the .sav file
    file_path = r'C:\Users\a\Desktop\3021 Industry Project\github folder\Industry-Project\image file\ecc_dti.sav'
    df, meta = pyreadstat.read_sav(file_path)
    print(df.head())


    
