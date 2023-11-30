import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_uint
import cv2
from improcessing.fibers import prepare_image, generate_initial_mask

filename = 'test_data/post/A_0.41_0.1_GEM12_S2_ET1-01-MAX_c3_ORG.tif'
filename = 'test_data/pre/A_0.41_0.1_GEM12-01_MAX_c2_ORG.tif'
tiffName = './data/tiffMask.tif'
tiffName_Optimized = './data/tiffMask_updated.tif'
originalName = './data/originalImage.tif'
resize = False
v_avg_threshold = 100
view_masks = 0

windowd = 3  # Pixel window where the dispersion will be averaged
window = 1  # Pixel window where the quantities will be averaged
windowf = 3  # Pixel window where the fiber field will be averaged
dh = 1  # Pixel window to smooth fiber field

if not cv2.os.path.exists(filename):
    print(filename)
    
im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
# plt.imshow(im)

imlength = max(im.shape)
if resize:
    im = cv2.resize(im, 1000/imlength)
else:
    windowd = round(windowd * imlength / 1000)
    window = round(window * imlength / 1000)
    windowf = round(windowf * imlength / 1000)
    dh = round(dh * imlength / 1000)

#Threshold Image
C, a = prepare_image(im, v_avg_threshold, view_masks)

#Create mask
# Options are mean, local otsu (param radius), and local (param block_size)
mask = generate_initial_mask(C, 'local', block_size = 201, remove_size=3)

#Save images for manual editing
io.imsave(os.path.dirname(filename) + '/fibers_mask_init.tif', 
          mask.astype(np.int8), check_contrast=False)

io.imsave(os.path.dirname(filename) + '/fibers_image.tif', 
          a.astype(np.int8), check_contrast=False)

#############
#Load manually edited mask 

mask = io.imread(tiffName_Optimized, as_gray = True)
originalMask = io.imread(tiffName, as_gray = True)