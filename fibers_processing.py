import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, transform, morphology, filters
import cv2
from improcessing.fibers import prepare_image, generate_initial_mask

filename = 'test_data/post/A_0.41_0.1_GEM12_S2_ET1-04-MAX_c3_ORG.tif'
resize = False
v_avg_threshold = 100
view_masks = 0

windowd = 3  # Pixel window where the dispersion will be averaged
window = 1  # Pixel window where the quantities will be averaged
windowf = 3  # Pixel window where the fiber field will be averaged
dh = 1  # Pixel window to smooth fiber field



# if not os.path.exists(filename):
#     print(filename)

im_og = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

#Threshold Image
im = exposure.equalize_adapthist(im_og, kernel_size=51)
thresh1 = filters.threshold_local(im, block_size=51)
mask1 = im > thresh1
mask1 = morphology.binary_closing(mask1, morphology.disk(1))
mask1 = morphology.binary_opening(mask1, morphology.disk(2))
# im = transform.rescale(im, 2)
# plt.imshow(im)
# plt.show()
# C, a = prepare_image(im, v_avg_threshold, view_masks)
# Create mask
# Options are mean, local otsu (param radius), and local (param block_size)
mask = generate_initial_mask(im_og, 'mean', radius = 41, block_size = 101, remove_size=11)
# mask = transform.rescale(mask, 0.5)
# mask = morphology.binary_dilation(mask, footprint=morphology.disk(2))

plt.imshow(mask1, cmap='binary_r')
# plt.imshow(mask, vmin=0, vmax=1)
plt.show()
#Save images for manual editing
io.imsave(os.path.dirname(filename) + '/fibers_mask_init.tif',
          mask1.astype(np.int8), check_contrast=False)

# io.imsave(os.path.dirname(filename) + '/fibers_image.tif',
#           a.astype(np.int8), check_contrast=False)

# #############
# #Load manually edited mask

# mask = io.imread(tiffName_Optimized, as_gray = True)
# originalMask = io.imread(tiffName, as_gray = True)