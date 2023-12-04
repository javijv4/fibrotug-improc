# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 19:30:59 2023

@author: laniq
"""
import os
from matplotlib import pyplot as plt
import numpy as np
from skimage import io, img_as_uint
from src.imregistration.ITKTransform import apply_transform, imageToArray, elastix_simple_transformation, rescale_image_intensity

# Path to the masks. Results will be saved in tissue_folder
tissue_fldr = 'test_data/'
pre_mask_file = tissue_fldr + 'pre/fibrotug_mask'
post_mask_file = tissue_fldr + 'post/fibrotug_mask'

ext = '.tif'

# List all the files to warp and write. The key will be the name of the output file
# i.e. this will save the warped image in tissue_folder/pre_act.tif 
towarp = {'pre_act': tissue_fldr + 'pre/A_0.41_0.1_GEM12-01_MAX_c2_ORG',
          'pre_dsp': tissue_fldr + 'pre/A_0.41_0.1_GEM12-01_MAX_c1_ORG'}

# List all the files to save in the tissue_folder path. The key will be the name of the output file
# i.e. this will save the warped image in tissue_folder/pre_act.tif 
tosave = {'post_act': tissue_fldr + 'post/A_0.41_0.1_GEM12_S2_ET1-01-MAX_c4_ORG',
          'post_dsp': tissue_fldr + 'post/A_0.41_0.1_GEM12_S2_ET1-01-MAX_c2_ORG'}

# Read masks
pre_mask = imageToArray(pre_mask_file + ext)
post_mask = imageToArray(post_mask_file + ext)

# Elastix transform
transform_params = elastix_simple_transformation(post_mask, pre_mask, mode='affine')

# warp mask
warped_mask = apply_transform(pre_mask, transform_params)
warped_mask[np.isclose(warped_mask, 0.)] = 0
warped_mask = warped_mask > 0.5
io.imsave(tissue_fldr + 'pre_mask' + ext, warped_mask.astype(np.int8), check_contrast=False)
io.imsave(tissue_fldr + 'post_mask' + ext, post_mask.astype(np.int8), check_contrast=False)

# warp requested images
for name in towarp.keys():
    img = imageToArray(towarp[name] + ext)
    warped_img= apply_transform(img, transform_params)
    io.imsave(tissue_fldr + name + ext, img_as_uint(warped_img), check_contrast=False)

# save requested images
for name in tosave.keys():
    img = imageToArray(tosave[name] + ext)
    rescaled_img = rescale_image_intensity(img)
    io.imsave(tissue_fldr + name + ext, img_as_uint(rescaled_img), check_contrast=False)


# Plots
def remove_background(mask):
    mask = mask.astype(float)
    mask[mask==0] = np.nan
    return mask

plt.figure()
plt.imshow(remove_background(post_mask), cmap='Reds', alpha=0.8, vmin=0, vmax=1)
plt.imshow(remove_background(warped_mask), cmap='Blues', alpha=0.8, vmin=0, vmax=1)
plt.axis('off')