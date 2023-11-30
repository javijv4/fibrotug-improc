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

tissue_fldr = 'test_data/'
pre_fldr = tissue_fldr + 'pre/'
post_fldr = tissue_fldr + 'post/'

pre_mask_file = 'fibrotug_mask'
post_mask_file = 'fibrotug_mask'

pre_dsp_file = 'A_0.41_0.1_GEM12-01_MAX_c1_ORG'
post_dsp_file = 'A_0.41_0.1_GEM12_S2_ET1-01-MAX_c2_ORG'

pre_act_file =  'A_0.41_0.1_GEM12-01_MAX_c2_ORG'
post_act_file = 'A_0.41_0.1_GEM12_S2_ET1-01-MAX_c4_ORG'

# Read masks
pre_mask = imageToArray(pre_fldr + pre_mask_file + '.tif')
post_mask = imageToArray(post_fldr + post_mask_file + '.tif')

# Elastix transform
transform_params = elastix_simple_transformation(post_mask, pre_mask, mode='affine')

# warp images
warped_mask = apply_transform(pre_mask, transform_params)
warped_mask[np.isclose(warped_mask, 0.)] = 0
warped_mask = warped_mask > 0.5

pre_dsp = imageToArray(pre_fldr + pre_dsp_file + '.tif')
pre_act = imageToArray(pre_fldr + pre_act_file + '.tif')

warped_dsp = apply_transform(pre_dsp, transform_params)
warped_act = apply_transform(pre_act, transform_params)

post_dsp = rescale_image_intensity(imageToArray(post_fldr + post_dsp_file + '.tif'))
post_act = rescale_image_intensity(imageToArray(post_fldr + post_act_file + '.tif'))

io.imsave(tissue_fldr + 'pre_dsp.tif', img_as_uint(warped_dsp), check_contrast=False)
io.imsave(tissue_fldr + 'pre_act.tif', img_as_uint(warped_act), check_contrast=False)
io.imsave(tissue_fldr + 'pre_mask.tif', warped_mask.astype(np.int8), check_contrast=False)

io.imsave(tissue_fldr + 'post_dsp.tif', img_as_uint(post_dsp), check_contrast=False)
io.imsave(tissue_fldr + 'post_act.tif', img_as_uint(post_act), check_contrast=False)
io.imsave(tissue_fldr + 'post_mask.tif', post_mask.astype(np.int8), check_contrast=False)

# Plots
import matplotlib.pyplot as plt
if not os.path.exists(tissue_fldr + 'plots/'): os.mkdir(tissue_fldr + 'plots/')

def plot_transform_results(pre_img, post_img, warped_img):
    # Create a new figure
    plt.figure(figsize=(6, 5))

    # Plot the first image (imageArrays[0])
    plt.subplot(131)  # 1 row, 3 columns, first subplot
    plt.imshow(pre_img, cmap='gray')
    plt.axis('off')
    plt.title('Pre')

    # Plot the second image (imageArrays[1])
    plt.subplot(132)  # 1 row, 3 columns, second subplot
    plt.imshow(post_img, cmap='gray')
    plt.axis('off')
    plt.title('Post')

    # Plot the third image (loaded from image)
    plt.subplot(133)  # 1 row, 3 columns, third subplot
    plt.imshow(warped_img, cmap='gray')
    plt.axis('off')
    plt.title('Warped')

    plt.tight_layout()  # Ensures proper spacing between subplots

plot_transform_results(pre_mask, post_mask, warped_mask)
plt.savefig(tissue_fldr+'plots/mask_transform.png', bbox_inches='tight', dpi=180, pad_inches = 0)
plot_transform_results(pre_dsp, post_dsp, warped_dsp)
plt.savefig(tissue_fldr+'plots/dsp_transform.png', bbox_inches='tight', dpi=180, pad_inches = 0)
plot_transform_results(pre_act, post_act, warped_act)
plt.savefig(tissue_fldr+'plots/act_transform.png', bbox_inches='tight', dpi=180, pad_inches = 0)

def remove_background(mask):
    mask = mask.astype(float)
    mask[mask==0] = np.nan
    return mask

plt.figure()
plt.imshow(remove_background(post_mask), cmap='Reds', alpha=0.8, vmin=0, vmax=1)
plt.imshow(remove_background(warped_mask), cmap='Blues', alpha=0.8, vmin=0, vmax=1)
plt.axis('off')
plt.savefig(tissue_fldr + 'plots/mask_comp_affine.png', bbox_inches='tight', dpi=180)