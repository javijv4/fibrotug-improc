#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 08:54:46 2024z

@author: Javiera Jilberto Vallejos
"""

import os
import numpy as np
from FtugTissue import FtugTissue, DSPProtocolTissue, find_images
import matplotlib.pyplot as plt
import meshio as io
from skimage import io as skio

path = '/home/jilberto/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/fibroTUG/DSP/Tissues/dataset2_2/'
sim_path = '/home/jilberto/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/fibroTUG/DSP/Simulations/3PM/dataset2_2/'
samples = os.listdir(path)
samples = [sample for sample in samples if os.path.isdir(path + sample)]
samples = sorted(samples)



meshsize = 9
pixel_size = 0.390*1e-3  #mm

#%%
samples = ['gem02', 'gem03', 'gem04', 'gem05', 'gem08', 'gem10']
dsp_images = []
dsp_raw_images = []
tissue_masks = []
for sample in samples:
    tissue_fldr = f'{path}/{sample}/'
    sims_fldr = f'{sim_path}/{sample}/'

    png_dump = tissue_fldr + 'png_dump/'
    exp_fldr = tissue_fldr + 'exp/'
    mesh_fldr = sims_fldr + 'mesh/'
    data_fldr = sims_fldr + 'data/'

    pre_fldr = tissue_fldr + 'pre/'
    pre_images = find_images(pre_fldr)


    post_fldr = tissue_fldr + 'post/'
    post_images = find_images(post_fldr)


    # Creating folders
    if not os.path.exists(png_dump):
        os.makedirs(png_dump)
    if not os.path.exists(mesh_fldr):
        os.makedirs(mesh_fldr)
    if not os.path.exists(data_fldr):
        os.makedirs(data_fldr)
    if not os.path.exists(exp_fldr):
        os.makedirs(exp_fldr)

    pre_tissue = FtugTissue(pre_fldr, pre_images)
    post_tissue = FtugTissue(post_fldr, post_images)
    dsp_raw_images.append(pre_tissue.dsp_image)


    dspexp = DSPProtocolTissue(tissue_fldr, pre_tissue, post_tissue, 'pre', out_fldr=sims_fldr, flip180=True)
    tissue_masks.append(pre_tissue.tissue_mask)

    dspexp.register_to_fixed('pre', mode='affine')

    dsp_images.append(dspexp.pre_images['dsp'])
#%%
from skimage import exposure
from skimage import filters

# Apply histogram equalization to each DSP image
images = []
for idx, img in enumerate(dsp_raw_images):
    img_eq = exposure.equalize_adapthist(img)
    img_raw_eq = exposure.equalize_adapthist(dsp_raw_images[idx])
    # img_eq = img.copy()

    thresh = filters.threshold_multiotsu(img_eq, classes=3)  # Get the second threshold value for binary segmentation
    binary = img_eq > thresh[1]
    img_thresh = img_eq * binary  # Apply the threshold to the image

    images.append(img_thresh)

    # Plot histogram and threshold value
    plt.figure(figsize=(5, 3))
    plt.hist(img_eq.ravel(), bins=256, color='gray', alpha=0.7)
    plt.axvline(thresh[0], color='red', linestyle='--')
    plt.axvline(thresh[1], color='red', linestyle='--')
    plt.title(f'Histogram for Sample {samples[idx]}')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    # plt.show()
#%%
    
fig, axes = plt.subplots(2, len(images), figsize=(4 * len(images), 8), squeeze=False)

for i, img in enumerate(images):
    # Calculate center and crop 100x100 window from the original (pre-thresholded) image
    orig_img = dsp_images[i]
    h, w = orig_img.shape
    center_y, center_x = h // 2, w // 2
    half = 50
    zoom_orig = orig_img[center_y - half:center_y + half, center_x - half:center_x + half]

    axes[0, i].imshow(zoom_orig, cmap='gray')
    axes[0, i].set_title(f'Sample {samples[i]} - Original Center 100x100')
    axes[0, i].axis('off')

    # Calculate center and crop 100x100 window
    h, w = img.shape
    center_y, center_x = h // 2, w // 2
    half = 50
    zoom = img[center_y - half:center_y + half, center_x - half:center_x + half]

    axes[1, i].imshow(zoom, cmap='gray')
    axes[1, i].set_title(f'Sample {samples[i]} - Center 100x100')
    axes[1, i].axis('off')


plt.tight_layout()
# plt.show()
plt.savefig('dsp.png', bbox_inches='tight', dpi=300)

#%%
from skimage import filters, morphology, util, exposure

disk = morphology.disk(21)

for idx, (img, mask) in enumerate(zip(images, tissue_masks)):
    img = img > 0.25
    gauss_img = filters.gaussian(img, sigma=10.0, preserve_range=True)
    gauss_img = (gauss_img - gauss_img[mask==1].min()) / (gauss_img[mask==1].max() - gauss_img[mask==1].min())  # Normalize to [0, 1]
    gauss_img[mask==0] = 0  # Apply mask to the Gaussian image

    sum_img = filters.rank.mean(util.img_as_ubyte(img), footprint=disk)/255
    sum_img[mask==0] = 0  # Apply mask to the sum image

    fig, axes = plt.subplots(1, 3, figsize=(6, 4))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Sample {samples[idx]} - Original')
    axes[0].axis('off')

    axes[1].imshow(gauss_img, cmap='gray')
    axes[1].set_title('Gaussian')
    axes[1].axis('off')

    axes[2].imshow(sum_img, cmap='gray')
    axes[2].set_title('Sum')
    axes[2].axis('off')

    plt.tight_layout()
    # plt.show()

    # Save the processed images
    sample = samples[idx]
    tissue_fldr = f'{path}/{sample}/'
    exp_fldr = tissue_fldr + 'exp/'
    pre_fldr = tissue_fldr + 'pre/'

    skio.imsave(f'{exp_fldr}/pre_dsp_gauss.tif', util.img_as_ubyte(gauss_img))
    skio.imsave(f'{exp_fldr}/pre_dsp_sum.tif', util.img_as_ubyte(sum_img))
    skio.imsave(f'{exp_fldr}/pre_dsp.tif', util.img_as_ubyte(img))


    print(pre_fldr + 'improc_dsp')
    np.savez(pre_fldr + 'improc_dsp', mask=img, density=sum_img)
