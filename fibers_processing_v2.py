#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 23:29:16 2024

@author: jjv
"""

import os
from skimage import io, transform, filters, exposure, morphology
from improcessing.fibers import prepare_image, generate_initial_mask, bwmorphclean, compute_local_density, compute_local_orientation, smooth_fiber_angles, compute_fiber_dispersion
from scipy.ndimage import distance_transform_edt
import cv2
import numpy as np
import matplotlib.pyplot as plt

# inputs
tissue_fldr = 'test_data/'
which = 'post'
im_og = io.imread(tissue_fldr + which + '_act.tif', as_gray = True)
tissue_mask = io.imread(tissue_fldr + which + '_mask.tif', as_gray = True)
plot = True
sigma_density = 1
averaging_window = 7


v_avg_threshold = 100;

do = 'process_fibers'

if do == 'init_fib_mask':
    # #Threshold Image
    # image, _ = prepare_image(im_og, v_avg_threshold)
    
    # #Create mask
    # # Options are mean, local otsu (param radius), and local (param block_size)
    # mask = generate_initial_mask(image, 'local', block_size = 201)

    #Threshold Image
    image = exposure.equalize_adapthist(im_og, kernel_size=51)
    thresh1 = filters.threshold_local(image, block_size=51)
    mask1 = image > thresh1
    mask1 = morphology.binary_closing(mask1, morphology.disk(1))
    mask1 = morphology.binary_opening(mask1, morphology.disk(2))
    
    mask = generate_initial_mask(im_og, 'mean', radius = 41, block_size = 101, remove_size=11)
    mask[tissue_mask==0] = 0
    
    io.imsave(tissue_fldr + '/' + which + '/fibers_mask_init.tif',
              mask1.astype(np.int8), check_contrast=False)

elif do == 'process_fibers':
#%%
    mask = io.imread(tissue_fldr + '/' + which + '/fibers_mask.tif', as_gray = True)
    mask[tissue_mask==0] = 0        # Should be redundant but just in case
        
    # Perform erosion and dilation
    disk = morphology.disk(1)
    mask = bwmorphclean(mask)
    mask = morphology.binary_erosion(mask, footprint=disk)
    mask = morphology.binary_dilation(mask, footprint=disk)
    mask = morphology.remove_small_holes(mask, area_threshold=2)
    mask = mask.astype(float)
    
    if plot:
        # Display the processed image
        plt.figure(1, clear=True)
        plt.imshow(mask, cmap='gray')
        plt.show()
        
    ldensity = compute_local_density(mask, sigma_density)
    
    if plot:
        # Display the processed image
        plt.figure(2, clear=True)
        plt.imshow(ldensity, cmap='viridis')
        plt.show()
        
    # Compute local orientation
    theta = compute_local_orientation(mask)
    
    # Display the processed image
    if plot:
        plt.figure(5, clear=True)
        plt.imshow(theta, cmap='RdBu')
    
    # First smooth angles
    smooth_theta = smooth_fiber_angles(theta, mask, window_size=averaging_window)
            
    if plot:
        plt.figure(5, clear=True)
        plt.imshow(mask, cmap='binary_r')
        plt.imshow(smooth_theta, cmap='RdBu', vmin=-np.pi/4, vmax=np.pi/4)
        
        
    # Compute dispersion
    smooth_theta_disp = smooth_fiber_angles(theta, mask, window_size=5)
    dispersion = compute_fiber_dispersion(smooth_theta_disp, mask, window_size=averaging_window)
    
    if plot:
        plt.figure(6, clear=True)
        plt.imshow(mask, cmap='binary_r')
        plt.imshow(dispersion, alpha=1, cmap='magma', vmin=0, vmax=0.5)
        
    # Save results
    np.savez(tissue_fldr + 'improc_' + which + '_fiber', angles=smooth_theta, mask=mask, dispersion=dispersion, density=ldensity)

