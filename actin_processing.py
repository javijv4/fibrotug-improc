#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 08:54:46 2023

@author: Javiera Jilberto Vallejos
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, filters
import src.improcessing.actin as actproc

# inputs
tissue_fldr = 'test_data/exp/'
which = 'pre'
image = io.imread(tissue_fldr + which + '_act.tif')
tissue_mask = io.imread(tissue_fldr + which + '_mask.tif')

do = 'compute_angles'

# Preparing image and calculate blobs mask
work_image = actproc.prepare_image(image, eq_block_size=49)

if do == 'init_blob_mask':
    blobs_mask = actproc.clean_blobs_mask(work_image, blob_threshold=0.9, eccentricity_threshold=0.9, blob_min_size=100)
    blobs_mask = transform.rescale(blobs_mask, 0.5)
    # Saving mask in folder
    io.imsave(tissue_fldr + which + '_blob_mask_init.tif',
              blobs_mask.astype(np.int8), check_contrast=False)

    plt.figure(1,clear=True)
    plt.imshow(image, cmap='binary_r')
    plt.imshow(blobs_mask, alpha=0.5)

    print('Blobs mask complete')

elif do == 'init_myo_mask':
    # Load blobs mask
    blobs_mask = io.imread(tissue_fldr + which + '_blob_mask_init.tif')
    blobs_mask = transform.rescale(blobs_mask, 2) > 0

    # Iterative thresholding process
    results = np.zeros(blobs_mask.shape)
    thresholds = [0.9,0.8,0.7,0.6,0.5,0.4]
    dilation = [4,4,3,3,2,2]
    results, remove_mask = actproc.actin_iterative_thresholding(work_image, thresholds, dilation=dilation, blobs_mask=blobs_mask)

    # Local thresholding
    angles, remove_mask = actproc.actin_local_thresholding(work_image, results, remove_mask, blobs_mask=blobs_mask)
    print('Myofibril angles calculation complete')

    # Compute Sarcomere Mask
    myo_mask = actproc.compute_myofibril_mask(angles)
    myo_mask = transform.rescale(myo_mask, 0.5)
    # Saving mask in folder
    io.imsave(tissue_fldr + which + '_myo_mask_init.tif',
              myo_mask.astype(np.int8), check_contrast=False)
    np.save(tissue_fldr + which + 'angles_init.npy', angles)

    plt.figure(1,clear=True)
    plt.imshow(image, cmap='binary_r')
    plt.imshow(myo_mask, alpha=0.5)


elif do == 'compute_angles':
    # Load previous calculations
    blobs_mask_init = actproc.clean_blobs_mask(work_image, blob_threshold=0.9, eccentricity_threshold=0.9, blob_min_size=100)
    blobs_mask = io.imread(tissue_fldr + which + '_blob_mask.tif')
    blobs_mask = transform.rescale(blobs_mask, 2) > 0
    interpolate_mask = blobs_mask_init.astype(int)-blobs_mask.astype(int)
    interpolate_mask[interpolate_mask<0] = 0

    myo_mask = io.imread(tissue_fldr + which + '_myo_mask.tif')
    myo_mask = transform.rescale(myo_mask, 2) > 0
    angles = np.load(tissue_fldr + which + 'angles_init.npy')

    # Nearest neighbor interpolation
    angles = actproc.nearest_interpolation(interpolate_mask, myo_mask, angles)
    myo_mask = myo_mask + interpolate_mask
    myo_mask[myo_mask>1] = 1

    # Mask results to tissue mask
    angles, myo_mask = actproc.mask_actin_results(angles, myo_mask, tissue_mask)
    print('Sarcomere mask complete')

    # Smooth myofibril results
    smooth_angles = actproc.smooth_actin_angles(angles, myo_mask, window_size = 11)

    # Compute dispersion
    dispersion = actproc.compute_dispersion(angles, myo_mask, window_size = 35)
    myo_mask = transform.rescale(myo_mask, 0.5) > 0
    print('Smoothing of angles complete')

    # Density
    density = filters.gaussian(myo_mask, sigma=2)

    # Save results
    np.savez(tissue_fldr + 'improc_' + which + '_actin', angles=smooth_angles, mask=myo_mask, dispersion=dispersion, density=density)

    # Plots
    from skimage import io, transform
    xy = np.vstack(np.where(myo_mask)).T
    vector_x = np.sin(smooth_angles[myo_mask])
    vector_y = -np.cos(smooth_angles[myo_mask])

    every=10
    plt.figure(1, clear=True)
    plt.imshow(image, cmap='binary_r')
    plt.quiver(xy[::every,1], xy[::every,0], vector_x[::every], vector_y[::every], color='r', scale_units='xy', scale=0.25,
                headwidth=0, headlength=0, headaxislength=0, width=0.003)
    plt.show()