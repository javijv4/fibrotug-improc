#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 08:54:46 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import improcessing.actin as actproc

# inputs
tissue_fldr = '/home/jilberto/Dropbox (University of Michigan)/Projects/fibroTUG/DSP/Tissues/gem04/'
which = 'post'
image = io.imread(tissue_fldr + which + '_act.tif')
tissue_mask = io.imread(tissue_fldr + which + '_mask.tif')

# Preparing image and calculate blobs mask
work_image = actproc.prepare_image(image, eq_block_size=49)
blobs_mask = actproc.clean_blobs_mask(work_image, blob_threshold=0.9, eccentricity_threshold=0.9, blob_min_size=100)
print('Blobs mask complete')

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

# Mask results to tissue mask
angles, myo_mask = actproc.mask_actin_results(angles, myo_mask, tissue_mask)
print('Sarcomere mask complete')

# Smooth myofibril results
smooth_angles = actproc.smooth_actin_angles(angles, myo_mask, window_size = 11)
print('Smoothing of angles complete')

# Save results
np.savez(tissue_fldr + 'improc_' + which + '_actin', angles=angles, smooth_angles=smooth_angles)

# Plots
from skimage import io, transform
myo_mask = transform.rescale(myo_mask, 0.5)
xy = np.vstack(np.where(myo_mask)).T
vector_x = np.sin(smooth_angles[myo_mask])
vector_y = -np.cos(smooth_angles[myo_mask])

every=10
plt.figure()
plt.imshow(image, cmap='binary_r')
plt.quiver(xy[::every,1], xy[::every,0], vector_x[::every], vector_y[::every], color='r', scale_units='xy', scale=0.2,
           headwidth=0, headlength=0, headaxislength=0)
plt.show()