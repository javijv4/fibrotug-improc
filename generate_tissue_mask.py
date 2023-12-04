#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:52:48 2023

@author: Javiera Jilberto Vallejos
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_uint
from improcessing.masks import prepare_image, generate_initial_mask

file = 'test_data/post/A_0.41_0.1_GEM12_S2_ET1-01-MAX_c3_ORG.tif'
file = '/home/jilberto/Dropbox (University of Michigan)/Projects/fibroTUG/DSP/Tissues/gem04/pre/A_0.41_0.1_GEM12-04_MAX_c2_ORG.tif'

# Reading file
img_og = io.imread(file)

# Generating mask
img = prepare_image(img_og)
mask = generate_initial_mask(img, remove_size=3, closing_block_size=10)

# Saving mask in folder
io.imsave(os.path.dirname(file) + '/fibrotug_mask_init.tif',
          mask.astype(np.int8), check_contrast=False)

# Plot
mask[mask==0] = np.nan
plt.figure()
plt.imshow(img_og, cmap = 'binary_r')
plt.imshow(mask, alpha=0.5, cmap = 'RdBu')
plt.show()
