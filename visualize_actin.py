#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:56:51 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform

# inputs
tissue_fldr = '/home/jilberto/Dropbox (University of Michigan)/Projects/fibroTUG/DSP/Tissues/gem04/'
which = 'pre'
image = io.imread(tissue_fldr + which + '_act.tif')
# tissue_mask = io.imread(tissue_fldr + which + '_mask.tif')
actin_results = np.load(tissue_fldr + 'improc_' + which + '_actin.npz', allow_pickle=True)
angles = actin_results['smooth_angles']
myo_mask = ~np.isnan(angles)

# xy = np.vstack(np.where(myo_mask).T
# vector_x = np.sin(angles[myo_mask])
# vector_y = -np.cos(angles[myo_mask])

# every=10
# plt.figure()
# plt.imshow(image, cmap='binary_r')
# plt.quiver(xy[::every,1], xy[::every,0], vector_x[::every], vector_y[::every], color='r', scale_units='xy', scale=0.2,
#            headwidth=0, headlength=0, headaxislength=0)
# plt.show()

myo_mask = myo_mask.astype(float)
myo_mask[myo_mask==0] = np.nan

plt.figure(1, clear=True)
plt.imshow(image, cmap = 'binary_r')
# plt.imshow(myo_mask, alpha=0.5, vmin=0, vmax=1)
plt.imshow(angles, alpha=0.5, vmin=-np.pi/4, vmax=np.pi/4, cmap='RdBu')
plt.axis('off')
plt.savefig(tissue_fldr + 'plots/improc_' + which + '_actin.png', dpi=300, bbox_inches='tight',pad_inches = 0)
plt.colorbar()

plt.figure(1, clear=True)
plt.imshow(image, cmap = 'binary_r')
plt.imshow(myo_mask, alpha=0.5, vmin=0, vmax=1, cmap='bwr')
plt.axis('off')
plt.savefig(tissue_fldr + 'plots/improc_' + which + '_myo_mask.png', dpi=300, bbox_inches='tight',pad_inches = 0)
