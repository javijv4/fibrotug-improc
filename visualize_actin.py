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
tissue_fldr = 'test_data/'
which = 'pre'
image = io.imread(tissue_fldr + which + '_act.tif')
# tissue_mask = io.imread(tissue_fldr + which + '_mask.tif')
actin_results = np.load(tissue_fldr + 'improc_' + which + '_actin.npz', allow_pickle=True)
angles = actin_results['angles']
dispersion = actin_results['dispersion']
myo_mask = ~np.isnan(angles)

xy = np.vstack(np.where(myo_mask)).T
vector_x = np.sin(angles[myo_mask])
vector_y = -np.cos(angles[myo_mask])

every=10
plt.figure(0, clear=True)
plt.imshow(image, cmap='binary_r')
plt.quiver(xy[::every,1], xy[::every,0], vector_x[::every], vector_y[::every], color='r', scale_units='xy', scale=0.2,
            headwidth=0, headlength=0, headaxislength=0)
plt.show()

myo_mask = myo_mask.astype(float)
myo_mask[myo_mask==0] = np.nan

plt.figure(1, clear=True)
plt.imshow(image, cmap = 'binary_r')
# plt.imshow(myo_mask, alpha=0.5, vmin=0, vmax=1)
plt.imshow(angles, alpha=0.7, vmin=-np.pi/4, vmax=np.pi/4, cmap='RdBu')
plt.axis('off')
cbar = plt.colorbar()
cbar.set_ticks([-np.pi/4,0,np.pi/4])
cbar.set_ticklabels(['-45', '0', '45'])
# cbar.ax.yaxis.set_ticks_position('left')
plt.savefig(tissue_fldr + 'plots/improc_' + which + '_actin.png', dpi=300, bbox_inches='tight',pad_inches = 0)

plt.figure(2, clear=True)
plt.imshow(image, cmap = 'binary_r')
plt.imshow(myo_mask, alpha=0.5, vmin=0, vmax=1, cmap='bwr')
plt.axis('off')
plt.savefig(tissue_fldr + 'plots/improc_' + which + '_myo_mask.png', dpi=300, bbox_inches='tight',pad_inches = 0)

plt.figure(3, clear=True)
plt.imshow(image, cmap = 'binary_r')
plt.imshow(dispersion, alpha=0.9, vmin=0, vmax=0.5, cmap='magma')
plt.axis('off')
cbar = plt.colorbar()
plt.savefig(tissue_fldr + 'plots/improc_' + which + '_dispersion.png', dpi=300, bbox_inches='tight',pad_inches = 0)
