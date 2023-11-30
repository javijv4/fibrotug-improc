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
which = 'post'
image = io.imread(tissue_fldr + which + '_act.tif')
tissue_mask = io.imread(tissue_fldr + which + '_mask.tif')
actin_results = np.load(tissue_fldr + 'improc_' + which + '_actin.npz', allow_pickle=True)
angles = actin_results['smooth_angles']
myo_mask = ~np.isnan(angles)
myo_mask = myo_mask.astype(float)
myo_mask[myo_mask==0] = np.nan

plt.figure(1, clear=True)
plt.imshow(image, cmap = 'binary_r')
# plt.imshow(myo_mask, alpha=0.5, vmin=0, vmax=1)
plt.imshow(angles, alpha=0.5, vmin=-np.pi/4, vmax=np.pi/4, cmap='RdBu')
plt.axis('off')
plt.savefig(tissue_fldr + 'plots/improc_' + which + '_actin.png', dpi=300, bbox_inches='tight')