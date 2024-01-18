#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:21:13 2024

@author: Javiera Jilberto Vallejos
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, exposure

fname = '/home/jilberto/Dropbox (University of Michigan)/Projects/fibroTUG/DSP/Tissues/gem04/exp/A_0.41_0.1_GEM12-S2-04'
img = io.imread(fname + '.tif')
cut = 511
rot = -65

shape_x = img.shape[1]

# Fix cut
new_img = np.zeros_like(img)
new_img[:,:,0:(shape_x-cut)] = img[:,:,cut:]
new_img[:,:,(shape_x-cut):] = img[:,:,:cut]

# Rotate
sample = new_img[0]
rot_sample = transform.rotate(sample, rot)

for i in range(new_img.shape[0]):
    new_img[i] = transform.rotate(new_img[i], rot, preserve_range=True)

def adjust_contrast(img):
    pt10 = np.percentile(img, 15)
    pt90 = np.percentile(img, 95)
    print(pt10, pt90)
    scaled = (img-pt10)/(pt90-pt10)
    scaled[scaled < 0] = 0
    scaled[scaled > 1] = 1
    return scaled

plt.figure(1, clear=True)
plt.imshow(adjust_contrast(new_img[10]), cmap='binary_r')
plt.vlines(np.arange(0, shape_x, 100)[1:], 0, shape_x, colors='r')
plt.ylim([0, shape_x])

# for i in range()

io.imsave(fname + '_fix.tif', new_img)