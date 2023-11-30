#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:29:48 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
from skimage import filters 
from skimage import morphology
from skimage import exposure
from skimage import measure


def prepare_image(image, eq_block_size=None):
    new_image = exposure.equalize_adapthist(image, kernel_size=eq_block_size)
    new_image = filters.unsharp_mask(new_image, radius=15, amount=1)

    return new_image

def generate_initial_mask(image, block_size=None, remove_size=5, closing_block_size=5):
    # Generating initial mask
    if block_size is None:
        block_size = np.floor(np.min(image.shape)/10)*2+1
    thresh = filters.threshold_local(image, block_size=block_size)
    mask = image > thresh

    # Removing noise due to local thresholding
    mask = morphology.remove_small_objects(mask, remove_size)
    mask = morphology.binary_closing(mask)


    # get largest cluster
    labelled = measure.label(mask, connectivity=1)
    rp = measure.regionprops(labelled)
    sizes = ([i.area for i in rp])
    mask = labelled == (np.argmax(sizes)+1)

    # Trying to fill any gap in the boundary
    mask = morphology.binary_closing(mask, footprint=morphology.disk(closing_block_size))

    # Reconstruct mask
    # https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_holes_and_peaks.html#sphx-glr-auto-examples-features-detection-plot-holes-and-peaks-py
    seed = np.copy(mask)
    seed[1:-1, 1:-1] = mask.max()
    mask = morphology.reconstruction(seed, mask, method='erosion')

    return mask
