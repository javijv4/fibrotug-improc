#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:29:48 2024

@author: Javiera Jilberto Vallejos
"""

from matplotlib import pyplot as plt
import numpy as np
from skimage import filters, morphology, exposure, measure, transform


def get_dsp_mask(image, tissue_mask=None, fiber_mask=None):
    if fiber_mask is None:
        mask = image > filters.threshold_otsu(image)
    else:
        fiber_threshold = filters.threshold_otsu(image[fiber_mask])
        mask = image > fiber_threshold*2

    # mask = morphology.remove_small_objects(mask, min_size=4)
    mask = morphology.binary_closing(mask, footprint=morphology.disk(2))

    if tissue_mask is not None:
        mask = mask * tissue_mask


    return mask


def process_dsp(mask, method='window'):
    if method == 'gaussian':
        dsp_density = filters.gaussian(mask, sigma=25)
    elif method == 'window':
        from scipy.signal import convolve
        window_size = 25
        window = morphology.disk(window_size)
        # window = np.ones([window_size, window_size])

        dsp_density = convolve(mask, window, mode='same', method='direct')

    dsp_density = dsp_density/np.max(dsp_density)

    return dsp_density