#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:29:48 2024

@author: Javiera Jilberto Vallejos
"""

from matplotlib import pyplot as plt
import numpy as np
from skimage import filters, morphology, exposure, measure, transform

def process_dsp_image(image, tissue_mask=None):
    mask = image > filters.threshold_otsu(image)

    mask = morphology.binary_closing(mask, footprint=morphology.disk(2))

    if tissue_mask is not None:
        mask = mask * tissue_mask
    

    return mask