#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:29:48 2025

@author: Javiera Jilberto Vallejos
"""

from skimage import io, feature, filters, morphology, measure
from scipy import ndimage
from matplotlib import pyplot as plt
import numpy as np


def check_mask(mask, cell_radius):
    area_cell = np.pi*cell_radius**2

    objects = measure.label(mask)
    props = measure.regionprops(objects)
    areas = np.array([prop.area for prop in props])

    largest_area = np.max(areas)

    if largest_area > 20*area_cell:
        return False

    eccentricities = np.array([prop.eccentricity for prop in props])
    good_area = (areas > area_cell*0.1) & (areas < area_cell*2)
    good_ecc = eccentricities < 0.95

    if np.sum(good_area & good_ecc) > 0.8*len(areas):
        return True
    else:
        return False

def find_nuclei(image, cleaning_size, cell_radius, min_distance,
                show=False):
    # Threshold image
    thresh_mult = 0.25
    check = False
    while not check:
        mask = image > filters.threshold_yen(image)*thresh_mult
        mask = morphology.binary_opening(mask, footprint=morphology.disk(cleaning_size))
        check = check_mask(mask, cell_radius)
        thresh_mult += 0.25
        if thresh_mult > 10:
            raise ValueError('Could not find a good threshold')

    dist = ndimage.distance_transform_edt(mask)

    # Template image
    template = morphology.disk(cell_radius)
    temp_dist = ndimage.distance_transform_edt(template)

    # Match template and find peaks
    result = feature.match_template(dist, temp_dist)
    result_pad = np.zeros_like(dist)
    result_pad[cell_radius:-cell_radius, cell_radius:-cell_radius] = result
    centers = result_pad > 0.1
    result_pad = result_pad*centers
    coords = feature.peak_local_max(result_pad, min_distance=min_distance, threshold_abs=0.1)

    # Figures if asked
    if show:
        fig, axes = plt.subplots(1, 3, figsize=(5, 5))
        mask = mask.astype(float)
        mask[mask == 0] = np.nan
        
        axes[0].imshow(image, cmap='binary_r')
        axes[0].imshow(mask, cmap='viridis', alpha=0.4, vmin=0, vmax=1)
        axes[0].plot(coords[:,1], coords[:,0], 'r.', ms=2)
        axes[0].axis('off')

        axes[1].imshow(result_pad, cmap='binary_r')
        axes[1].plot(coords[:,1], coords[:,0], 'r.', ms=2)
        axes[1].axis('off')
        
        axes[2].imshow(image, cmap='binary_r')
        axes[2].plot(coords[:,1], coords[:,0], 'r.', ms=2)
        axes[2].axis('off')
        
        plt.tight_layout()

    return coords