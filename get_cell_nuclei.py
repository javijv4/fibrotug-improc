#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 07:58:05 2024

@author: Javiera Jilberto Vallejos
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature, filters, morphology
from scipy import ndimage

def find_nuclei(image, cleaning_size, thresh_mult, cell_radius, min_distance,
                show_mask=False, show_template_match=False):
    # Threshold image
    mask = image > filters.threshold_yen(image)*thresh_mult
    mask = morphology.binary_opening(mask, footprint=morphology.disk(cleaning_size))
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
    if show_mask:
        plt.figure(1, clear=True)
        plt.imshow(image, cmap='binary_r')
        plt.imshow(mask, alpha=0.5, cmap='viridis', vmin=0, vmax=1)
        plt.plot(coords[:,1], coords[:,0], 'r.', ms=2)
        plt.axis('off')

    if show_template_match:
        plt.figure(2, clear=True)
        plt.imshow(result_pad, cmap='binary_r')
        plt.plot(coords[:,1], coords[:,0], 'r.', ms=2)
        plt.axis('off')

    return coords

tissue_fldr = '/home/jilberto/Dropbox (University of Michigan)/Projects/fibroTUG/DSP/Tissues/dataset2/gem02/'
png_dump = tissue_fldr + 'png_dump/'
image = io.imread(tissue_fldr + 'day9/A_0.41_0.1_GEM12_s1_day9-02_MAX_c1_ORG.tif')
out_file = tissue_fldr + 'data/nuclei_centers.txt'

thresh_mult = 3.0                   # If the thresholding didnt do a good job, you can modify the threshold value with this
cell_radius = 20                    # Aprox cell radius in pixels
cleaning_size = cell_radius // 3    # This defines the disk size for cleaning the mask (i.e get rid of isolated pixels and others)
min_distance = cell_radius // 4     # Minimum distance between two cell nuclei

coords = find_nuclei(image, cleaning_size, thresh_mult, cell_radius, min_distance,
                show_mask=True, show_template_match=True)         # This two options are to visualize intermediate steps

np.savetxt(out_file, coords)

# Show result
plt.figure(3, clear=True)
plt.imshow(image, cmap='binary_r')
plt.plot(coords[:,1], coords[:,0], 'r.', ms=2)
plt.axis('off')
plt.savefig(png_dump + 'nuclei.png', dpi=180, bbox_inches='tight')
