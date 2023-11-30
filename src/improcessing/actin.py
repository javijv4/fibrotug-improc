#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:29:48 2023

@author: Javiera Jilberto Vallejos
"""

from matplotlib import pyplot as plt
import numpy as np
from skimage import filters, morphology, exposure, measure, transform

def prepare_image(image, eq_block_size=None):
    # Improving image
    if eq_block_size is None:
        eq_block_size = np.floor(np.min(image.shape)/20)*2+1
    new_image = exposure.equalize_adapthist(image, kernel_size=eq_block_size, clip_limit=0.03)

    # Rescaling image
    new_image = transform.rescale(new_image, 2)

    return new_image


def clean_blobs_mask(image, blob_threshold=0.97, eccentricity_threshold=0.95, blob_min_size=50, dilation=4):
    # Generating mask
    mask = image>blob_threshold
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=blob_min_size)

    # Finding blobs
    labelled = measure.label(mask)
    props = measure.regionprops(labelled)

    rmask = np.zeros_like(mask)
    for j, p in enumerate(props):
        if p.eccentricity > eccentricity_threshold: continue
        rmask += (labelled == (j+1))

    # Dilating mask
    rmask = morphology.binary_dilation(rmask, footprint=morphology.disk(dilation))

    return rmask


def actin_iterative_thresholding(image, thresholds, dilation=2, blobs_mask=None):
    # Dilation must be a list of the thresholds variable length
    if isinstance(dilation, int):
        dilation = [dilation]*len(thresholds)

    it_remove_mask = np.zeros(image.shape, dtype=bool)
    results = np.zeros(image.shape)
    for i, thresh in enumerate(thresholds):
        # Threshold mask
        mask = (image>thresh)

        # If given, remove blobs
        if blobs_mask is not None:
            mask = mask.astype(int) - blobs_mask.astype(int)
            mask[mask<0] = 0
            mask = mask.astype(bool)

        # Remove noise
        mask = morphology.remove_small_objects(mask, min_size=15)

        # Finding only elongated regions
        labelled = measure.label(mask)
        props = measure.regionprops(labelled)

        for j, p in enumerate(props):
            if p.eccentricity < 0.9: continue
            # Mask of the region
            rmask = labelled == (j+1)
            rmask = morphology.binary_dilation(rmask, footprint=morphology.disk(dilation[i]))

            # Remove regions already processed
            rmask = rmask.astype(int) - it_remove_mask.astype(int)
            rmask[rmask<0] = 0
            rmask = rmask.astype(bool)

            # Save results and remove_mask
            results[rmask] = p.orientation
            it_remove_mask += rmask


    return results, it_remove_mask


def actin_local_thresholding(image, results, remove_mask, blobs_mask=None, block_size=51):
    # Local thresholding
    thresh = filters.threshold_local(image, block_size=block_size)
    mask = image > thresh

    # Remove already found regions
    mask = (mask.astype(int) - remove_mask.astype(int))

    # If given, remove blobs
    if blobs_mask is not None:
        mask = mask.astype(int) - blobs_mask.astype(int)

    mask[mask<0] = 0
    mask = mask.astype(bool)

    # Clean-up
    mask = morphology.remove_small_objects(mask, min_size=30)

    # Finding only elongated regions
    labelled = measure.label(mask)
    props = measure.regionprops(labelled)

    for i, p in enumerate(props):
        if p.eccentricity < 0.95: continue
        # Mask of the region
        rmask = labelled == (i+1)
        rmask = morphology.binary_dilation(rmask, footprint=morphology.disk(2))

        # Remove regions already processed
        rmask = rmask.astype(int) - remove_mask.astype(int)
        rmask[rmask<0] = 0
        rmask = rmask.astype(bool)

        # Save results and remove_mask
        results[rmask] = p.orientation
        remove_mask += rmask

    return results, remove_mask


def compute_myofibril_mask(angles, dilation=2):
    myo_mask = np.abs(angles) > 0
    myo_mask = morphology.binary_dilation(myo_mask, footprint=morphology.disk(dilation))

    return myo_mask


def mask_actin_results(angles, myo_mask, tissue_mask):
    mask = transform.rescale(tissue_mask, 2)
    mask = mask.astype(bool)
    angles[~mask] = 0
    myo_mask[~mask] = 0

    return angles, myo_mask


def smooth_actin_angles(angles, myo_mask, window_size = 11, resize=0.5):
    from scipy.signal import convolve

    window = np.ones([window_size, window_size])

    sum_angles = convolve(angles, window, mode='same', method='direct')
    sum_mask = convolve(myo_mask, window, mode='same', method='direct')

    smooth_angles = np.zeros_like(angles)
    smooth_angles[myo_mask==1] = sum_angles[myo_mask==1]/sum_mask[myo_mask==1]

    if resize != 1.0:
        smooth_angles = transform.rescale(smooth_angles, resize)
        myo_mask = transform.rescale(myo_mask, resize)
    smooth_angles[myo_mask==0] = np.nan

    return smooth_angles

