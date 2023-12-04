# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:32:15 2023

@author: laniq
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from skimage import filters, morphology, img_as_ubyte

def prepare_image(im, v_avg_threshold, view_masks):
    im = im[:, 5:-5]

    a = im.astype(float)
    a = signal.medfilt2d(a)
    
    v_image_range = [np.min(a), np.max(a)]
    v_image = (a - v_image_range[0]) / (v_image_range[1] - v_image_range[0])
    
    M = np.zeros_like(v_image)
    Co = v_image
    j = Co
    
    avg = np.nanpercentile(j, v_avg_threshold)
    is_below_avg = (Co <= avg) * Co + avg * (Co > avg)
    Co = (is_below_avg - np.nanmin(is_below_avg)) / (np.nanmax(is_below_avg) - np.nanmin(is_below_avg))
    
    if view_masks:
        Co = Co[1:-1, :]
        C = Co + M + (M > 0) - 1e-5 * (Co > 1e-5)
    else:
        C = Co - 1e-5 * (Co > 1e-5)
    
    # plt.imshow(C)
    # plt.title("Threshold Image") 
    # plt.show() 
    
    return C, a

def generate_initial_mask(image, method, block_size = 201, radius = 1,  remove_size=5):
    """
    :param method: Options are mean, local otsu (param radius), and local (param block_size)
    :return: intitial mask after thresholding
    """ 
    binaryMask = np.zeros_like(image)

    if method.lower() == "mean":
        thresh = filters.threshold_mean(image[np.isfinite(image)])
        mask = image > thresh

    elif method.lower() == "local otsu":
        selem = morphology.disk(radius)

        image = img_as_ubyte(image)
        thresh = filters.rank.otsu(image, selem)
        mask = image > thresh
  
    elif method.lower() == "local":
        if block_size is None:
            block_size = np.floor(np.min(image.shape)/10)*2+1
        thresh = filters.threshold_local(image, block_size=block_size, offset=0)
        mask = image > thresh
        

        
    else:
        # Handle the case where the input is not recognized
        print("Input not recognized")

    # # Removing noise due to local thresholding
    # mask = morphology.remove_small_objects(mask, remove_size)
    # mask = morphology.binary_closing(mask)

    binaryMask = mask.astype(float)
    binaryMask[np.isclose(binaryMask, 0.)] = np.nan

    # plt.figure()
    # plt.imshow(binaryMask)
    # plt.title(method + " threshold")
    # plt.show()
    return binaryMask
    
    