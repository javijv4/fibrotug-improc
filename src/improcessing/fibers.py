# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:32:15 2023

@author: laniq
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from skimage import filters, morphology, exposure, img_as_ubyte
from scipy.ndimage import distance_transform_edt
import cv2

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
    

# Calculate fiber density using Gaussian filter
def compute_local_density(mask, sigma_density):
    ldensity = filters.gaussian(mask, sigma=sigma_density)
    ldensity = ldensity / np.max(ldensity)
    return ldensity

def bwmorphclean(image):
    (ih, iw) = image.shape[:2]
    pad = 1
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((ih, iw), dtype="float32")

    for y in np.arange(pad, ih + pad):
        for x in np.arange(pad, iw + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            if roi[0,0] == roi[1,0] == roi [2,0] == roi [0,1] == roi [0,2] == roi [1,2] == roi [2,1] == roi [2,2] == 0:
                output[y - pad, x - pad] = 0
            else:
                output[y - pad, x - pad] = roi[1,1]

    output = exposure.rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output

    
def compute_local_orientation(mask):
    # # Clean the mask (Should we remove unconnected objects?)
    # mn = bwmorphclean(mask)
    # disk = morphology.disk(1)
    # mn = morphology.binary_dilation(mn, footprint=disk)
    # mn = morphology.binary_erosion(mn, footprint=disk)
    mn = mask   # TODO If I do this, I need to do it outside
        
    #IMAGE LOCAL ORIENTATION: eventually turn to function
    b = distance_transform_edt((np.logical_not(1-mn)))  # b shows how far I am away from the edge of the mask ...
    
    # computing the gradient image ...
    gx = np.zeros_like(b)
    fx = np.copy(gx)
    gy = np.zeros_like(b)
    fy = np.copy(gy)
    
    # Assume dx = 1 and use central differences
    gx[1:-1, :] = (b[2:, :] - b[:-2, :]) / (2)  # derivative db/dx
    gy[:, 1:-1] = (b[:, 2:] - b[:, :-2]) / (2)  # derivative db/dy
    
    # Computing the fiber components per pixel ...
    rotate = True
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if mn[i, j] != 1:  # if we are not a fiber pixel, move on ...
                continue
            v = np.array([gx[i, j], gy[i, j]])  # gradient ...
            if np.linalg.norm(v) < 1e-8:  # if the gradient is zero, we skip this one...
                continue
            v = v / np.linalg.norm(v)  # normalize so we have a unit vector ...
            if rotate:
                f = np.array([-v[1], v[0]])  # rotate the vector 90 degrees ...
            else:
                f = np.array([v[0], v[1]])  # rotate the vector 90 degrees ...
            if f[0] < 0:  # swap orientation of the vector so that it points between the posts (x-direction) ...
                f = -f
                
            fx[i, j] = f[0]  # set the x component of the fibers ...
            fy[i, j] = f[1]  # set the y component of the fibers ...
    theta = np.arcsin(fy)
            
    return theta


def smooth_fiber_angles(angles, fib_mask, window_size = 7):
    from scipy.signal import convolve

    window = np.ones([window_size, window_size])

    sum_angles = convolve(angles, window, mode='same', method='direct')
    sum_mask = convolve(fib_mask, window, mode='same', method='direct')

    smooth_angles = np.zeros_like(angles)
    smooth_angles[fib_mask==1] = sum_angles[fib_mask==1]/sum_mask[fib_mask==1]
    # smooth_angles[fib_mask==0] = np.nan

    return smooth_angles


def compute_fiber_dispersion(angles, fib_mask, window_size = 7):
    from scipy.signal import convolve
    window = np.ones([window_size, window_size])

    fx = np.cos(angles)
    fy = np.sin(angles)

    sum_fx = convolve(fx, window, mode='same', method='direct')
    sum_fy = convolve(fy, window, mode='same', method='direct')
    sum_mask = convolve(fib_mask, window, mode='same', method='direct')

    mean_fx = np.zeros_like(angles)
    mean_fx[fib_mask==1] = sum_fx[fib_mask==1]/sum_mask[fib_mask==1]
    mean_fy = np.zeros_like(angles)
    mean_fy[fib_mask==1] = sum_fy[fib_mask==1]/sum_mask[fib_mask==1]
    norm = np.sqrt(mean_fx**2 + mean_fy**2)
    mean_fx[norm > 0] = mean_fx[norm > 0]/norm[norm > 0]
    mean_fy[norm > 0] = mean_fy[norm > 0]/norm[norm > 0]

    fx[fib_mask==0] = np.nan
    fy[fib_mask==0] = np.nan

    dispersion = np.zeros_like(angles)
    for i in range(angles.shape[0]):
        for j in range(angles.shape[1]):
            if fib_mask[i,j] == 0: continue
            imin = np.max([0, i-window_size])
            imax = np.min([angles.shape[0]-1, i+window_size])
            jmin = np.max([0, j-window_size])
            jmax = np.min([angles.shape[1]-1, j+window_size])
            N = np.sum(fib_mask[imin:imax,jmin:jmax])
            if N <2:
                continue

            theta = np.arcsin(mean_fx[i,j]*fy[imin:imax,jmin:jmax] -
                                        mean_fy[i,j]*fx[imin:imax,jmin:jmax])

            # Von Mises dispersion
            R = np.sqrt((np.nansum(np.nansum(np.cos(theta*2+np.pi)))/N)**2 + (np.nansum(np.nansum(np.sin(theta*2+np.pi)))/N)**2);
            dispersion[i,j] = 0.5*(1-R);

    return dispersion