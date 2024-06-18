#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:46:35 2023

@author: Javiera Jilberto Vallejos
"""
import numpy as np
from matplotlib import pyplot as plt
from src.imregistration.ITKTransform import imageToArray
from skimage import measure, transform, io

import cv2

rotation = 2.0
crop = 125

rot_back = True

# Path to the masks. Results will be saved in tissue_folder
tissue_fldr = '../Tissues/gem04/exp/'
pre_mask_file = tissue_fldr + 'pre_mask'
post_mask_file = tissue_fldr + 'post_mask'
back_img_file = tissue_fldr + 'pre_act'

ext = '.tif'

# Read masks
pre_mask = imageToArray(pre_mask_file + ext)
post_mask = imageToArray(post_mask_file + ext)
extended_mask = pre_mask + post_mask
extended_mask[extended_mask>1] = 1

npad = 20
mask = np.pad(extended_mask, npad)
rot_mask = transform.rotate(mask, rotation, resize=True)
margin = np.array(rot_mask.shape) - np.array(mask.shape)
limx = margin[0]//2
limy = margin[1]//2
cnts = measure.find_contours(rot_mask, 0.5)[0].astype(np.int32)

# cropping to original size
cnts[:,0] -= limx
cnts[:,1] -= limy
rot_mask = rot_mask[limx:-limx, limy:-limy]
corner_x, corner_y, dim_x, dim_y = cv2.boundingRect(cnts)

box = np.zeros_like(rot_mask)
box[corner_x:(corner_x+dim_x), corner_y:(corner_y+dim_y)] = 1

crop_box = np.zeros_like(rot_mask)
crop_box[(corner_x+crop):(corner_x+dim_x-crop), corner_y:(corner_y+dim_y)] = 1
rot_mask = rot_mask[npad:-npad, npad:-npad]
box = box[npad:-npad, npad:-npad]
crop_box = crop_box[npad:-npad, npad:-npad]

# Read images
if rot_back:
    act_img = transform.rotate(imageToArray(back_img_file + ext), rotation)
else:
    act_img = imageToArray(back_img_file + ext)

# Compute box vertex
xmin = np.max([corner_x+crop-npad,0])
xmax = corner_x+dim_x-crop-npad
ymin = np.max([corner_y-npad,0])
ymax = corner_y+dim_y-npad

# Save values
vertex = np.array([xmin,xmax,ymin,ymax])
np.savez(tissue_fldr + 'transformation', rotation=rotation, box=vertex)

new_box = np.zeros_like(pre_mask)
new_box[xmin:xmax] = 1

# Saving mask in folder
io.imsave(tissue_fldr + '/fibrotug_mask.tif',
          extended_mask.astype(np.int8), check_contrast=False)

#%%
def remove_background(mask):
    mask = mask.astype(float)
    mask[np.isclose(mask,0)] = np.nan
    return mask

plt.figure(1, clear=True)
plt.imshow(act_img, cmap='binary_r')
plt.imshow(remove_background(rot_mask), alpha=0.3, vmin=0, vmax=1)
plt.imshow(remove_background(box), alpha=0.3, vmin=0, vmax=1)
plt.imshow(remove_background(crop_box), alpha=0.3, vmin=0, vmax=1)
# plt.axis('off')
# plt.savefig('check.png', bbox_inches='tight', dpi=180, pad_inches=0)