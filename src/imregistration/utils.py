#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:40:34 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
from skimage import transform, measure, draw
from scipy import ndimage, optimize
import cv2


def rotate_crop(imgs, rotation, crop, npad=20):
    new_imgs = {}
    for name in imgs.keys():
        new_img = transform.rotate(imgs[name], rotation)
        new_img = new_img[crop[0]:crop[1], crop[2]:crop[3]]
        new_imgs[name] = new_img

    return new_imgs


def find_box_points(hull, npad=20):
    mask = np.pad(hull, npad)

    # Function to calculate the difference in areas
    def mask_difference(points, original_mask):
        polygon_vertices = np.round(points.reshape([-1,2])).astype(int)
        generated_mask = draw.polygon2mask(original_mask.shape, polygon_vertices)

        return np.sum(np.logical_xor(original_mask, generated_mask))

    cnts = measure.find_contours(mask, 0.5)[0].astype(np.int32)
    rect = cv2.minAreaRect(cnts)
    initial_points = np.intp(cv2.boxPoints(rect))

    # Optimization using SciPy
    result = optimize.minimize(mask_difference, initial_points.flatten(), args=(mask), method='Powell')

    # Extract optimized points
    optimized_points = result.x.reshape((4, 2))-npad

    return optimized_points


def find_tissue_axis(mask):
    mask_mod = np.pad(mask, 20) # Just to avoid having cut boundaries
    cnts = measure.find_contours(mask_mod, 0.5)[0].astype(np.int32)
    print(cv2.minAreaRect(cnts))
    center, dimensions, rotation = cv2.minAreaRect(cnts)
    rect = cv2.minAreaRect(cnts)
    box = np.int0(cv2.boxPoints(rect))

    return center, dimensions, rotation


"""
The following codes are taken from https://github.com/HibaKob/MicroBundleCompute/tree/master
"""

def insert_borders(mask: np.ndarray, border: int = 10) -> np.ndarray:
    """Given a mask. Will make the borders around it 0."""
    mask[0:border, :] = 0
    mask[-border:, :] = 0
    mask[:, 0:border] = 0
    mask[:, -border:] = 0
    return mask


def compute_unit_vector(point1, point2) -> np.ndarray:
    """Given two 2D points. Will return the unit vector between them"""
    vec = point2 - point1
    dist = np.linalg.norm(vec)
    vec = vec/dist
    return vec


def box_to_unit_vec(box: np.ndarray) -> np.ndarray:
    """Given the rectangular box. Will compute the unit vector of the longest side."""
    side_1 = np.linalg.norm(box[1] - box[0])
    side_2 = np.linalg.norm(box[2] - box[1])
    if side_1 > side_2:
        # side_1 is the long axis
        vec = compute_unit_vector(box[0], box[1])
    else:
        # side_2 is the long axis
        vec = compute_unit_vector(box[1], box[2])
    return vec

def box_to_center_points(box: np.ndarray) -> float:
    """Given the rectangular box. Will compute the center as the midpoint of a diagonal."""
    center_row = np.mean([box[0, 0], box[2, 0]])
    center_col = np.mean([box[0, 1], box[2, 1]])
    return center_row, center_col


def mask_to_box(mask: np.ndarray) -> np.ndarray:
    """Given a mask. Will return the minimum area bounding rectangle."""
    # insert borders to the mask
    border = 10
    mask_mod = insert_borders(mask, border)
    # find contour
    mask_mod_one = (mask_mod > 0).astype(np.float64)
    mask_thresh_blur = ndimage.gaussian_filter(mask_mod_one, 1)
    cnts = measure.find_contours(mask_thresh_blur, 0.75)[0].astype(np.int32)
    # find minimum area bounding rectangle
    rect = cv2.minAreaRect(cnts)
    box = np.int0(cv2.boxPoints(rect))
    return box


def axis_from_mask(mask: np.ndarray) -> np.ndarray:
    """Given a folder path. Will import the mask and determine its long axis."""
    box = mask_to_box(mask)
    vec = box_to_unit_vec(box)
    center = box_to_center_points(box)
    return center, vec
