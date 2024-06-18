#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:40:34 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
from skimage import transform, measure, draw, morphology
from scipy import ndimage, optimize
import cv2
from matplotlib.widgets import Slider, TextBox, RectangleSelector
import matplotlib.pyplot as plt


def normalize_image(image, mask=None, binary=False):
    if mask is not None:
        mask = morphology.binary_erosion(mask, morphology.disk(5))
        vmin = image[mask].min()
        vmax = image[mask].max()
    else:
        vmin = image.min()
        vmax = image.max()
    image = (image - vmin)/(vmax - vmin)
    if binary:
        image = image > 0.5

    image[image<0] = 0
    image[image>1] = 1
    
    return image

def rotate_crop(imgs, rotation, crop, npad=20):
    new_imgs = {}
    for name in imgs.keys():
        new_img = transform.rotate(imgs[name], rotation)
        new_img = new_img[crop[0]:crop[1], crop[2]:crop[3]]
        new_imgs[name] = new_img

    return new_imgs

def rotate_interactive(mask, params):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Display the initial image
    ax.imshow(mask, cmap='binary_r')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    hlines = np.linspace(*ylim, 20)
    hlines += (hlines[1]-hlines[0])/2
    ax.hlines(hlines, *xlim, color='r', lw=0.5)
    ax.set_ylim(ylim)

    # Create a slider for rotation angle
    ax_angle = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider_angle = Slider(ax_angle, 'Rotation Angle', -180, 180, valinit=0)
    
    # Create a text box for entering the rotation angle
    ax_textbox = plt.axes([0.25, 0.05, 0.65, 0.03])
    textbox_angle = TextBox(ax_textbox, 'Rotation Angle', initial='0')

    # Function to update the image based on the rotation angle entered in the text box
    def update_rotation_textbox(text):
        try:
            params[0] = float(text)
            rotated_tissue = transform.rotate(mask, params[0])
            ax.clear()
            ax.imshow(rotated_tissue, cmap='binary_r')
            ax.hlines(hlines, *xlim, color='r', lw=0.5)
            ax.set_ylim(ylim)
            return params[0]
        except ValueError:
            pass

    # Register the update function with the text box
    textbox_angle.on_submit(update_rotation_textbox)
    # Function to update the image based on the rotation params[0]
    def update_rotation(val):
        params[0] = slider_angle.val
        rotated_tissue = transform.rotate(mask, params[0])
        ax.clear()
        ax.imshow(rotated_tissue, cmap='binary_r')
        ax.hlines(hlines, *xlim, color='r', lw=0.5)
        ax.set_ylim(ylim)   
        return params[0]

    # Register the update function with the slider
    slider_angle.on_changed(update_rotation)
    plt.show()

    return params




class CroppingWindow:
    def __init__(self, image):
        self.image = image
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)
        self.rs = RectangleSelector(self.ax, self.onselect, useblit=True, interactive=True)
        plt.connect('key_press_event', self.toggle_selector)
        plt.show()

    def onselect(self, eclick, erelease):
        self.x1, self.y1 = int(eclick.xdata), int(eclick.ydata)
        self.x2, self.y2 = int(erelease.xdata), int(erelease.ydata)

    def toggle_selector(self, event):
        if event.key in ['Q', 'q'] and self.rs.active:
            self.rs.set_active(False)
        if event.key in ['A', 'a'] and not self.rs.active:
            self.rs.set_active(True)

def crop_interactive(image, rotation):
    image = transform.rotate(image, rotation)
    cropping_window = CroppingWindow(image)
    return cropping_window.x1, cropping_window.x2, cropping_window.y1, cropping_window.y2


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
