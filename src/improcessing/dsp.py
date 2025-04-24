#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:29:48 2024

@author: Javiera Jilberto Vallejos
"""

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, TextBox
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

def get_dsp_mask_interactive(image, tissue_mask=None, fiber_mask=None):
    img = exposure.equalize_adapthist(image)

    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax1.imshow(img, cmap='gray')
    ax1.axis('off')

    center = (img.shape[0] // 2, img.shape[1] // 2)
    zoom_size = 50  # Size of the zoomed-in area
    zoomed_img = img[center[0] - zoom_size:center[0] + zoom_size, center[1] - zoom_size:center[1] + zoom_size]
    ax2.imshow(zoomed_img, cmap='gray')
    ax2.axis('off')

    axcolor = 'lightgoldenrodyellow'
    ax_threshold = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_textbox = plt.axes([0.25, 0.15, 0.1, 0.05])

    threshold_init = filters.threshold_otsu(img)
    slider_threshold = Slider(ax_threshold, 'Threshold', 0.0, 1.0, valinit=threshold_init / img.max())
    text_box = TextBox(ax_textbox, 'Input Threshold', initial=str(threshold_init / img.max()))

    def update(val):
        threshold = slider_threshold.val * img.max()
        mask = img > threshold

        mask = morphology.binary_closing(mask, footprint=morphology.disk(2))

        if tissue_mask is not None:
            mask = mask * tissue_mask

        mask = mask.astype(float)
        mask[mask == 0] = np.nan
        ax1.imshow(img, cmap='gray')
        ax1.imshow(mask, cmap='viridis', vmin=0, vmax=1, alpha=0.5)
        zoomed_mask = mask[center[0] - zoom_size:center[0] + zoom_size, center[1] - zoom_size:center[1] + zoom_size]
        ax2.imshow(zoomed_img, cmap='gray')
        ax2.imshow(zoomed_mask, cmap='viridis', vmin=0, vmax=1, alpha=0.5)
        fig.canvas.draw_idle()

    def submit(text):
        try:
            val = float(text)
            slider_threshold.set_val(val)
        except ValueError:
            pass

    slider_threshold.on_changed(update)
    text_box.on_submit(submit)
    plt.show()

    threshold = slider_threshold.val * img.max()
    mask = img > threshold

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

    dsp_density = dsp_density/np.sum(dsp_density)

    return dsp_density