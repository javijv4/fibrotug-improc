#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 08:54:46 2024

@author: Javiera Jilberto Vallejos
"""

import os
import numpy as np
from FtugTissue import FtugTissue, find_images
import matplotlib.pyplot as plt
from glob import glob


path = '/home/jilberto/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/fibroTUG/DSP/Tissues/dataset2_2/'
samples = os.listdir(path)
samples = [sample for sample in samples if os.path.isdir(path + sample)]
samples = sorted(samples)

dataset=2
if dataset == 1:
    mask_type = 'actin'
elif dataset == 2:
    mask_type = 'both'

#gem03 post, gem 04 pre, gem05 pre
samples = ['gem02'] #, 'gem03', 'gem04', 'gem05', 'gem08', 'gem10']
days = ['pre']
force_compute = False
for sample in samples:
    for day in days:
        tissue_fldr = f'{path}/{sample}/{day}/'
        # if 'gem' not in sample:
        #     continue
        # if os.path.exists(f'{tissue_fldr}/improc_dsp.npz'):
        #     continue

        print(f'Processing {sample} {day}')
    
        # try: 
        png_dump = tissue_fldr + 'png_dump/'
        if not os.path.exists(png_dump):
            os.makedirs(png_dump)

        images = find_images(tissue_fldr, dataset=dataset)

        tissue = FtugTissue(tissue_fldr, images)
        tissue.plot_images(png_dump)

        tissue.get_tissue_mask(tissue_mask_type=mask_type, force_compute=False)
        tissue.plot_tissue_mask(png_dump)

        # Fiber processing
        if 'fibers' in images:
            tissue.get_fiber_mask()
            tissue.process_fibers(force_compute=False)
            tissue.plot_fiber_processing(png_dump)

        # Actinin processing
        if 'actin' in images:
            tissue.get_actin_blob_mask()
            tissue.get_actin_mask()
            tissue.process_actin(force_compute=False)
            tissue.plot_actin_processing(png_dump)
            tissue.create_cell_mask()

        # DSP processing
        if 'dsp' in images:
            tissue.process_dsp(method='window', mask_method=1, force_compute=False)
            tissue.plot_dsp_processing(png_dump)
            tissue.plot_dsp_processing_zoom(png_dump, [740, 840, 140, 240])
        # except:
        #     print(f'Error processing {sample} {day}')
        #     continue