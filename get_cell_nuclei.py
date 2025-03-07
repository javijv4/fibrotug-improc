#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 07:58:05 2024

@author: Javiera Jilberto Vallejos
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from improcessing.nuclei import find_nuclei
from skimage import io
from glob import glob
import cheartio as chio

dataset_fldr = '/home/jilberto/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/fibroTUG/DSP/Tissues/'

dapi_imgs = glob(f'{dataset_fldr}/**/post/*c1*ORG.tif', recursive=True)

cell_radius = 18                    # Aprox cell radius in pixels
cleaning_size = cell_radius // 2   # This defines the disk size for cleaning the mask (i.e get rid of isolated pixels and others)
min_distance = cell_radius // 4     # Minimum distance between two cell nuclei

processed_number=0
# dapi_imgs = ['/home/jilberto/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/fibroTUG/DSP/Tissues/dataset2_2/elk01/post/A_0.41_0.1_ELK_s1_day9-01_MAX_c1_ORG.tif']
for img_path in dapi_imgs:
    try:
        print(f'Processing {img_path}')
        fldr = os.path.dirname(img_path) + '/'

        png_dump = fldr + 'png_dump/'
        image = io.imread(img_path)

        coords = find_nuclei(image, cleaning_size, cell_radius, min_distance,
                        show=True)         # This two options are to visualize intermediate steps
        plt.savefig(f'{png_dump}/nuclei.png', bbox_inches='tight', dpi=180)

        param_dict = {'nuclei_number': len(coords)}
        chio.dict_to_pfile(f'{fldr}/nuclei_cont.txt', param_dict)
        processed_number += 1
    except:
        pass

print('Number of files processed', processed_number)
