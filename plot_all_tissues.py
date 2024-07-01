#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/06/21 08:42:14

@author: Javiera Jilberto Vallejos
'''

import os
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from skimage import io, transform

def get_images(tissue_fldr):
    images = {}

    tif_files = glob(tissue_fldr + '*.tif')
    for tif_file in tif_files:
        if 'day7' in tissue_fldr:
            if 'c1+2+3' in tif_file:
                continue
            elif 'c1' in tif_file:
                images['dsp'] = io.imread(tif_file)
            elif 'c2' in tif_file:
                images['fibers'] = io.imread(tif_file)
            elif 'c3' in tif_file:
                images['actin'] = io.imread(tif_file)
        elif 'day9' in tissue_fldr:
            if 'c1+2+3' in tif_file:
                continue
            elif 'c2' in tif_file:
                images['dsp'] = transform.rotate(io.imread(tif_file), 180)
            elif 'c3' in tif_file:
                images['fibers'] = transform.rotate(io.imread(tif_file), 180)
            elif 'c4' in tif_file:
                images['actin'] = transform.rotate(io.imread(tif_file), 180)

    return images


samples = ['gem02', 'gem03', 'gem04', 'gem05', 'gem08', 'gem10']
days = ['day7', 'day9']

fig, axs = plt.subplots(len(samples), 5, num=2, clear=True, figsize=(4,10))
fig.tight_layout()

for i, sample in enumerate(samples):
    tissue_fldr = f'../DSP/Tissues/dataset2/{sample}/'

    day7_images = get_images(f'{tissue_fldr}/day7/')
    day9_images = get_images(f'{tissue_fldr}/day9/')

    axs[i, 0].imshow(day7_images['dsp'], cmap='gray')
    axs[i, 1].imshow(day7_images['fibers'], cmap='gray')
    axs[i, 2].imshow(day7_images['actin'], cmap='gray')
    axs[i, 3].imshow(day9_images['actin'], cmap='gray')
    axs[i, 4].imshow(day9_images['fibers'], cmap='gray')
    axs[i, 0].set_ylabel(f'{sample}' )

for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.subplots_adjust(wspace=0.01, hspace=0.05)

plt.savefig('all_tissues.png', bbox_inches='tight', pad_inches=0.1, dpi=300)


fig, axs = plt.subplots(len(samples), 5, num=2, clear=True, figsize=(4,10))

for i, sample in enumerate(samples):
    tissue_fldr = f'../DSP/Tissues/dataset2/{sample}/'

    day7_images = get_images(f'{tissue_fldr}/day7/')
    day9_images = get_images(f'{tissue_fldr}/day9/')

    data = np.load(f'{tissue_fldr}day7/improc_actin.npz')
    day7_actin_density = data['density']
    # day7_actin_density[day7_actin_density == 0] = np.nan

    data = np.load(f'{tissue_fldr}day7/improc_dsp.npz')
    day7_dsp_density = data['density']
    # day7_dsp_density[day7_dsp_density == 0] = np.nan

    data = np.load(f'{tissue_fldr}day7/improc_fiber.npz')
    day7_fibers_density = data['density']
    # day7_fibers_density[day7_fibers_density == 0] = np.nan

    data = np.load(f'{tissue_fldr}day9/improc_actin.npz')
    day9_actin_density = transform.rotate(data['density'], 180)
    # day9_actin_density[day9_actin_density == 0] = np.nan

    data = np.load(f'{tissue_fldr}day9/improc_fiber.npz')
    day9_fibers_density = transform.rotate(data['density'], 180)
    # day9_fibers_density[day9_fibers_density == 0] = np.nan


    axs[i, 0].imshow(day7_images['dsp'], cmap='gray')
    axs[i, 1].imshow(day7_images['fibers'], cmap='gray')
    axs[i, 2].imshow(day7_images['actin'], cmap='gray')
    axs[i, 3].imshow(day9_images['actin'], cmap='gray')
    axs[i, 4].imshow(day9_images['fibers'], cmap='gray')

    axs[i, 0].imshow(day7_dsp_density, cmap='viridis', alpha=0.7)
    axs[i, 1].imshow(day7_fibers_density, cmap='viridis', alpha=0.7)
    axs[i, 2].imshow(day7_actin_density, cmap='viridis', alpha=0.7)
    axs[i, 3].imshow(day9_actin_density, cmap='viridis', alpha=0.7)
    axs[i, 4].imshow(day9_fibers_density, cmap='viridis', alpha=0.7)
    axs[i, 0].set_ylabel(f'{sample}' )

for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.subplots_adjust(wspace=0.01, hspace=0.05)

plt.savefig('all_rhos.png', bbox_inches='tight', pad_inches=0.1, dpi=300)