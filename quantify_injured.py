#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/06/21 08:42:14

@author: Javiera Jilberto Vallejos
'''

import numpy as np
import meshio as io
import matplotlib.pyplot as plt
from skimage import io, morphology
import pandas as pd

path = '/Users/jjv/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/fibroTUG/DSP/Tissues/dataset2_2/'
samples = ['gem02', 'gem03', 'gem04', 'gem05', 'gem08', 'gem10', 'wt01', 'wt02', 'wt04', 'wt05', 'wt07', 'wt08']
fig, axs = plt.subplots(1, 3, figsize=(10, 3))

tissue_region_label = {}
fiber_density = {}
actin_density = {}
dsp_density = {}
tissue_masks = {}
post_actin_density = {}
for sample in samples:
    exp_folder = f'{path}/{sample}/exp/'
    imgs = {}
    for quant in ['tissue_mask', 'fiber_density', 'actin_density', 'dsp_mask', 'cell_mask']:
        imgs[quant] = io.imread(f'{exp_folder}/pre_{quant}.tif')

    imgs['post_cell_mask'] = io.imread(f'{exp_folder}/post_cell_mask.tif')
    imgs['post_actin_density'] = io.imread(f'{exp_folder}/post_actin_density.tif')
    injured_region = imgs['post_cell_mask'] - imgs['cell_mask']
    injured_region = morphology.binary_opening(injured_region, morphology.disk(5))
    injured_region[imgs['tissue_mask'] == 0] = 0
    non_injured_region = np.logical_not(injured_region)
    non_injured_region[imgs['tissue_mask'] == 0] = 0

    region_label = np.zeros_like(imgs['tissue_mask'])
    region_label[injured_region] = 1
    region_label[non_injured_region] = 2
    tissue_region_label[sample] = region_label

    fiber_density[sample] = imgs['fiber_density']
    actin_density[sample] = imgs['actin_density']
    dsp_density[sample] = imgs['dsp_mask']
    tissue_masks[sample] = imgs['tissue_mask']
    post_actin_density[sample] = imgs['post_actin_density']


#%% Plot 1. Injured tissues
samples = ['gem02', 'gem03', 'gem08', 'gem10']
fiber_quant = {'injured': [], 'non_injured': []}
actin_quant = {'injured': [], 'non_injured': []}
dsp_quant = {'injured': [], 'non_injured': []}
for sample in samples:
    regions = tissue_region_label[sample]
    injured_region = regions == 1
    non_injured_region = regions == 2
    area_injured = np.sum(injured_region)
    area_non_injured = np.sum(non_injured_region)
    total_area = area_injured + area_non_injured

    injured_fiber_density = np.sum(fiber_density[sample][injured_region]) / area_injured
    injured_actin_density = np.sum(actin_density[sample][injured_region]) / area_injured

    non_injured_fiber_density = np.sum(fiber_density[sample][non_injured_region]) / area_non_injured
    non_injured_actin_density = np.sum(actin_density[sample][non_injured_region]) / area_non_injured
    non_injured_dsp_density = np.sum(dsp_density[sample][non_injured_region]) / area_non_injured

    total_dsp_density = np.sum(dsp_density[sample]) 
    injured_dsp_density = np.sum(dsp_density[sample][injured_region]) / total_dsp_density
    non_injured_dsp_density = np.sum(dsp_density[sample][non_injured_region]) / total_dsp_density

    fiber_quant['injured'].append(injured_fiber_density)
    fiber_quant['non_injured'].append(non_injured_fiber_density)
    actin_quant['injured'].append(injured_actin_density)
    actin_quant['non_injured'].append(non_injured_actin_density)
    dsp_quant['injured'].append(injured_dsp_density)
    dsp_quant['non_injured'].append(non_injured_dsp_density)

    axs[0].plot([1, 2], [non_injured_fiber_density, injured_fiber_density], 'o-')
    axs[0].set_xticks([1, 2])
    axs[0].set_xticklabels(['Non-injured', 'Injured'])
    axs[0].set_ylabel('Fiber Density')

    axs[1].plot([1, 2], [non_injured_actin_density, injured_actin_density], 'o-')
    axs[1].set_xticks([1, 2])
    axs[1].set_xticklabels(['Non-injured', 'Injured'])
    axs[1].set_ylabel('Actin Density')

    axs[2].plot([1, 2], [non_injured_dsp_density, injured_dsp_density], 'o-')
    axs[2].set_xticks([1, 2])
    axs[2].set_xticklabels(['Non-injured', 'Injured'])
    axs[2].set_ylabel('DSP Density') 
    
plt.tight_layout()
plt.subplots_adjust(hspace=2.)
plt.show()

# Save data
fiber_quant_df = pd.DataFrame.from_dict(fiber_quant, orient='index').transpose()
fiber_quant_df.to_csv('fiber_quant.csv', index=False)
actin_quant_df = pd.DataFrame.from_dict(actin_quant, orient='index').transpose()
actin_quant_df.to_csv('actin_quant.csv', index=False)
dsp_quant_df = pd.DataFrame.from_dict(dsp_quant, orient='index').transpose()
dsp_quant_df.to_csv('dsp_quant.csv', index=False)

#%% Plot 2. DSP density
samples = ['gem02', 'gem03', 'gem04', 'gem05', 'gem08', 'gem10', 'wt01', 'wt02', 'wt04', 'wt05', 'wt07', 'wt08']
dsp_total_density = {'gem': [], 'wt': []}

for sample in samples:
    if 'gem' in sample:
        dsp_total_density['gem'].append(np.sum(dsp_density[sample]) / np.sum(tissue_masks[sample]))
    else:
        dsp_total_density['wt'].append(np.sum(dsp_density[sample]) / np.sum(tissue_masks[sample]))

dsp_total_density_df = pd.DataFrame.from_dict(dsp_total_density, orient='index').transpose()
dsp_total_density_df.to_csv('dsp_total_density.csv', index=False)

#%% Plot 3. Pre actin density
samples = ['gem02', 'gem03', 'gem04', 'gem05', 'gem08', 'gem10', 'wt01', 'wt02', 'wt04', 'wt05', 'wt07', 'wt08']
actin_total_density = {'gem day 7': [], 
                       'gem day 9': [],
                       'wt day 7': [],
                       'wt day 9': []}

for sample in samples:
    if 'gem' in sample:
        actin_total_density['gem day 7'].append(np.sum(actin_density[sample]) / np.sum(tissue_masks[sample]))
        actin_total_density['gem day 9'].append(np.sum(post_actin_density[sample]) / np.sum(tissue_masks[sample]))
    else:
        actin_total_density['wt day 7'].append(np.sum(actin_density[sample]) / np.sum(tissue_masks[sample]))
        actin_total_density['wt day 9'].append(np.sum(post_actin_density[sample]) / np.sum(tissue_masks[sample]))

actin_total_density_df = pd.DataFrame.from_dict(actin_total_density, orient='index').transpose()
actin_total_density_df.to_csv('actin_total_density.csv', index=False)