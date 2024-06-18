#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  18 09:46:35 2023

@author: Javiera Jilberto Vallejos
"""
import os
import re
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

tissue_fldr = '/home/jilberto/Dropbox (University of Michigan)/Projects/fibroTUG/DSP/Tissues/dataset2/elk01/day7/'

files = os.listdir(tissue_fldr)

names = []
batch_index = []
sample_index = []
imgs = []
for file in files:
    lst = re.split('_|-', file)
    lst[-1] = lst[-1].split('.')[0]

    # TODO sample image checker
    if 'ELK' in lst:
        names.append('ELK')
    elif 'wt' in lst:
        names.append('WT')
    elif 'GEM' in lst:
        names.append('GEM')

    # Load images
    imgs.append(io.imread('{}/{}'.format(tissue_fldr, file)))


print(len(imgs))


videos = []
images = []
for i, img in enumerate(imgs):

    shape = img.shape
    if len(shape) > 2:
        if shape[2] == 3:  # RGB
            all_channels = img
        else:
            videos.append(img)
    else:
        images.append(img)

plot_size = np.max([len(videos), len(images)])

fig, axs = plt.subplots(2, plot_size+1, num=1, clear=True)
axs[0,0].imshow(all_channels)
for i, img in enumerate(images):
    axs[0,i+1].imshow(img, cmap='binary_r')

for i, img in enumerate(videos):
    axs[1,i+1].imshow(img[0], cmap='binary_r')

for ax in axs.flatten():
    ax.axis('off')