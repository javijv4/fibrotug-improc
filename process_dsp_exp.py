#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 08:54:46 2024z

@author: Javiera Jilberto Vallejos
"""

import os
import numpy as np
from FtugTissue import FtugTissue, DSPProtocolTissue
import matplotlib.pyplot as plt
import meshio as io
from skimage import io as skio

tissue_fldr = '../DSP/Tissues/dataset2/gem02/'
png_dump = tissue_fldr + 'png_dump/'
exp_fldr = tissue_fldr + 'exp/'
mesh_fldr = tissue_fldr + 'mesh/'
data_fldr = tissue_fldr + 'data/'

downsample = 10
meshsize = 5
pixel_size = 0.390*1e-3  #um

pre_fldr = tissue_fldr + 'day7/'
pre_images = {'dsp': 'A_0.41_0.1_GEM12_s1_day7-02_MAX_c1_ORG.tif',
          'fibers': 'A_0.41_0.1_GEM12_s1_day7-02_MAX_c2_ORG.tif',
          'actin': 'A_0.41_0.1_GEM12_s1_day7-02_MAX_c3_ORG.tif'}


post_fldr = tissue_fldr + 'day9/'
post_images = {'dsp': 'A_0.41_0.1_GEM12_s1_day9-02_MAX_c2_ORG.tif',
            'fibers': 'A_0.41_0.1_GEM12_s1_day9-02_MAX_c3_ORG.tif',
            'actin': 'A_0.41_0.1_GEM12_s1_day9-02_MAX_c4_ORG.tif'}


# Creating folders
if not os.path.exists(png_dump):
    os.makedirs(png_dump)
if not os.path.exists(mesh_fldr):
    os.makedirs(mesh_fldr)
if not os.path.exists(data_fldr):
    os.makedirs(data_fldr)
if not os.path.exists(exp_fldr):
    os.makedirs(exp_fldr)

pre_tissue = FtugTissue(pre_fldr, pre_images)
post_tissue = FtugTissue(post_fldr, post_images)

dspexp = DSPProtocolTissue(tissue_fldr, pre_tissue, post_tissue, 'pre', flip180=True)

dspexp.register_to_fixed('pre', mode='affine')
dspexp.plot_warped_mask()
plt.savefig(png_dump + 'masks.png', dpi=180)
dspexp.plot_images()
plt.savefig(png_dump + 'images.png')
dspexp.plot_images(which=['actin', 'actin_density', 'actin_angle', 'actin_dispersion'])
plt.savefig(png_dump + 'actin.png')
dspexp.plot_images(which=['fibers', 'fiber_density', 'fiber_angle', 'fiber_dispersion'])
plt.savefig(png_dump + 'fiber.png')
dspexp.plot_images(which=['dsp', 'dsp_density'])
plt.savefig(png_dump + 'dsp.png')

mesh = dspexp.generate_mesh(downsample=downsample, meshsize=meshsize, pixel_size=pixel_size)
io.write(mesh_fldr + 'mesh.vtu', mesh)

# Save tissue mask
skio.imsave(exp_fldr + 'tissue_mask.tif', dspexp.mesh_tissue_mask)

