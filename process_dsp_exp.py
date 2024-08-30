#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 08:54:46 2024z

@author: Javiera Jilberto Vallejos
"""

import os
import numpy as np
from FtugTissue import FtugTissue, DSPProtocolTissue, find_images
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
pixel_size = 0.390*1e-3  #mm

pre_fldr = tissue_fldr + 'day7/'
pre_images = find_images(pre_fldr)


post_fldr = tissue_fldr + 'day9/'
post_images = find_images(post_fldr)


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
dspexp.save_images(exp_fldr)

mesh = dspexp.generate_mesh(downsample=downsample, meshsize=meshsize, pixel_size=pixel_size,
                            use_fiber_mask=True)
io.write('mesh.vtu', mesh)
io.write(mesh_fldr + 'mesh.vtu', mesh)


