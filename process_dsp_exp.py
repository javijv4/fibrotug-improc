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

path = '/home/jilberto/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/fibroTUG/DSP/Tissues/dataset2_2/'
sim_path = '/home/jilberto/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/fibroTUG/DSP/Simulations/dataset2_2/'
samples = os.listdir(path)
samples = [sample for sample in samples if os.path.isdir(path + sample)]
samples = sorted(samples)

samples = ['gem08']
for sample in samples:
    if 'gem' not in sample: continue
    # if sample == 'gem02': continue
    # if sample == 'gem03': continue
    # if sample == 'gem04': continue
    # if sample == 'gem05': continue
    print(sample)
    tissue_fldr = f'{path}/{sample}/'
    sims_fldr = f'{sim_path}/{sample}/'

    png_dump = tissue_fldr + 'png_dump/'
    exp_fldr = tissue_fldr + 'exp/'
    mesh_fldr = sims_fldr + 'mesh/'
    data_fldr = sims_fldr + 'data/'

    meshsize = 5
    pixel_size = 0.390*1e-3  #mm

    pre_fldr = tissue_fldr + 'pre/'
    pre_images = find_images(pre_fldr)


    post_fldr = tissue_fldr + 'post/'
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

    dspexp = DSPProtocolTissue(tissue_fldr, pre_tissue, post_tissue, 'pre', out_fldr=sims_fldr, flip180=True)

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


    # One mesh
    mesh = dspexp.generate_mesh(meshsize=meshsize, pixel_size=pixel_size,
                                use_fiber_mask=True, subdivide_fibers=False)
    io.write('mesh.vtu', mesh)
    io.write(mesh_fldr + 'mesh.vtu', mesh)

    # # Separated tissue and fiber meshes
    # tissue_mesh = dspexp.generate_tissue_mesh(meshsize=meshsize, pixel_size=pixel_size,
    #                                     use_fiber_mask=False)
    # io.write(mesh_fldr + 'tissue_mesh.vtu', tissue_mesh)
    # fiber_mesh = dspexp.generate_fiber_mesh(meshsize=meshsize*2, pixel_size=pixel_size)
    # io.write(mesh_fldr + 'fiber_mesh.vtu', fiber_mesh)

    # #%% Grab post displacement data
    import shutil

    shutil.copyfile(f'{pre_fldr}/day7_post_disp.INIT', f'{data_fldr}/day7_post_disp.INIT')
    shutil.copyfile(f'{pre_fldr}/day7_ET1_post_disp.INIT', f'{data_fldr}/day7_ET1_post_disp.INIT')
    shutil.copyfile(f'{pre_fldr}/day7_6hrs_post_disp.INIT', f'{data_fldr}/day7_6hrs_post_disp.INIT')
    shutil.copyfile(f'{post_fldr}/day9_post_disp.INIT', f'{data_fldr}/day9_post_disp.INIT')