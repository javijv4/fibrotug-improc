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

path = '/Users/jjv/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/fibroTUG/DSP/Tissues/dataset2_2/'
sim_path = '/Users/jjv/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/fibroTUG/DSP/Simulations/3PM/dataset2_2/'
samples = os.listdir(path)
samples = [sample for sample in samples if os.path.isdir(path + sample)]
samples = sorted(samples)



meshsize = 9
pixel_size = 0.390*1e-3  #mm

samples = ['gem02', 'gem03', 'gem04', 'gem05', 'gem08', 'gem10']
for sample in samples:
    tissue_fldr = f'{path}/{sample}/'
    sims_fldr = f'{sim_path}/{sample}/'

    png_dump = tissue_fldr + 'png_dump/'
    exp_fldr = tissue_fldr + 'exp/'
    mesh_fldr = sims_fldr + 'mesh/'
    data_fldr = sims_fldr + 'data/'

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

    # dspexp = DSPProtocolTissue(tissue_fldr, pre_tissue, post_tissue, 'pre', out_fldr=sims_fldr, flip180=True)

    # dspexp.register_to_fixed('pre', mode='affine')
    # dspexp.plot_warped_mask()
    # plt.savefig(png_dump + 'masks.png', dpi=180)
    # dspexp.plot_images()
    # plt.savefig(png_dump + 'images.png')
    # dspexp.plot_images(which=['actin', 'actin_density', 'actin_angle', 'actin_dispersion'])
    # plt.savefig(png_dump + 'actin.png')
    # dspexp.plot_images(which=['fibers', 'fiber_density', 'fiber_angle', 'fiber_dispersion'])
    # plt.savefig(png_dump + 'fiber.png')
    # dspexp.plot_images(which=['dsp', 'dsp_density'])
    # plt.savefig(png_dump + 'dsp.png')
    # dspexp.save_images(exp_fldr)


    # # # One mesh
    # # mesh = dspexp.generate_mesh(meshsize=meshsize, pixel_size=pixel_size,
    # #                             use_fiber_mask=False, subdivide_fibers=False)
    # # io.write('mesh.vtu', mesh)
    # # io.write(mesh_fldr + 'mesh.vtu', mesh)

    # # Separated tissue and fiber meshes
    # # tissue_mesh = dspexp.generate_tissue_mesh(meshsize=meshsize, pixel_size=pixel_size,
    # #                                     use_fiber_mask=False)
    # fiber_mesh, tissue_mesh = dspexp.generate_fiber_mesh(meshsize=meshsize, pixel_size=pixel_size)
    # io.write(mesh_fldr + 'tissue_mesh.vtu', tissue_mesh)
    # print(tissue_mesh)
    # io.write(mesh_fldr + 'fiber_mesh.vtu', fiber_mesh)

    # %% Grab post displacement data
    import cheartio as chio
    day7_data = chio.read_dfile(f'{pre_fldr}/day7_post_disp.INIT')
    day7_ET1_data = chio.read_dfile(f'{pre_fldr}/day7_ET1_post_disp.INIT')
    day7_6hrs_data = chio.read_dfile(f'{pre_fldr}/day7_6hrs_post_disp.INIT')
    day9_data = chio.read_dfile(f'{post_fldr}/day9_post_disp.INIT')

    day7_time, day7_disp = day7_data[:, 0], day7_data[:, 1]
    day7_vel = np.gradient(day7_disp, day7_time)
    day7_ET1_time, day7_ET1_disp = day7_ET1_data[:, 0], day7_ET1_data[:, 1]
    day7_ET1_vel = np.gradient(day7_ET1_disp, day7_ET1_time)
    day7_6hrs_time, day7_6hrs_disp = day7_6hrs_data[:, 0], day7_6hrs_data[:, 1]
    day7_6hrs_vel = np.gradient(day7_6hrs_disp, day7_6hrs_time)
    day9_time, day9_disp = day9_data[:, 0], day9_data[:, 1]
    day9_vel = np.gradient(day9_disp, day9_time)

    save_disp = np.column_stack((day7_time, day7_disp))
    save_vel = np.column_stack((day7_time, day7_vel))
    chio.write_dfile(f'{data_fldr}/day7_post_disp.INIT', save_disp)
    chio.write_dfile(f'{data_fldr}/day7_post_vel.INIT', save_vel)

    save_disp = np.column_stack((day7_ET1_time, day7_ET1_disp))
    save_vel = np.column_stack((day7_ET1_time, day7_ET1_vel))
    chio.write_dfile(f'{data_fldr}/day7_ET1_post_disp.INIT', save_disp)
    chio.write_dfile(f'{data_fldr}/day7_ET1_post_vel.INIT', save_vel)

    save_disp = np.column_stack((day7_6hrs_time, day7_6hrs_disp))
    save_vel = np.column_stack((day7_6hrs_time, day7_6hrs_vel))
    chio.write_dfile(f'{data_fldr}/day7_6hrs_post_disp.INIT', save_disp)
    chio.write_dfile(f'{data_fldr}/day7_6hrs_post_vel.INIT', save_vel)

    save_disp = np.column_stack((day9_time, day9_disp))
    save_vel = np.column_stack((day9_time, day9_vel))
    chio.write_dfile(f'{data_fldr}/day9_post_disp.INIT', save_disp)
    chio.write_dfile(f'{data_fldr}/day9_post_vel.INIT', save_vel)



    # # # import shutil

    # # # shutil.copyfile(f'{pre_fldr}/day7_post_disp.INIT', f'{data_fldr}/day7_post_disp.INIT')
    # # # shutil.copyfile(f'{pre_fldr}/day7_ET1_post_disp.INIT', f'{data_fldr}/day7_ET1_post_disp.INIT')
    # # # shutil.copyfile(f'{pre_fldr}/day7_6hrs_post_disp.INIT', f'{data_fldr}/day7_6hrs_post_disp.INIT')
    # # # shutil.copyfile(f'{post_fldr}/day9_post_disp.INIT', f'{data_fldr}/day9_post_disp.INIT')