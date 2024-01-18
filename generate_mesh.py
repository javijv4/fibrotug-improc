#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:46:35 2023

@author: Javiera Jilberto Vallejos
"""
import numpy as np
import os
from src.imregistration.utils import rotate_crop
from src.imregistration.ITKTransform import imageToArray
from src.img2mesh import mask2mesh
import meshio as io
import cheartio as chio

downsample = 10
meshsize = 5
pixel_size = 0.390*1e-3  #um

# Path to the masks. Results will be saved in tissue_folder
tissue_fldr = 'test_data/exp/'
cheart_fldr = 'test_data/mesh/'
mask_file = tissue_fldr + 'fibrotug_mask'
act_file = tissue_fldr + 'pre_act'
fib_file = tissue_fldr + 'post_fib'
ext = '.tif'

info = np.load(tissue_fldr + 'transformation.npz')
rotation = info['rotation']
box = info['box']

post_act_data = np.load(tissue_fldr + 'improc_post_actin.npz')
act_data = np.load(tissue_fldr + 'improc_pre_actin.npz')
fib_data = np.load(tissue_fldr + 'improc_post_fiber.npz')

# Read images and data rotate/crop
ftug_mask = imageToArray(mask_file + ext)
act_img = imageToArray(act_file + ext)
fib_img = imageToArray(fib_file + ext)
img_data = {
        'act_img' : act_img,
        'fib_img' : fib_img,
        'tissue_mask' : ftug_mask,
        'act_angles' : act_data['angles'],
        'act_mask' : act_data['mask'],
        'act_rho' : act_data['density'],
        'act_disp' : act_data['dispersion'],
        'post_act_angles' : post_act_data['angles'],
        'post_act_mask' : post_act_data['mask'],
        'post_act_rho' : post_act_data['density'],
        'post_act_disp' : post_act_data['dispersion'],
        'fib_angles' : fib_data['angles'],
        'fib_mask' : fib_data['mask'],
        'fib_rho' : fib_data['density'],
        'fib_disp' : fib_data['dispersion'],
        }
to_mesh = ['act_rho', 'act_angles', 'act_disp', 'post_act_rho', 'post_act_angles', 'post_act_disp', 'fib_rho', 'fib_angles', 'fib_disp']

# Crop images
img_data = rotate_crop(img_data, rotation, box)
tissue_mask = img_data['tissue_mask']

# Generate mesh
mesh = mask2mesh(tissue_mask, downsample, meshsize=meshsize)
ij_nodes = np.floor(mesh.points).astype(int)

# Project data to mesh
for field in to_mesh:
    mesh.point_data[field] = img_data[field][ij_nodes[:,0], ij_nodes[:,1]]

# Get vectors
act_angles = mesh.point_data['act_angles']
act_f = np.vstack([np.cos(act_angles), np.sin(act_angles)]).T
act_s = np.vstack([act_f[:,1], -act_f[:,0]]).T
mesh.point_data['act_f'] = np.hstack([act_f, np.zeros([len(act_f),1])])
save_act_f = np.hstack([act_f, act_s])

fib_angles = mesh.point_data['fib_angles']
fib_f = np.vstack([np.cos(fib_angles), np.sin(fib_angles)]).T
fib_s = np.vstack([fib_f[:,1], -fib_f[:,0]]).T
mesh.point_data['fib_f'] = np.hstack([fib_f, np.zeros([len(fib_f),1])])
save_fib_f = np.hstack([fib_f, fib_s])

# Fiber rho to cells
midpoints = np.mean(mesh.points[mesh.cells[0].data], axis=1)
ij_nodes = np.floor(midpoints).astype(int)
elem_fib_rho = img_data['fib_rho'][ij_nodes[:,0], ij_nodes[:,1]]
mesh.cell_data['fib_elem_rho'] = [elem_fib_rho]

io.write(cheart_fldr + 'mesh.vtu', mesh)

# Save to CH
if not os.path.exists(cheart_fldr): os.mkdir(cheart_fldr)

# Deal with mesh
mesh.points[:,0] = mesh.points[:,0] - np.min(mesh.points[:,0])

# Find boundary
xyz = mesh.points
ien = mesh.cells[0].data
length = np.max(mesh.points[:,0])

bndry = []
for i in range(len(ien)):
    k = np.where(xyz[ien[i], 0] < 0 + 1e-10)[0]
    j = np.where(xyz[ien[i], 0] > length - 1e-10)[0]

    if len(k) == 0 and len(j) == 0:
        continue

    if len(k) == 2:
        v = ien[i, k]
        id_value = 1
    elif len(j) == 2:
        v = ien[i, j]
        id_value = 2
    else:
        continue

    bndry.append(np.array([i, v[0], v[1], id_value]))

bdata = np.vstack(bndry).astype(int)

# rescale with pixel size
mesh.points = mesh.points * pixel_size

chio.write_mesh(cheart_fldr + 'tissue', mesh.points, mesh.cells[0].data)
chio.write_bfile(cheart_fldr + 'tissue', bdata)

for field in to_mesh:
    chio.write_dfile(cheart_fldr + field + '.FE', mesh.point_data[field])

chio.write_dfile(cheart_fldr + 'fiber_orientation.FE', save_fib_f)
chio.write_dfile(cheart_fldr + 'act_orientation.FE', save_act_f)
chio.write_dfile(cheart_fldr + 'elem_fiber_rho.FE', elem_fib_rho)