#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/07/01 17:13:22

@author: Javiera Jilberto Vallejos
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.io import loadmat
from scipy.ndimage import distance_transform_edt
import meshio as io
import cheartio as chio
import phasegen.gen_topologies as gt
from skimage import io as skio

np.random.seed(1992)
pixel_size = 0.390*1e-3  #um
elongation_factor = 2

tissue_fldr = '/home/jilberto/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/fibroTUG/DSP/fibrotug-dsp-sims/3P/Tissues/gem02/'
cell_generation = 'random'
use_utc = True
use_dsp_rho = False

mesh_fldr = tissue_fldr + 'mesh/'
data_fldr = tissue_fldr + 'data/'
exp_fldr = tissue_fldr + 'exp/'
png_dump = tissue_fldr + 'png_dump/'

# Read mesh
mesh = chio.read_mesh(mesh_fldr + 'tissue', meshio=True)
bdata = chio.read_bfile(mesh_fldr + 'tissue')

# # Clean mesh folder
# for filename in os.listdir(mesh_fldr):
#     file_path = os.path.join(mesh_fldr, filename)
#     os.remove(file_path)

# Seed nuclei
bk = skio.imread(exp_fldr + 'tissue_mask.tif')

i = np.arange(bk.shape[0])
j = np.arange(bk.shape[1])
I, J = np.meshgrid(j, i)

axis_dir = J/np.max(J)
trans_dir = np.zeros_like(axis_dir)
for i in range(J.shape[0]):
    lims = np.where(bk[i,:]>0)[0][np.array([0,-1])]
    size = lims[1]-lims[0]
    trans_dir[i] = (I[i]-lims[0])/size

trans_dir[trans_dir < 0] = 0
trans_dir[trans_dir > 1] = 1

fig, axs = plt.subplots(1,2, figsize=(10,5), num=0, clear=True)
axs[0].imshow(np.round(axis_dir*10))
axs[1].imshow(np.round(trans_dir*10))

bk_dist = distance_transform_edt(bk)

if cell_generation == 'random':
    xycells = gt.seed_random_cells(bk, bk_dist)
elif cell_generation == 'image':
    xycells = np.load('cell_centers.npy')
ijcells = np.round(xycells).astype(int)
bk_cells = bk[ijcells[:,0], ijcells[:,1]]
xycells = xycells[bk_cells>0]

# Plotting cell centers
plt.figure(1, clear=True)
plt.imshow(bk, alpha=0.5)
plt.plot(xycells[:,1], xycells[:,0], 'o')
plt.axis('off')
plt.savefig(png_dump + 'img_cell_centers.png', dpi=180, bbox_inches='tight', pad_inches=0)


# Loading mesh
xyz = mesh.points
ien = mesh.cells[0].data
triang = tri.Triangulation(xyz[:,0], xyz[:,1], ien)

ijk = np.round(xyz/pixel_size).astype(int)
axis = axis_dir[ijk[:,0], ijk[:,1]]
trans = trans_dir[ijk[:,0], ijk[:,1]]
utc = np.vstack([axis, trans]).T
utc_triang = tri.Triangulation(utc[:,0], utc[:,1], ien)

axis_cells = axis_dir[ijcells[:,0], ijcells[:,1]]
trans_cells = trans_dir[ijcells[:,0], ijcells[:,1]]
utc_cells = np.vstack([axis_cells, trans_cells]).T

# Plotting mesh and cell centers
plt.figure(2, clear=True)
plt.triplot(utc_triang)
plt.plot(utc_cells[:,0], utc_cells[:,1], 'k.', ms=2)
plt.gca().set_aspect('equal')
plt.axis('off')
plt.savefig(png_dump + 'mesh_cell_centers.png', dpi=180, bbox_inches='tight', pad_inches=0)

# Assign elements to cells
cell_number = gt.assign_elements_to_cells(utc_cells, utc, ien, elongation_factor)


# Change dimensions to mesh
xycells *= pixel_size

# Plotting mesh with cell numbers
plt.figure(1, clear=True)
# plt.imshow(image, cmap='binary_r')
plt.tripcolor(triang, cell_number, cmap='jet', alpha=1, edgecolor='none')
plt.plot(xycells[:,0], xycells[:,1], 'k.', ms=2)
plt.gca().set_aspect('equal')
plt.axis('off')
plt.savefig(png_dump + 'mesh_cell_numbers.png', dpi=180, bbox_inches='tight', pad_inches=0)