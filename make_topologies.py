#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:59:00 2024

@author: Javiera Jilberto Vallejos
"""
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

seed = 19910
np.random.seed(seed)
pixel_size = 0.390*1e-3  #um
elongation_factor = 5

tissue_fldr = '/home/jilberto/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/fibroTUG/DSP/fibrotug-dsp-sims/3P/Tissues/gem10/'
print(tissue_fldr)
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
bk = skio.imread(exp_fldr + 'pre_tissue_mask.tif')
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


# Assign elements to cells
if use_utc:
    axis_dir, trans_dir = gt.get_utc_images(bk)
    ijk = np.round(xyz/pixel_size).astype(int)
    utc = gt.get_utc(ijk, axis_dir, trans_dir)
    utc_cells = gt.get_utc(ijcells, axis_dir, trans_dir)

    cell_number = gt.assign_elements_to_cells(utc_cells, utc, ien, elongation_factor/2.5)

else:
    cell_number = gt.assign_elements_to_cells(xycells, xyz, ien, elongation_factor)

# Change dimensions to mesh
xycells *= pixel_size

# Plotting mesh and cell centers
plt.figure(2, clear=True)
plt.triplot(triang)
plt.plot(xycells[:,0], xycells[:,1], 'k.', ms=2)
plt.gca().set_aspect('equal')
plt.axis('off')
plt.savefig(png_dump + 'mesh_cell_centers.png', dpi=180, bbox_inches='tight', pad_inches=0)

# Checking for isolated cells
neigh_elems, neigh_cells = gt.get_elem_neighbors(ien, cell_number)
cell_number, neigh_cells = gt.find_isolated_cells(cell_number, neigh_elems, neigh_cells)

# Save cell number
mesh.cell_data['cell_number'] = [cell_number]
chio.write_dfile(data_fldr + 'cell_number.FE', cell_number)
io.write(data_fldr + 'cells.vtu', mesh)

# Plotting mesh with cell numbers
plt.figure(1, clear=True)
# plt.imshow(image, cmap='binary_r')
plt.tripcolor(triang, cell_number, cmap='jet', alpha=1, edgecolor='none')
plt.plot(xycells[:,0], xycells[:,1], 'k.', ms=2)
plt.gca().set_aspect('equal')
plt.axis('off')
plt.savefig(png_dump + 'mesh_cell_numbers.png', dpi=180, bbox_inches='tight', pad_inches=0)


#%%
# Read DSP density
dsp_rho = chio.read_dfile(data_fldr + 'pre_dsp_density.FE')

# Write tissue mesh
chio.write_mesh(mesh_fldr + 'tissue', mesh.points, mesh.cells[0].data)
chio.write_bfile(mesh_fldr + 'tissue', bdata)

# Generating mesh for pressure
disc_mesh, disc_bdata, disc_mesh_map = gt.generate_fully_disc_mesh(mesh, cell_number)
chio.write_mesh(mesh_fldr + 'tissue_disc', disc_mesh.points, disc_mesh.cells[0].data)
chio.write_bfile(mesh_fldr + 'tissue_disc', disc_bdata)
io.write(data_fldr + 'disc_mesh.vtu', disc_mesh)

# Generating cell mesh
if use_dsp_rho:
    cell_mesh, connected_nodes, cell_mesh_map = gt.generate_connected_disc_mesh(mesh, cell_number, dsp_density=dsp_rho)
else:
    cell_mesh, connected_nodes, cell_mesh_map = gt.generate_connected_disc_mesh(mesh, cell_number)
cell_mesh.cell_data['cell_number'] = [cell_number]

io.write(data_fldr + 'cell_mesh.vtu', cell_mesh)

chio.write_mesh(mesh_fldr + 'tissue_cell', cell_mesh.points, cell_mesh.cells[0].data)
chio.write_bfile(mesh_fldr + 'tissue_cell', bdata)

# To visualize
cell_boundary_mesh = gt.get_boundary_mesh(mesh, connected_nodes, disc_mesh)
io.write(data_fldr + 'cell_bdry_mesh.vtu', cell_boundary_mesh)
