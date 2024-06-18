#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:17:43 2024

@author: Javiera Jilberto Vallejos
"""

import meshio as io
import cheartio as chio
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import random


tissue_fldr = '/home/jilberto/Dropbox (University of Michigan)/Projects/fibroTUG/DSP/Tissues/dataset2/gem02/'
mesh = chio.read_mesh(tissue_fldr + 'mesh/tissue', meshio=True)
cell_number = chio.read_dfile(tissue_fldr + 'data/cell_number.FE')
fibelem = chio.read_dfile(tissue_fldr + 'data/fiber_density_elem.FE')
sarc_rho = chio.read_dfile(tissue_fldr + 'data/pre_actin_density.FE')
sarcelem = np.mean(sarc_rho[mesh.cells[0].data], axis=1)

cell_labels = np.unique(cell_number)
connections = np.zeros(len(cell_number))
for e in range(len(cell_labels)):
    cell_elems = np.where(cell_number==e)[0]
    nelems = len(cell_elems)
    cell_fib = fibelem[cell_elems]
    cell_sarc = sarcelem[cell_elems]

    candidate_elems = cell_elems[(cell_fib>0.8)*(cell_sarc>0.8)]
    print(len(candidate_elems))
    n = np.max([int(nelems/10), int(len(candidate_elems)/5)])

    if len(candidate_elems) < n:
        connections[candidate_elems] = 1
        print('Cell {} has less than 10% of the elements with high fiber and actin density'.format(e))
    else:
        celems = random.sample(list(candidate_elems), n)
        connections[celems] = 1
        print('Cell {} has {} elements with high fiber and actin density'.format(e, n))


# xyz = mesh.points
# ien = mesh.cells[0].data

# minx = np.min(xyz[:,0])
# L = np.max(xyz[:,0])-minx

# freq = 8

# x = np.linspace(0,1,1000)
# cosx = 1-(np.cos(x*(2*np.pi)*freq)+1)/2
# fx = (cosx > 0.95).astype(float)

# x = x*L + minx
# connect_func = interp1d(x, fx)

# midpoints = np.mean(xyz[ien], axis=1)
# connect = (connect_func(midpoints[:,0])*(fibelem > 0.8)*(sarcelem > 0.8)) > 0.5
# connect = connect.astype(float)
chio.write_dfile(tissue_fldr + 'data/connect.FE', connections)
# chio.write_dfile(tissue_fldr + 'data/sarc_rho_elem.INIT', sarcelem)

mesh.cell_data['connect'] = [connections]
mesh.cell_data['sarc_rho'] = [sarcelem]
mesh.cell_data['fibelem'] = [fibelem]
io.write(tissue_fldr + 'data/cell_matrix.vtu', mesh)

# plt.figure(1, clear=True)
# plt.plot(x, fx)
# plt.savefig('check.png')