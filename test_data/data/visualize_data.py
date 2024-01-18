#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:27:43 2024

@author: jjv
"""

import meshio as io
import cheartio as chio

mesh = chio.read_mesh('tissue', meshio=True)


to_mesh = ['act_rho', 'act_angles', 'act_disp', 'fib_rho', 'fib_angles', 'fib_disp']

for field in to_mesh:
    mesh.point_data[field] = chio.read_dfile(field + '.FE')
    
io.write('check.vtu', mesh)
