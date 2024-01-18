#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:46:35 2023

@author: Javiera Jilberto Vallejos
"""
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure, morphology, transform, filters
import meshio as io
import pygmsh


def mask2mesh(mask, downsample, meshsize=3):
    # Find boundary polygon
    mask = np.pad(mask, pad_width=((20,20),(0,0)))
    mask = filters.gaussian(mask) > 0.75
    boundary = measure.find_contours(mask)[0]
    boundary[:,0] = boundary[:,0] - np.min(boundary[:,0]) + 0.5
    length = np.max(boundary[:,0])-np.min(boundary[:,0])
    bounds_dn = np.where(np.isclose(boundary[:,0],0.5))[0]
    bounds_up = np.where(np.isclose(boundary[:,0],length+0.5))[0]
    vertex_dn_l = bounds_dn[np.argmin(boundary[bounds_dn,1])]
    vertex_dn_r = bounds_dn[np.argmax(boundary[bounds_dn,1])]
    vertex_up_l = bounds_up[np.argmin(boundary[bounds_up,1])]
    vertex_up_r = bounds_up[np.argmax(boundary[bounds_up,1])]

    corners = np.array([vertex_dn_l,vertex_dn_r,vertex_up_r,vertex_up_l])

    points_r = boundary[(vertex_dn_r+downsample):vertex_up_r-downsample:downsample]
    points_l = boundary[(vertex_up_l+downsample)::downsample]

    poly = np.vstack([boundary[vertex_dn_l], boundary[vertex_dn_r],
                    points_r,
                    boundary[vertex_up_r], boundary[vertex_up_l],
                    points_l])

    # mesh
    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(poly,
            mesh_size=meshsize,
        )
        mesh = geom.generate_mesh()

    points = mesh.points[:,0:2]
    tri_mesh = io.Mesh(points, {'triangle': mesh.cells_dict['triangle']})

    return tri_mesh