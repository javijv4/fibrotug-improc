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


def clean_mask(mask):
    new_mask = mask > 0.5
    label = measure.label(new_mask)
    prop = measure.regionprops(label)

    if len(prop) == 1:
        return new_mask
    area = [p.area for p in prop]
    ind = np.argmax(area)

    return label == (ind+1)


def mask2mesh(mask, downsample, meshsize=3):
    mask = clean_mask(mask)

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

    if vertex_up_r == 0:
        points_r = boundary[(vertex_dn_r+downsample):-downsample:downsample]
        points_l = boundary[(vertex_up_l+downsample):vertex_dn_l-downsample:downsample]
    elif vertex_up_l == 0:
        points_r = boundary[(vertex_dn_r+downsample):vertex_up_r-downsample:downsample]
        points_l = boundary[downsample:vertex_dn_l-downsample:downsample]
    elif vertex_dn_l == 0:
        points_r = boundary[(vertex_dn_r+downsample):vertex_up_r-downsample:downsample]
        points_l = boundary[(vertex_up_l+downsample):-downsample:downsample]
    elif vertex_dn_r == 0:
        points_r = boundary[downsample:vertex_up_r-downsample:downsample]
        points_l = boundary[vertex_up_l+downsample:vertex_dn_l-downsample:downsample]

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


def project_img2_mesh(mesh, img_data):
    ij_nodes = np.floor(mesh.points).astype(int)

    # Project data to mesh
    for field in img_data.keys():
        f = img_data[field][ij_nodes[:,0], ij_nodes[:,1]]  # TODO if the image stuff is done correctly, then you need to get rid of this
        f[np.isnan(f)] = 0
        mesh.point_data[field] = f

    return mesh


def get_boundary_data(mesh):
    xyz = mesh.points
    ien = mesh.cells[0].data
    xmin = np.min(mesh.points[:,0])
    xmax = np.max(mesh.points[:,0])

    bndry = []
    for i in range(len(ien)):
        k = np.where(np.isclose(xyz[ien[i], 0], xmin))[0]
        j = np.where(np.isclose(xyz[ien[i], 0], xmax))[0]

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

    return bdata


def get_width_at_boundaries(mesh, bdata):
    nodes_b1 = np.unique(bdata[bdata[:,-1]==1, 1:-1])
    nodes_b2 = np.unique(bdata[bdata[:,-1]==2, 1:-1])

    w1 = np.max(mesh.points[nodes_b1,1]) - np.min(mesh.points[nodes_b1,1])
    w2 = np.max(mesh.points[nodes_b2,1]) - np.min(mesh.points[nodes_b2,1])

    return w1, w2


def compute_local_width(tissue_mask, pixel_size):
    xcoord = np.arange(tissue_mask.shape[0])*pixel_size
    width = np.zeros_like(xcoord)
    for i in range(tissue_mask.shape[0]):
        width[i] = np.sum(tissue_mask[i])*pixel_size
        if i == tissue_mask.shape[0]//2:
            inds = np.where(tissue_mask[i]>0.5)[0]
            midpoint = ((inds[-1] - inds[0])/2 + inds[0])*pixel_size

    return xcoord, width, midpoint
