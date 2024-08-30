#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:46:35 2023

@author: Javiera Jilberto Vallejos
"""
import numpy as np
from scipy.spatial.distance import cdist

from matplotlib import pyplot as plt

from skimage import measure, morphology, transform, filters
from imregistration.utils import normalize_image

import shapely.geometry as sg
import meshio as io
import pygmsh

from tqdm import tqdm

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
    boundary = np.append(boundary, boundary[0,None], axis=0)
    polygon = sg.Polygon(boundary)
    length = polygon.length
    k = np.round(length/meshsize).astype(int)
    equi_points = np.transpose([polygon.exterior.interpolate(t).xy for t in np.linspace(0,polygon.length,k,False)])
    equi_points = equi_points.squeeze().T

    # mesh
    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(equi_points,
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


def create_submesh(mesh, map_mesh_submesh_elems):
    dim = mesh.points.shape[1]
    submesh_elems = mesh.cells[0].data[map_mesh_submesh_elems]
    submesh_xyz = np.zeros([len(np.unique(submesh_elems)),dim])
    map_mesh_submesh = np.ones(mesh.points.shape[0], dtype=int)*-1
    map_submesh_mesh = np.zeros(submesh_xyz.shape[0], dtype=int)
    child_elems_new = np.zeros(submesh_elems.shape, dtype=int)

    cont = 0
    for e in range(submesh_elems.shape[0]):
        for i in range(submesh_elems.shape[1]):
            if map_mesh_submesh[submesh_elems[e,i]] == -1:
                child_elems_new[e,i] = cont
                submesh_xyz[cont] = mesh.points[submesh_elems[e,i]]
                map_mesh_submesh[submesh_elems[e,i]] = cont
                map_submesh_mesh[cont] = submesh_elems[e,i]
                cont += 1
            else:
                child_elems_new[e,i] = map_mesh_submesh[submesh_elems[e,i]]

    submesh = io.Mesh(submesh_xyz, {mesh.cells[0].type: child_elems_new})
    return submesh, map_mesh_submesh, map_submesh_mesh


def find_boundary(mesh):
    xyz = mesh.points
    ien = mesh.cells[0].data
    minx = np.min(mesh.points[:,0])
    maxx = np.max(mesh.points[:,0])

    bndry = []
    for i in range(len(ien)):
        vertex = xyz[ien[i]]
        k = np.where(np.isclose(vertex[:,0], minx))[0]
        j = np.where(np.isclose(vertex[:,0], maxx))[0]

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


def get_elem_neighbors(ien):
    neigh_elems = np.zeros([len(ien), 3], dtype=int)
    for e in tqdm(range(len(ien))):
        elem_nodes = ien[e]
        neigh = np.where(np.sum(np.isin(ien, elem_nodes), axis=1)==2)[0]
        neighbors = -np.ones(3, dtype=int)
        neighbors[0:len(neigh)] = neigh
        neigh_elems[e] = neighbors

    return neigh_elems


def mask2mesh_with_fibers(tissue_mask, fiber_mask, rescale=4, meshsize=5):
    meshsize = meshsize*rescale

    # Get contour using tissue_mask
    mesh_tissue_mask = tissue_mask.astype(int)
    mesh_tissue_mask = normalize_image(transform.rescale(mesh_tissue_mask, rescale))
    mesh_tissue_mask = morphology.binary_closing(mesh_tissue_mask, footprint=morphology.disk(2))

    mask = clean_mask(mesh_tissue_mask)
    mask[0:10] = 0
    mask[-10:] = 0

    # Find boundary polygon
    mask = np.pad(mask, pad_width=((20,20),(0,0)))
    mask = filters.gaussian(mask) > 0.75
    tissue_mask = mask


    # Find holes
    mesh_fiber_mask = fiber_mask.astype(int)
    mesh_fiber_mask = normalize_image(transform.rescale(mesh_fiber_mask, rescale))

    # Generate mesh and grab node coordinates
    mask = clean_mask(mesh_fiber_mask)

    # Find boundary polygon
    mask = np.pad(mask, pad_width=((20,20),(0,0)))
    mask[~tissue_mask] = 0
    mask += ~tissue_mask
    mask = filters.gaussian(mask) > 0.5
    boundaries = measure.find_contours(~mask)

    # Tissue boundary
    tissue_mask = morphology.binary_erosion(tissue_mask, footprint=morphology.disk(2))
    boundary = measure.find_contours(tissue_mask)[0]
    polygon = sg.Polygon(boundary)
    length = polygon.length
    k = np.round(length/meshsize).astype(int)
    equi_points = np.transpose([polygon.exterior.interpolate(t).xy for t in np.linspace(0,polygon.length,k,False)])
    equi_points = equi_points.squeeze().T
    equi_points[:,0] -= 20.5
    tissue_equi_points = equi_points


    holes = []
    len_holes = []
    for boundary in boundaries:

        boundary = np.append(boundary, boundary[0,None], axis=0)
        polygon = sg.Polygon(boundary)
        length = polygon.length
        k = np.round(length/meshsize).astype(int)
        if k < 4:
            continue
        equi_points = np.transpose([polygon.exterior.interpolate(t).xy for t in np.linspace(0,polygon.length,k,False)])
        equi_points = equi_points.squeeze().T
        equi_points[:,0] -= 20.5
        # equi_points = np.append(equi_points, equi_points[0,None], axis=0)
        holes.append(equi_points)
        len_holes.append(len(equi_points))

    # Order holes based on len_holes
    holes = [hole for _, hole in sorted(zip(len_holes, holes), key=lambda x: x[0])][::-1]

    # mesh
    with pygmsh.occ.Geometry() as geom:

        borders = []
        for hole in tqdm(holes):
            borders.append(geom.add_polygon(hole,
                mesh_size=meshsize,
            ))

        # main surface
        tissue = geom.add_polygon(tissue_equi_points,
            mesh_size=meshsize
        )

        # # Cutout the fiber boundary
        cuts = []
        for border in tqdm(borders):
            cuts.append(geom.boolean_intersection([tissue, border], delete_first=False)[0])

        geom.boolean_fragments(tissue, cuts)

        mesh = geom.generate_mesh()

    points = mesh.points[:,0:2]
    tri_mesh = io.Mesh(points, {'triangle': mesh.cells_dict['triangle']})

    # Clean bad elements
    xyz = tri_mesh.points
    ien = tri_mesh.cells_dict['triangle']
    _, bfaces = get_surface_mesh(tri_mesh)

    xmin = np.min(tri_mesh.points[:,0])
    xmax = np.max(tri_mesh.points[:,0])
    post1_faces = np.where(np.all(np.isclose(xyz[bfaces][:,:,0], xmin), axis=1))[0]
    post2_faces = np.where(np.all(np.isclose(xyz[bfaces][:,:,0], xmax), axis=1))[0]
    post_nodes = np.union1d(bfaces[post1_faces], bfaces[post2_faces])

    del_elems = 100

    it = 1
    while del_elems > 0:
        xyz, ien, del_elems = fix_bad_elements(xyz, ien, post_nodes)
        print('Iteration:', it, 'Deleted elements:', del_elems)
        it += 1


    # Obtaining mask of fiber elements
    midpoints = np.mean(xyz[ien], axis=1)
    inholes = np.zeros(len(midpoints))
    spoints = [sg.Point(point) for point in midpoints]

    print('Finding fiber elems')
    for hole in tqdm(holes):
        poly = sg.polygon.Polygon(hole)
        marker = poly.contains(spoints)
        inholes[marker] = 1

    elem_fiber_mask = 1-inholes

    xyz = xyz/rescale
    tri_mesh = io.Mesh(xyz, {'triangle': ien})

    # Check for isolated cells
    print('Finding fiber_mesh neighbors')
    fiber_elems = np.where(elem_fiber_mask==1)[0]
    elem_neigh = get_elem_neighbors(tri_mesh.cells[0].data[fiber_elems])
    non_neigh = np.sum(elem_neigh==-1, axis=1)
    isolated_cells = fiber_elems[np.where(non_neigh==3)[0]]

    print(isolated_cells)
    for e in isolated_cells:
        elem_fiber_mask[e] = 0

    # Check cells with one neighbor
    one_neigh = np.sum(elem_neigh==-1, axis=1)
    one_neigh_cells = np.where(one_neigh==2)[0]

    for e in one_neigh_cells:
        neigh = elem_neigh[e, elem_neigh[e] >= 0][0]
        neigh_neigh = elem_neigh[neigh]

        if np.sum(neigh_neigh == -1) == 2: # two isolated cells
            elem_fiber_mask[fiber_elems[e]] = 0

    # Create fiber mesh
    fiber_mesh, map_mesh_fiber, map_fiber_mesh = create_submesh(tri_mesh, np.where(elem_fiber_mask==1)[0])



    return tri_mesh, elem_fiber_mask, fiber_mesh

def get_surface_mesh(mesh):
    ien = mesh.cells[0].data

    array = np.array([[0,1],[1,2],[2,0]])
    nelems = np.repeat(np.arange(ien.shape[0]),3)
    faces = np.vstack(ien[:,array])
    sort_faces = np.sort(faces,axis=1)

    f, i, c = np.unique(sort_faces, axis=0, return_counts=True, return_index=True)
    ind = i[np.where(c==1)[0]]
    bfaces = faces[ind]
    belem = nelems[ind]

    return belem, bfaces


def tri_quality_aspect_ratio(xyz, ien):
    if xyz.shape[1] == 2:
        xyz = np.concatenate([xyz, np.zeros((xyz.shape[0], 1))], axis=1)
    points = xyz[ien]
    assert points.shape[1] == 3 and points.shape[2] == 3

    l0 = points[:,1,:] - points[:,0,:]
    l1 = points[:,2,:] - points[:,1,:]
    l2 = points[:,0,:] - points[:,2,:]

    l0_length = np.linalg.norm(l0, axis=1)
    l1_length = np.linalg.norm(l1, axis=1)
    l2_length = np.linalg.norm(l2, axis=1)

    lmax = np.max([l0_length, l1_length, l2_length], axis=0)

    area = 0.5 * np.linalg.norm(np.cross(l0, l1, axisa=1, axisb=1), axis=1)

    r = 2 * area / np.sum([l0_length, l1_length, l2_length], axis=0)

    return lmax/(2*np.sqrt(3)*r)


def fix_bad_elements(xyz, ien, post_nodes):
    quality = tri_quality_aspect_ratio(xyz, ien)

    to_fix = np.where(quality > 10)[0]

    elem_del = []
    for i in to_fix:
        nodes = ien[i]
        points = xyz[nodes]
        dist = cdist(points, points)
        unique_dist = np.unique(dist[dist>0])

        if len(np.unique(nodes)) < 3:
            elem_del.append(i)

        # Case 1: sliver
        elif len(unique_dist) > 1:
            to_del = nodes[np.where(dist==np.min(unique_dist))[0]]

            ispost = np.isin(to_del, post_nodes)
            if np.all(~ispost) or np.all(ispost):

                new_point = np.mean(xyz[to_del], axis=0)

                node_number = np.min(to_del)
                xyz[node_number] = new_point

                del_node = np.max(to_del)
                ien[ien==del_node] = node_number
                ien[ien>del_node] -= 1

                xyz = np.delete(xyz, [del_node], axis=0)
                elem_del.append(i)

            else:
                node_number = to_del[ispost]
                del_node = to_del[~ispost]
                ien[ien==del_node] = node_number
                ien[ien>del_node] -= 1
                xyz = np.delete(xyz, [del_node], axis=0)
                elem_del.append(i)
        else:
            print(i)


    ien = np.delete(ien, elem_del, axis=0)
    return xyz, ien, len(elem_del)


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
