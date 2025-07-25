#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:46:35 2023

@author: Javiera Jilberto Vallejos
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from matplotlib import pyplot as plt

from skimage import measure, morphology, transform, filters
from imregistration.utils import normalize_image

import shapely.geometry as sg
from shapelysmooth import taubin_smooth

import meshio as io
import pygmsh
import trimesh
from trimesh.geometry import faces_to_edges
from trimesh.grouping import unique_rows

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


def add_posts_to_mask(mask):
    pass

def mask2mesh(mask, meshsize=3, add_post=False):
    mask = clean_mask(mask)
    if add_post:
        mask = add_posts_to_mask(mask)

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
    xyz = tri_mesh.points

    return tri_mesh

def mask2mesh_rect(mask, meshsize=3, add_post=False):
    mask = clean_mask(mask)
    if add_post:
        mask = add_posts_to_mask(mask)

    # Find boundary polygon
    mask = np.pad(mask, pad_width=((20,20),(0,0)))
    props = measure.regionprops(mask.astype(int))[0]
    bbox = props.bbox
    corners = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]], [bbox[2], bbox[1]]])
    corners = corners.astype(float)
    corners[:,0] = corners[:,0] - np.min(corners[:,0]) + 0.5

    # mesh
    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(corners,
            mesh_size=meshsize,
        )
        mesh = geom.generate_mesh()

    points = mesh.points[:,0:2].copy()
    tri_mesh = io.Mesh(points, {'triangle': mesh.cells_dict['triangle']})
    xyz = tri_mesh.points
    length = np.max(xyz[:,0]) - np.min(xyz[:,0])
    xyz[:,0] = (xyz[:,0]-np.min(xyz[:,0]))/length*(length-1)+0.5
    trimesh.points = xyz

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

def get_surface_mesh(mesh):
    ien = mesh.cells[0].data

    if ien.shape[1] == 3:   # Assuming triangle
        array = np.array([[0,1],[1,2],[2,0]])
        nelems = np.repeat(np.arange(ien.shape[0]),3)
        faces = np.vstack(ien[:,array])
        sort_faces = np.sort(faces,axis=1)

        f, i, c = np.unique(sort_faces, axis=0, return_counts=True, return_index=True)
        ind = i[np.where(c==1)[0]]
        bfaces = faces[ind]
        belem = nelems[ind]


    elif ien.shape[1] == 4:   # Assuming tetra
        array = np.array([[0,1,2],[1,2,3],[0,1,3],[2,0,3]])
        nelems = np.repeat(np.arange(ien.shape[0]),4)
        faces = np.vstack(ien[:,array])
        sort_faces = np.sort(faces,axis=1)

        f, i, c = np.unique(sort_faces, axis=0, return_counts=True, return_index=True)
        ind = i[np.where(c==1)[0]]
        bfaces = faces[ind]
        belem = nelems[ind]

    elif ien.shape[1] == 27:   # Assuming hex27
        array = np.array([[0,1,5,4,8,17,12,16,22],
                          [1,2,6,5,9,18,13,17,21],
                          [2,3,7,6,10,19,14,18,23],
                          [3,0,4,7,11,16,15,19,20],
                          [0,1,2,3,8,9,10,11,24],
                          [4,5,6,7,12,13,14,15,25]])
        nelems = np.repeat(np.arange(ien.shape[0]),6)
        faces = np.vstack(ien[:,array])
        sort_faces = np.sort(faces,axis=1)

        f, i, c = np.unique(sort_faces, axis=0, return_counts=True, return_index=True)
        ind = i[np.where(c==1)[0]]
        bfaces = faces[ind]
        belem = nelems[ind]
        
    return belem, bfaces

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

def subdivide_mesh_smart(mesh, mask=None):
    xyz = mesh.points
    ien = mesh.cells_dict['triangle']
    tri = trimesh.Trimesh(vertices=xyz, faces=ien)

    faces = tri.faces
    faces_subset = faces
    vertices = tri.vertices

    # find the unique edges of our faces subset
    edges = np.sort(faces_to_edges(faces_subset), axis=1)
    unique, inverse = unique_rows(edges)
    # then only produce one midpoint per unique edge
    mid = vertices[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)

    # Find nodes in the interface
    if mask is None:
        _, bfaces = get_surface_mesh(mesh)
        interface_nodes = np.unique(bfaces)
        interface_nodes_og = interface_nodes
    else:
        interface_nodes_og = np.intersect1d(ien[mask], ien[~mask])

    # the new faces_subset with correct winding
    f = np.column_stack(
        [
            faces_subset[:, 0],
            mid_idx[:, 0],
            mid_idx[:, 2],
            mid_idx[:, 0],
            faces_subset[:, 1],
            mid_idx[:, 1],
            mid_idx[:, 2],
            mid_idx[:, 1],
            faces_subset[:, 2],
            mid_idx[:, 0],
            mid_idx[:, 1],
            mid_idx[:, 2],
        ]
    )

    # If mask is None, then we just divide the elements
    if mask is None or np.all(mask==1):
        new_faces = np.vstack((f[:,0:3], f[:,3:6], f[:,6:9], f[:,9:12])).astype(int)
        new_vertices = np.vstack((vertices, mid))

        # Deleting nodes that are not in the new_faces
        new_xyz = np.zeros((np.unique(new_faces).shape[0],xyz.shape[1]))
        new_ien = np.zeros(new_faces.shape, dtype=int)
        map_nodes = np.full(new_vertices.shape[0], -1, dtype=int)

        cont = 0
        for e in range(new_ien.shape[0]):
            for i in range(new_ien.shape[1]):
                if map_nodes[new_faces[e,i]] == -1:
                    new_ien[e,i] = cont
                    new_xyz[cont] = new_vertices[new_faces[e,i]]
                    map_nodes[new_faces[e,i]] = cont
                    cont += 1
                else:
                    new_ien[e,i] = map_nodes[new_faces[e,i]]

        mesh = io.Mesh(points=new_xyz, cells=[('triangle', new_ien)])
        _, bfaces = get_surface_mesh(mesh)
        interface_nodes = np.unique(bfaces)

        io.write('check2.vtu', mesh)
        return mesh, interface_nodes


    # Getting connectivity for the divided and non-divided faces
    print('Dividing elements to make a watertight mesh')
    len_add_to_divide = 10
    it = 0

    while len_add_to_divide > 0:
        elems_to_keep = np.where(mask==0)[0]
        divided_f = f[mask==1].reshape((-1, 3))
        non_divided_f = ien[elems_to_keep]
        non_divided_mid_idx = mid_idx[elems_to_keep]

        # Fixing non-watertightness
        elems, idx = np.where(np.isin(non_divided_mid_idx, divided_f))

        # In the first iteration the interface nodes are saved.
        if it == 0:
            interface_nodes = np.unique(non_divided_mid_idx[elems,idx])
            interface_nodes = np.union1d(interface_nodes, interface_nodes_og).astype(int)

        # Checking if an element needs to be divided more than once
        un, conts = np.unique(elems, return_counts=True)
        add_to_divide = un[np.where(conts > 1)[0]]
        mask[elems_to_keep[add_to_divide]] = 1
        len_add_to_divide = len(add_to_divide)
        print(it, elems_to_keep[add_to_divide])
        it+=1

    to_divide_mask = np.zeros(len(non_divided_f), dtype=bool)
    to_divide_mask[elems] = True

    new_elems = []
    for e, i in zip(elems, idx):
        if i == 0: # the other divider node is 2
            elem1 = np.array([non_divided_f[e][0], non_divided_mid_idx[e][0], non_divided_f[e][2]], dtype=int)
            elem2 = np.array([non_divided_mid_idx[e][0], non_divided_f[e][1], non_divided_f[e][2]], dtype=int)
            new_elems.append(elem1)
            new_elems.append(elem2)
        elif i == 1: # the other divider node is 0
            elem1 = np.array([non_divided_f[e][1], non_divided_mid_idx[e][1], non_divided_f[e][0]], dtype=int)
            elem2 = np.array([non_divided_mid_idx[e][1], non_divided_f[e][2], non_divided_f[e][0]], dtype=int)
            new_elems.append(elem1)
            new_elems.append(elem2)
        elif i == 2: # the other divider node is 1
            elem1 = np.array([non_divided_f[e][2], non_divided_mid_idx[e][2], non_divided_f[e][1]], dtype=int)
            elem2 = np.array([non_divided_mid_idx[e][2], non_divided_f[e][0], non_divided_f[e][1]], dtype=int)
            new_elems.append(elem1)
            new_elems.append(elem2)

    new_elems = np.vstack(new_elems).astype(int)
    new_faces = np.vstack((divided_f, new_elems, non_divided_f[~to_divide_mask])).astype(int)
    new_vertices = np.vstack((vertices, mid))

    # Deleting nodes that are not in the new_faces
    new_xyz = np.zeros((np.unique(new_faces).shape[0],xyz.shape[1]))
    new_ien = np.zeros(new_faces.shape, dtype=int)
    map_nodes = np.full(new_vertices.shape[0], -1, dtype=int)

    cont = 0
    for e in range(new_ien.shape[0]):
        for i in range(new_ien.shape[1]):
            if map_nodes[new_faces[e,i]] == -1:
                new_ien[e,i] = cont
                new_xyz[cont] = new_vertices[new_faces[e,i]]
                map_nodes[new_faces[e,i]] = cont
                cont += 1
            else:
                new_ien[e,i] = map_nodes[new_faces[e,i]]

    interface_nodes = map_nodes[interface_nodes]

    mesh = io.Mesh(points=new_xyz, cells=[('triangle', new_ien)])

    io.write('check2.vtu', mesh)
    return mesh, interface_nodes



def tri_quality_radius_ratio(xyz, ien):
    points = xyz[ien]
    assert points.shape[1] == 3 and points.shape[2] == 3

    l0 = points[:,1,:] - points[:,0,:]
    l1 = points[:,2,:] - points[:,1,:]
    l2 = points[:,0,:] - points[:,2,:]

    norm_l0 = np.linalg.norm(l0, axis=1)
    norm_l1 = np.linalg.norm(l1, axis=1)
    norm_l2 = np.linalg.norm(l2, axis=1)

    cross_l2_l0 = np.cross(l2,l0, axisa=1, axisb=1)

    area = 0.5*np.linalg.norm(cross_l2_l0, axis=1)

    sum_norms = (norm_l0 + norm_l1 + norm_l2)
    r = 2*area/sum_norms
    R = norm_l0*norm_l1*norm_l2/(2*r*sum_norms)
    quality = R/(2*r)

    return quality


def optimize_mesh(mesh, interface_nodes, bfaces):
    xyz = mesh.points
    ien = mesh.cells_dict['triangle']

    if xyz.shape[1] == 2:
        xyz = np.concatenate([xyz, np.zeros((xyz.shape[0], 1))], axis=1)

    # Calculate initial quality
    quality = tri_quality_radius_ratio(xyz, ien)
    
    moving_nodes = np.unique(ien[quality>2])

    # Improve mesh quality
    bnodes = np.unique(bfaces)
    interface_nodes = np.union1d(bnodes, interface_nodes)

    moving_nodes = np.setdiff1d(moving_nodes, interface_nodes)
    fixed_nodes = np.setdiff1d(np.arange(len(xyz)), moving_nodes)

    # fixed_nodes = np.union1d(bnodes, interface_nodes)
    # moving_nodes = np.setdiff1d(np.arange(len(xyz)), fixed_nodes)

    if len(moving_nodes) == 0:
        print('No nodes to move')
        return mesh

    ien_mod = ien[quality>2]
    def perturb_mesh_quality(pert):
        disp = np.zeros(xyz.shape)
        disp[moving_nodes,0:2] = pert.reshape([-1,2])
        pert_xyz = xyz + disp

        quality = tri_quality_radius_ratio(pert_xyz, ien_mod)-1

        return np.linalg.norm(quality**2*10)/len(ien)
    
    def callbackF(Xi):
        print('Optimizing quality:', perturb_mesh_quality(Xi))

    x0 = np.zeros(len(moving_nodes)*2)
    sol = minimize(perturb_mesh_quality, x0, method='CG', tol=1e-6, options={'maxiter': 10, 'disp': True, 'eps': 1e-5}, callback=callbackF)
    disp = np.zeros(xyz.shape)
    disp[moving_nodes,0:2] = sol.x.reshape([-1,2])
    pert_xyz = xyz + disp
    print(np.linalg.norm(tri_quality_radius_ratio(pert_xyz, ien)-1))

    mesh.points = pert_xyz[:,0:2]

    io.write('check.vtu', mesh)

    return mesh

def subdivide_mesh(tri_mesh):
    xyz = tri_mesh.points
    ien = tri_mesh.cells[0].data
    mesh = trimesh.Trimesh(xyz, ien)
    mesh = mesh.subdivide()

    xyz = mesh.vertices
    ien = mesh.faces
    tri_mesh = io.Mesh(xyz, {'triangle': ien})
    return tri_mesh


def subdivide_mesh_fibers(mesh, elem_fiber_mask, bfaces):
    elem_fiber_mask = elem_fiber_mask.copy()

    # We first subdivide
    mesh, interface_nodes = subdivide_mesh_smart(mesh, elem_fiber_mask) 
    io.write('check2.vtu', mesh)

    # # Then we optimize the mesh
    # print('Optimizing mesh')
    # mesh = optimize_mesh(mesh, interface_nodes, bfaces)
    # io.write('check3.vtu', mesh)

    return mesh
    


def mask2mesh_with_fibers(tissue_mask_og, fiber_mask_og, rescale=4, meshsize=10, subdivide_fibers=False):
    rescaled_meshsize = meshsize*rescale

    # Get contour using tissue_mask
    mesh_tissue_mask = tissue_mask_og.astype(int)
    mesh_tissue_mask = normalize_image(transform.rescale(mesh_tissue_mask, rescale))
    mesh_tissue_mask = morphology.binary_closing(mesh_tissue_mask, footprint=morphology.disk(2))
    mesh_tissue_mask[:,0] = 0
    mesh_tissue_mask[:,-1] = 0

    mask = clean_mask(mesh_tissue_mask)
    mask[0:10] = 0
    mask[-10:] = 0

    # Find boundary polygon
    mask = np.pad(mask, pad_width=((20,20),(0,0)))
    mask = filters.gaussian(mask) > 0.75
    tissue_mask = mask

    # Find holes
    mesh_fiber_mask = fiber_mask_og.astype(int)
    mesh_fiber_mask = normalize_image(transform.rescale(mesh_fiber_mask, rescale))
    if np.all(fiber_mask_og == 0):
        mask = np.zeros_like(mesh_fiber_mask)
    else:
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
    k = np.round(length/rescaled_meshsize).astype(int)
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
        k = np.round(length/rescaled_meshsize).astype(int)
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


    # Plot boundary and holes
    fig, ax = plt.subplots(figsize=(20,10))
    ax.plot(tissue_equi_points[:, 1], tissue_equi_points[:, 0], 'r.', linewidth=2, label='Boundary')
    for hole in holes:
        ax.plot(hole[:, 1], hole[:, 0], 'b-', linewidth=1, label='Hole')
    ax.set_aspect('equal')


    # mesh
    with pygmsh.occ.Geometry() as geom:

        borders = []
        for hole in tqdm(holes):
            borders.append(geom.add_polygon(hole,
                mesh_size=rescaled_meshsize
            ))

        # main surface
        tissue = geom.add_polygon(tissue_equi_points,
            mesh_size=rescaled_meshsize
        )

        # # Cutout the fiber boundary
        cuts = []
        for border in tqdm(borders):
            cuts.append(geom.boolean_intersection([tissue, border], delete_first=False)[0])

        geom.boolean_fragments(tissue, cuts)

        mesh = geom.generate_mesh()

    points = mesh.points[:,0:2]
    tri_mesh = io.Mesh(points, {'triangle': mesh.cells_dict['triangle']})
    xyz = tri_mesh.points
    ien = tri_mesh.cells_dict['triangle']
    _, bfaces = get_surface_mesh(tri_mesh)

    if np.sum(fiber_mask_og) == 0:
        tri_mesh.points = tri_mesh.points/rescale
        return tri_mesh, None, None

    # Obtaining mask of fiber elements
    midpoints = np.mean(xyz[ien], axis=1)
    inholes = np.zeros(len(midpoints), dtype=int)
    spoints = [sg.Point(point) for point in midpoints]

    print('Finding fiber elems')
    for hole in tqdm(holes):
        poly = sg.polygon.Polygon(hole)
        marker = poly.contains(spoints)
        inholes[marker] = 1

    elem_fiber_mask = 1-inholes

    tri_mesh.cell_data['fiber mask'] = [elem_fiber_mask]
    io.write('check1.vtu', tri_mesh)

    # # Plot the mesh
    # fig, ax = plt.subplots()
    # plt.tripcolor(xyz[:,1], xyz[:,0], ien, elem_fiber_mask, shading='flat') 
    # ax.set_aspect('equal')

    # # Plot borders
    # for hole in holes:
    #     ax.plot(hole[:, 1], hole[:, 0], 'b-', linewidth=1, label='Hole')
    # ax.plot(tissue_equi_points[:, 1], tissue_equi_points[:, 0], 'r-', linewidth=2, label='Boundary')

    # plt.show()
    

    print('Removing disconnected cells')
    disconnected_cells = find_disconnected_cells(ien[elem_fiber_mask==1])
    cells = np.arange(len(ien))
    cells = cells[elem_fiber_mask==1]
    elem_fiber_mask[cells[disconnected_cells]] = 0

    # Check for isolated cells
    print('Finding fiber_mesh neighbors')
    fiber_elems = np.where(elem_fiber_mask==1)[0]
    elem_neigh = get_elem_neighbors(tri_mesh.cells[0].data[fiber_elems])
    non_neigh = np.sum(elem_neigh==-1, axis=1)
    isolated_cells = fiber_elems[np.where(non_neigh==3)[0]]

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

    tri_mesh.cell_data['fiber mask'] = [elem_fiber_mask]

    io.write('check2.vtu', tri_mesh)

    # Subdivide fiber elements if needed
    if subdivide_fibers:
        print('Subdividing fiber elements')
        tri_mesh = subdivide_mesh_fibers(tri_mesh, elem_fiber_mask, bfaces)

    # Getting boundaries to fix bad elements
    xmin = np.min(tri_mesh.points[:,0])
    xmax = np.max(tri_mesh.points[:,0])
    post1_faces = np.where(np.all(np.isclose(xyz[bfaces][:,:,0], xmin), axis=1))[0]
    post2_faces = np.where(np.all(np.isclose(xyz[bfaces][:,:,0], xmax), axis=1))[0]
    post_nodes = np.union1d(bfaces[post1_faces], bfaces[post2_faces])

    # Loop until all bad elements are fixed
    del_elems = 100
    it = 1
    while del_elems > 0:
        xyz, ien, elem_fiber_mask, del_elems = fix_bad_post_elements(xyz, ien, post_nodes, elem_fiber_mask)
        print('Iteration:', it, 'Deleted elements:', del_elems)
        it += 1

    xyz = xyz/rescale
    tri_mesh = io.Mesh(xyz, {'triangle': ien})
    io.write('check3.vtu', tri_mesh)

    # Just making sure the mesh is built correctly (probably not the smartest way of doing this)
    tri_mesh, _, _ = create_submesh(tri_mesh, np.arange(len(tri_mesh.cells[0].data)))

    # Create fiber mesh
    fiber_mesh, map_mesh_fiber, map_fiber_mesh = create_submesh(tri_mesh, np.where(elem_fiber_mask==1)[0])
    io.write('check4.vtu', tri_mesh)

    return tri_mesh, elem_fiber_mask, fiber_mesh



def mask2mesh_only_fibers(tissue_mask_og, fiber_mask_og, rescale=4, meshsize=10, subdivide_fibers=False):
    rescaled_meshsize = meshsize*rescale

    # Get contour using tissue_mask
    mesh_tissue_mask = tissue_mask_og.astype(int)
    mesh_tissue_mask = normalize_image(transform.rescale(mesh_tissue_mask, rescale))
    mesh_tissue_mask = morphology.binary_closing(mesh_tissue_mask, footprint=morphology.disk(2))

    mask = clean_mask(mesh_tissue_mask)
    mask = mesh_tissue_mask
    mask[0:10] = 0
    mask[-10:] = 0

    # Find boundary polygon
    mask = np.pad(mask, pad_width=((20,20),(0,0)))
    mask = filters.gaussian(mask) > 0.75
    tissue_mask = mask


    # Find holes
    mesh_fiber_mask = fiber_mask_og.astype(int)
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
    min_area = 0.5*(rescaled_meshsize/2)**2
    k = np.round(length/(rescaled_meshsize)*2).astype(int)
    equi_points = np.transpose([polygon.exterior.interpolate(t).xy for t in np.linspace(0,polygon.length,k,False)])
    equi_points = equi_points.squeeze().T
    equi_points[:,0] -= 20.5
    minx = np.min(equi_points[:,0])
    maxx = np.max(equi_points[:,0])
    tissue_equi_points = equi_points

    # Smooth tissue
    tissue_equi_points = np.vstack((tissue_equi_points, tissue_equi_points[0])) # closing the polygon
    tissue_equi_points = np.array(taubin_smooth(tissue_equi_points, factor=0.1, mu=0.1, steps=1))[:-1]
    polygon = sg.Polygon(tissue_equi_points)
    length = polygon.length
    k = np.round(length/(rescaled_meshsize)*3).astype(int)
    tissue_equi_points = np.transpose([polygon.exterior.interpolate(t).xy for t in np.linspace(0,polygon.length,k,False)])
    tissue_equi_points = tissue_equi_points.squeeze().T
    tissue_equi_points[tissue_equi_points[:,0] < minx, 0] = minx
    tissue_equi_points[tissue_equi_points[:,0] > maxx, 0] = maxx


    holes = []
    len_holes = []
    for boundary in boundaries:

        boundary = np.append(boundary, boundary[0,None], axis=0)
        polygon = sg.Polygon(boundary)
        if polygon.area < min_area:
            continue
        length = polygon.length
        k = np.round(length/(rescaled_meshsize)*2).astype(int)
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

    # Smooth holes
    holes_smooth = []
    for hole in holes:
        h = np.vstack((hole, hole[0]))
        h = np.array(taubin_smooth(h, factor=0.1, mu=0.1, steps=1))[:-1]
        if h.shape[0] >= 4:
            polygon = sg.Polygon(h)
            length = polygon.length
            k = np.round(length/(rescaled_meshsize)*2).astype(int)
            h = np.transpose([polygon.exterior.interpolate(t).xy for t in np.linspace(0,polygon.length,k,False)])
            h = h.squeeze().T
            h[h[:,0] < minx, 0] = minx
            h[h[:,0] > maxx, 0] = maxx

        try:
            poly_hole = sg.Polygon(h)
        except:
            continue
        if poly_hole.area < min_area:
            continue

        holes_smooth.append(h)
        
    holes = holes_smooth


    # Plot tissue_equi_points and holes
    fig, ax = plt.subplots(figsize=(60, 20))
    ax.imshow(mask, cmap='gray', origin='lower')
    ax.plot(tissue_equi_points[:, 1], tissue_equi_points[:, 0], 'r.-', linewidth=2, label='Tissue Boundary')
    for hole in holes:
        ax.plot(hole[:, 1], hole[:, 0], 'b.-', linewidth=1, label='Hole')
    ax.set_aspect('equal')

    # mesh
    plt.figure(1)
    with pygmsh.occ.Geometry() as geom:

        borders = []
        for hole in tqdm(holes):
            borders.append(geom.add_polygon(hole,
                mesh_size=rescaled_meshsize
            ))

        # main surface
        tissue = geom.add_polygon(tissue_equi_points,
            mesh_size=rescaled_meshsize
        )

        # # Cutout the fiber boundary
        cuts = []
        for border in tqdm(borders):
            p = np.array([p.x for p in border.points])
            plt.plot(p[:, 1], p[:, 0], 'b.-', linewidth=1)
            cuts.append(geom.boolean_intersection([tissue, border], delete_first=False)[0])

        geom.boolean_fragments(tissue, cuts)

        mesh = geom.generate_mesh(verbose=True)

    points = mesh.points[:,0:2]
    tri_mesh = io.Mesh(points, {'triangle': mesh.cells_dict['triangle']})
    xyz = tri_mesh.points
    ien = tri_mesh.cells_dict['triangle']

    io.write('check0.vtu', tri_mesh)
    
    # Obtaining mask of fiber elements
    midpoints = np.mean(xyz[ien], axis=1)
    inholes = np.zeros(len(midpoints), dtype=int)
    spoints = [sg.Point(point) for point in midpoints]

    print('Finding fiber elems')
    for hole in tqdm(holes):
        poly = sg.polygon.Polygon(hole)
        marker = poly.contains(spoints)
        inholes[marker] = 1

    elem_fiber_mask = 1-inholes

    tri_mesh.cell_data['fiber mask'] = [elem_fiber_mask]
    io.write('check1.vtu', tri_mesh)

    print('Removing disconnected cells')
    disconnected_cells = find_disconnected_cells(ien[elem_fiber_mask==1])
    cells = np.arange(len(ien))
    cells = cells[elem_fiber_mask==1]
    elem_fiber_mask[cells[disconnected_cells]] = 0

    # Check for isolated cells
    print('Finding fiber_mesh neighbors')
    fiber_elems = np.where(elem_fiber_mask==1)[0]
    elem_neigh = get_elem_neighbors(tri_mesh.cells[0].data[fiber_elems])
    non_neigh = np.sum(elem_neigh==-1, axis=1)
    isolated_cells = fiber_elems[np.where(non_neigh==3)[0]]

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

    tri_mesh.cell_data['fiber mask'] = [elem_fiber_mask]
    fiber_mesh, _, _ = create_submesh(tri_mesh, np.where(elem_fiber_mask==1)[0])
    xyz = fiber_mesh.points
    ien = fiber_mesh.cells_dict['triangle']

    elem_fiber_mask = np.ones(len(fiber_mesh.cells[0].data), dtype=int)
    belems, bfaces = get_surface_mesh(fiber_mesh)

    # Getting boundaries to fix bad elements
    print('Deleting bad elements on the boundaries')
    xmin = np.min(fiber_mesh.points[:,0])
    xmax = np.max(fiber_mesh.points[:,0])
    post1_faces = np.where(np.all(np.isclose(xyz[bfaces][:,:,0], xmin), axis=1))[0]
    post2_faces = np.where(np.all(np.isclose(xyz[bfaces][:,:,0], xmax), axis=1))[0]
    post_nodes = np.union1d(bfaces[post1_faces], bfaces[post2_faces])

    # Loop until all bad elements are fixed
    del_elems = 100
    it = 1
    while del_elems > 0:
        xyz, ien, elem_fiber_mask, del_elems = fix_bad_post_elements(xyz, ien, post_nodes, elem_fiber_mask)
        print('Iteration:', it, 'Deleted elements:', del_elems)
        it += 1

    xyz = xyz/rescale

    # Check if bad elements only have one neighbor
    fiber_mesh = io.Mesh(xyz, {'triangle': ien})
    belems, bfaces = get_surface_mesh(fiber_mesh)
    xyz, ien = fix_quality_one_neigh_elems(xyz, ien, belems)

    tri_mesh = io.Mesh(xyz, {'triangle': ien})
    _, bfaces = get_surface_mesh(tri_mesh)
    io.write('check.vtu', fiber_mesh)
    
    # Subdivide fiber elements if needed
    if subdivide_fibers:
        print('Subdividing fiber elements')
        tri_mesh = subdivide_mesh_fibers(tri_mesh, elem_fiber_mask, bfaces)

    # Dummy process to get rid of isolated points
    fiber_mesh, _, _ = create_submesh(tri_mesh, np.arange(len(tri_mesh.cells[0].data)))

    # Create tissue mesh

    with pygmsh.occ.Geometry() as geom:
        # main surface
        tissue = geom.add_polygon(tissue_equi_points,
            mesh_size=rescaled_meshsize*2
        )

        mesh = geom.generate_mesh(verbose=True)

    points = mesh.points[:,0:2]
    tissue_mesh = io.Mesh(points, {'triangle': mesh.cells_dict['triangle']})
    tissue_mesh.points = tissue_mesh.points/rescale


    return fiber_mesh, tissue_mesh




def mask2mesh_only_fibers_toy_test_full(tissue_mask_og, fiber_mask_og, rescale=4, meshsize=10, subdivide_fibers=False, clean_fiber_mask=True):
    rescaled_meshsize = meshsize*rescale

    # Get contour using tissue_mask
    mesh_tissue_mask = tissue_mask_og.astype(int)
    mesh_tissue_mask = normalize_image(transform.rescale(mesh_tissue_mask, rescale))
    mesh_tissue_mask = morphology.binary_closing(mesh_tissue_mask, footprint=morphology.disk(2))

    mask = clean_mask(mesh_tissue_mask)
    mask = mesh_tissue_mask
    mask[0:10] = 0
    mask[-10:] = 0

    # Find boundary polygon
    mask = np.pad(mask, pad_width=((20,20),(0,0)))
    # mask = filters.gaussian(mask) > 0.75
    tissue_mask = mask


    # Find holes
    mesh_fiber_mask = fiber_mask_og.astype(int)
    mesh_fiber_mask = normalize_image(transform.rescale(mesh_fiber_mask, rescale))

    # Generate mesh and grab node coordinates
    if clean_fiber_mask:
        mask = clean_mask(mesh_fiber_mask)
    else:
        mask = mesh_fiber_mask

    # Find boundary polygon
    mask = np.pad(mask, pad_width=((20,20),(0,0)))
    mask[~tissue_mask] = 0
    mask += ~(tissue_mask) #morphology.binary_erosion(tissue_mask, footprint=morphology.disk(4)))
    mask = filters.gaussian(mask) > 0.5
    boundaries = measure.find_contours(~mask)

    # Tissue boundary
    # tissue_mask = morphology.binary_erosion(tissue_mask, footprint=morphology.disk(5))
    tissue_mask[0:20] = 0
    tissue_mask[-20:] = 0

    boundary = measure.find_contours(tissue_mask)[0]
    polygon = sg.Polygon(boundary)
    length = polygon.length
    min_area = 0.5*(rescaled_meshsize/2)**2
    k = np.round(length/(rescaled_meshsize)*2).astype(int)
    equi_points = np.transpose([polygon.exterior.interpolate(t).xy for t in np.linspace(0,polygon.length,k,False)])
    equi_points = equi_points.squeeze().T
    equi_points[:,0] -= 20.5
    minx = np.min(equi_points[:,0])
    maxx = np.max(equi_points[:,0])
    tissue_equi_points = equi_points

    # Smooth tissue
    tissue_equi_points = np.vstack((tissue_equi_points, tissue_equi_points[0])) # closing the polygon
    tissue_equi_points = np.array(taubin_smooth(tissue_equi_points, factor=0.1, mu=0.1, steps=1))[:-1]
    polygon = sg.Polygon(tissue_equi_points)
    length = polygon.length
    k = np.round(length/(rescaled_meshsize)*3).astype(int)
    tissue_equi_points = np.transpose([polygon.exterior.interpolate(t).xy for t in np.linspace(0,polygon.length,k,False)])
    tissue_equi_points = tissue_equi_points.squeeze().T
    tissue_equi_points[tissue_equi_points[:,0] < minx, 0] = minx
    tissue_equi_points[tissue_equi_points[:,0] > maxx, 0] = maxx


    holes = []
    len_holes = []
    for boundary in boundaries:

        boundary = np.append(boundary, boundary[0,None], axis=0)
        polygon = sg.Polygon(boundary)
        if polygon.area < min_area:
            continue
        length = polygon.length
        k = np.round(length/(rescaled_meshsize)*2).astype(int)
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

    # Smooth holes
    holes_smooth = []
    for hole in holes:
        h = np.vstack((hole, hole[0]))
        h = np.array(taubin_smooth(h, factor=0.1, mu=0.1, steps=1))[:-1]
        if h.shape[0] >= 4:
            polygon = sg.Polygon(h)
            length = polygon.length
            k = np.round(length/(rescaled_meshsize)*2).astype(int)
            h = np.transpose([polygon.exterior.interpolate(t).xy for t in np.linspace(0,polygon.length,k,False)])
            h = h.squeeze().T
            h[h[:,0] < minx, 0] = minx
            h[h[:,0] > maxx, 0] = maxx

        try:
            poly_hole = sg.Polygon(h)
        except:
            continue
        if poly_hole.area < min_area:
            continue

        holes_smooth.append(h)
        
    holes = holes_smooth


    # Plot tissue_equi_points and holes
    fig, ax = plt.subplots(figsize=(60, 20))
    # ax.imshow(mask, cmap='gray', origin='lower')
    ax.plot(tissue_equi_points[:, 1], tissue_equi_points[:, 0], 'r.-', linewidth=2, label='Tissue Boundary')
    for hole in holes:
        ax.plot(hole[:, 1], hole[:, 0], 'b.-', linewidth=1, label='Hole')
    ax.set_aspect('equal')

    # mesh
    plt.figure(1)
    with pygmsh.occ.Geometry() as geom:

        borders = []
        for hole in tqdm(holes):
            borders.append(geom.add_polygon(hole,
                mesh_size=rescaled_meshsize
            ))

        # main surface
        tissue = geom.add_polygon(tissue_equi_points,
            mesh_size=rescaled_meshsize
        )

        # # Cutout the fiber boundary
        cuts = []
        for border in tqdm(borders[:-1]):
            p = np.array([p.x for p in border.points])
            plt.plot(p[:, 1], p[:, 0], 'b.-', linewidth=1)
            try:
                intersect = geom.boolean_intersection([tissue, border], delete_first=False)[0]
            except:
                print('Intersection failed, skipping border')
                continue
            cuts.append(intersect)

        geom.boolean_fragments(tissue, cuts)

        mesh = geom.generate_mesh(verbose=True)

    points = mesh.points[:,0:2]
    tri_mesh = io.Mesh(points, {'triangle': mesh.cells_dict['triangle']})
    xyz = tri_mesh.points
    ien = tri_mesh.cells_dict['triangle']

    io.write('check0.vtu', tri_mesh)
    
    # Obtaining mask of fiber elements
    midpoints = np.mean(xyz[ien], axis=1)
    inholes = np.zeros(len(midpoints), dtype=int)
    spoints = [sg.Point(point) for point in midpoints]

    print('Finding fiber elems')
    for hole in tqdm(holes):
        poly = sg.polygon.Polygon(hole)
        marker = poly.contains(spoints)
        inholes[marker] = 1

    elem_fiber_mask = 1-inholes

    tri_mesh.cell_data['fiber mask'] = [elem_fiber_mask]
    io.write('check1.vtu', tri_mesh)

    print('Removing disconnected cells')
    disconnected_cells = find_disconnected_cells(ien[elem_fiber_mask==1])
    cells = np.arange(len(ien))
    cells = cells[elem_fiber_mask==1]
    elem_fiber_mask[cells[disconnected_cells]] = 0

    # Check for isolated cells
    print('Finding fiber_mesh neighbors')
    fiber_elems = np.where(elem_fiber_mask==1)[0]
    elem_neigh = get_elem_neighbors(tri_mesh.cells[0].data[fiber_elems])
    non_neigh = np.sum(elem_neigh==-1, axis=1)
    isolated_cells = fiber_elems[np.where(non_neigh==3)[0]]

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

    tri_mesh.cell_data['fiber mask'] = [elem_fiber_mask]
    fiber_mesh, _, _ = create_submesh(tri_mesh, np.where(elem_fiber_mask==1)[0])
    xyz = fiber_mesh.points
    ien = fiber_mesh.cells_dict['triangle']

    elem_fiber_mask = np.ones(len(fiber_mesh.cells[0].data), dtype=int)
    belems, bfaces = get_surface_mesh(fiber_mesh)

    # Getting boundaries to fix bad elements
    print('Deleting bad elements on the boundaries')
    xmin = np.min(fiber_mesh.points[:,0])
    xmax = np.max(fiber_mesh.points[:,0])
    post1_faces = np.where(np.all(np.isclose(xyz[bfaces][:,:,0], xmin), axis=1))[0]
    post2_faces = np.where(np.all(np.isclose(xyz[bfaces][:,:,0], xmax), axis=1))[0]
    post_nodes = np.union1d(bfaces[post1_faces], bfaces[post2_faces])

    # Loop until all bad elements are fixed
    del_elems = 100
    it = 1
    while del_elems > 0:
        xyz, ien, elem_fiber_mask, del_elems = fix_bad_post_elements(xyz, ien, post_nodes, elem_fiber_mask)
        print('Iteration:', it, 'Deleted elements:', del_elems)
        it += 1

    xyz = xyz/rescale

    # Check if bad elements only have one neighbor
    fiber_mesh = io.Mesh(xyz, {'triangle': ien})
    belems, bfaces = get_surface_mesh(fiber_mesh)
    xyz, ien = fix_quality_one_neigh_elems(xyz, ien, belems)

    tri_mesh = io.Mesh(xyz, {'triangle': ien})
    _, bfaces = get_surface_mesh(tri_mesh)
    io.write('check.vtu', fiber_mesh)
    
    # Subdivide fiber elements if needed
    if subdivide_fibers:
        print('Subdividing fiber elements')
        tri_mesh = subdivide_mesh_fibers(tri_mesh, elem_fiber_mask, bfaces)

    # Dummy process to get rid of isolated points
    fiber_mesh, _, _ = create_submesh(tri_mesh, np.arange(len(tri_mesh.cells[0].data)))

    # Create tissue mesh

    with pygmsh.occ.Geometry() as geom:
        # main surface
        tissue = geom.add_polygon(tissue_equi_points,
            mesh_size=rescaled_meshsize*2
        )

        mesh = geom.generate_mesh(verbose=True)

    points = mesh.points[:,0:2]
    tissue_mesh = io.Mesh(points, {'triangle': mesh.cells_dict['triangle']})
    tissue_mesh.points = tissue_mesh.points/rescale


    return fiber_mesh, tissue_mesh



def mask2mesh_only_fibers_toy_test(tissue_mask_og, fiber_mask_og, rescale=4, meshsize=10, add_posts=False, subdivide_fibers=False):
    rescaled_meshsize = meshsize*rescale

    # Get contour using tissue_mask
    mesh_tissue_mask = tissue_mask_og.astype(int)
    mesh_tissue_mask = normalize_image(transform.rescale(mesh_tissue_mask, rescale)) > 0.5
    # mesh_tissue_mask = morphology.binary_closing(mesh_tissue_mask, footprint=morphology.disk(2))

    # mask = clean_mask(mesh_tissue_mask)
    mask = mesh_tissue_mask
    mask[0:10] = 0
    mask[-10:] = 0

    # Find boundary polygon
    mask = np.pad(mask, pad_width=((20,20),(10,10)))
    # mask = filters.gaussian(mask) > 0.75
    tissue_mask = mask.astype(bool)


    # Find holes
    mesh_fiber_mask = fiber_mask_og.astype(int)
    mesh_fiber_mask = normalize_image(transform.rescale(mesh_fiber_mask, rescale))

    # Generate mesh and grab node coordinates
    mask = clean_mask(mesh_fiber_mask)

    # Find boundary polygon
    mask = np.pad(mask, pad_width=((20,20),(10,10)))
    mask[~tissue_mask] = 0
    mask += ~tissue_mask
    # mask = filters.gaussian(mask) > 0.5
    boundaries = measure.find_contours(~mask)

    min_area = 0.5*rescaled_meshsize**2

    # Tissue boundary
    # tissue_mask = morphology.binary_erosion(tissue_mask, footprint=morphology.disk(2))

    boundary = measure.find_contours(tissue_mask.astype(int))[0]
    xmin, ymin = np.min(boundary, axis=0)
    xmax, ymax = np.max(boundary, axis=0)
    equi_points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    equi_points[:,0] -= 20.5
    equi_points[:,1] -= 10.5
    xmin, ymin = np.min(equi_points, axis=0)
    xmax, ymax = np.max(equi_points, axis=0)
    tissue_equi_points = equi_points

    holes = []
    len_holes = []
    for boundary in boundaries:

        boundary = np.append(boundary, boundary[0,None], axis=0)
        polygon = sg.Polygon(boundary)
        if polygon.area < min_area:
            continue
        length = polygon.length
        k = np.round(length/rescaled_meshsize).astype(int)
        if k < 4:
            continue
        equi_points = np.transpose([polygon.exterior.interpolate(t).xy for t in np.linspace(0,polygon.length,k,False)])
        equi_points = equi_points.squeeze().T
        equi_points[:,0] -= 20.5
        equi_points[:,1] -= 10.5
        # equi_points = np.append(equi_points, equi_points[0,None], axis=0)
        holes.append(equi_points)
        len_holes.append(len(equi_points))

    # Order holes based on len_holes
    holes = [hole for _, hole in sorted(zip(len_holes, holes), key=lambda x: x[0])][::-1]

    # # Smooth holes
    # holes_smooth = []
    # for hole in holes:
    #     h = np.vstack((hole, hole[0]))
    #     h = np.array(taubin_smooth(h))[:-1]
    #     h[h[:,0] < xmin, 0] = xmin
    #     h[h[:,0] > xmax, 0] = xmax

    #     poly_hole = sg.Polygon(h)
    #     if poly_hole.area < min_area:
    #         continue

    #     holes_smooth.append(h)
        
    # holes = holes_smooth


    # Plot tissue_equi_points and holes
    fig, ax = plt.subplots(figsize=(40, 8))
    ax.plot(tissue_equi_points[:, 1], tissue_equi_points[:, 0], 'r.-', linewidth=2, label='Tissue Boundary')
    for hole in holes:
        ax.plot(hole[:, 1], hole[:, 0], 'b.-', linewidth=1, label='Hole')
    ax.set_aspect('equal')

    # mesh
    plt.figure(1)
    with pygmsh.occ.Geometry() as geom:

        borders = []
        for hole in tqdm(holes):
            borders.append(geom.add_polygon(hole,
                mesh_size=rescaled_meshsize
            ))

        # main surface
        tissue = geom.add_polygon(tissue_equi_points,
            mesh_size=rescaled_meshsize
        )

        # # Cutout the fiber boundary
        cuts = []
        for border in tqdm(borders):
            p = np.array([p.x for p in border.points])
            plt.plot(p[:, 1], p[:, 0], 'b.-', linewidth=1)
            cuts.append(geom.boolean_intersection([tissue, border], delete_first=False)[0])

        geom.boolean_fragments(tissue, cuts)

        mesh = geom.generate_mesh(verbose=True)

    points = mesh.points[:,0:2]
    tri_mesh = io.Mesh(points, {'triangle': mesh.cells_dict['triangle']})
    xyz = tri_mesh.points
    ien = tri_mesh.cells_dict['triangle']
    
    # Obtaining mask of fiber elements
    midpoints = np.mean(xyz[ien], axis=1)
    inholes = np.zeros(len(midpoints), dtype=int)
    spoints = [sg.Point(point) for point in midpoints]

    print('Finding fiber elems')
    for hole in tqdm(holes):
        poly = sg.polygon.Polygon(hole)
        marker = poly.contains(spoints)
        inholes[marker] = 1

    elem_fiber_mask = 1-inholes

    print('Removing disconnected cells')
    disconnected_cells = find_disconnected_cells(ien[elem_fiber_mask==1])
    cells = np.arange(len(ien))
    cells = cells[elem_fiber_mask==1]
    elem_fiber_mask[cells[disconnected_cells]] = 0

    # Check for isolated cells
    print('Finding fiber_mesh neighbors')
    fiber_elems = np.where(elem_fiber_mask==1)[0]
    elem_neigh = get_elem_neighbors(tri_mesh.cells[0].data[fiber_elems])
    non_neigh = np.sum(elem_neigh==-1, axis=1)
    isolated_cells = fiber_elems[np.where(non_neigh==3)[0]]

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

    tri_mesh.cell_data['fiber mask'] = [elem_fiber_mask]
    fiber_mesh, _, _ = create_submesh(tri_mesh, np.where(elem_fiber_mask==1)[0])
    xyz = fiber_mesh.points
    ien = fiber_mesh.cells_dict['triangle']

    elem_fiber_mask = np.ones(len(fiber_mesh.cells[0].data), dtype=int)
    belems, bfaces = get_surface_mesh(fiber_mesh)

    # Getting boundaries to fix bad elements
    print('Deleting bad elements on the boundaries')
    xmin = np.min(fiber_mesh.points[:,0])
    xmax = np.max(fiber_mesh.points[:,0])
    post1_faces = np.where(np.all(np.isclose(xyz[bfaces][:,:,0], xmin), axis=1))[0]
    post2_faces = np.where(np.all(np.isclose(xyz[bfaces][:,:,0], xmax), axis=1))[0]
    post_nodes = np.union1d(bfaces[post1_faces], bfaces[post2_faces])

    # Loop until all bad elements are fixed
    del_elems = 100
    it = 1
    while del_elems > 0:
        xyz, ien, elem_fiber_mask, del_elems = fix_bad_post_elements(xyz, ien, post_nodes, elem_fiber_mask)
        print('Iteration:', it, 'Deleted elements:', del_elems)
        it += 1

    xyz = xyz/rescale

    # Check if bad elements only have one neighbor
    fiber_mesh = io.Mesh(xyz, {'triangle': ien})
    belems, bfaces = get_surface_mesh(fiber_mesh)
    xyz, ien = fix_quality_one_neigh_elems(xyz, ien, belems)

    tri_mesh = io.Mesh(xyz, {'triangle': ien})
    _, bfaces = get_surface_mesh(tri_mesh)
    io.write('check.vtu', tri_mesh)
    
    # Subdivide fiber elements if needed
    if subdivide_fibers:
        print('Subdividing fiber elements')
        tri_mesh = subdivide_mesh_fibers(tri_mesh, elem_fiber_mask, bfaces)

    # Dummy process to get rid of isolated points
    fiber_mesh, _, _ = create_submesh(tri_mesh, np.arange(len(tri_mesh.cells[0].data)))

    # Create tissue mesh

    with pygmsh.occ.Geometry() as geom:
        # main surface
        tissue = geom.add_polygon(tissue_equi_points,
            mesh_size=rescaled_meshsize*2
        )

        mesh = geom.generate_mesh(verbose=True)

    points = mesh.points[:,0:2]
    tissue_mesh = io.Mesh(points, {'triangle': mesh.cells_dict['triangle']})
    tissue_mesh.points = tissue_mesh.points/rescale


    return fiber_mesh, tissue_mesh




def mask2mesh_only_fibers_toy_test2(tissue_mask_og, fiber_mask_og, rescale=4, meshsize=10, add_posts=False, subdivide_fibers=False):
    rescaled_meshsize = meshsize*rescale

    # Get contour using tissue_mask
    mesh_tissue_mask = tissue_mask_og.astype(int)
    mesh_tissue_mask = normalize_image(transform.rescale(mesh_tissue_mask, rescale)) > 0.5
    mesh_tissue_mask = morphology.binary_closing(mesh_tissue_mask, footprint=morphology.disk(2))

    # mask = clean_mask(mesh_tissue_mask)
    mask = mesh_tissue_mask
    mask[0:10] = 0
    mask[-10:] = 0
    mask[:,-50:] = 0

    # Find boundary polygon
    mask = np.pad(mask, pad_width=((20,20),(10,10)))
    # mask = filters.gaussian(mask) > 0.75
    tissue_mask = mask.astype(bool)


    # Find holes
    mesh_fiber_mask = fiber_mask_og.astype(int)
    mesh_fiber_mask = normalize_image(transform.rescale(mesh_fiber_mask, rescale))

    # Generate mesh and grab node coordinates
    mask = clean_mask(mesh_fiber_mask)

    # Find boundary polygon
    mask = np.pad(mask, pad_width=((20,20),(10,10)))
    mask[~tissue_mask] = 0
    mask += ~tissue_mask
    mask[:,-60:] = 0
    mask[0:45] = 1
    mask[-45:] = 1
    mask[:,-5:] = 1
    # mask = filters.gaussian(mask) > 0.5
    boundaries = measure.find_contours(~mask)

    min_area = 0.5*rescaled_meshsize**2

    # Tissue boundary
    # tissue_mask = morphology.binary_erosion(tissue_mask, footprint=morphology.disk(2))

    boundary = measure.find_contours(tissue_mask.astype(int))[0]
    polygon = sg.Polygon(boundary)
    length = polygon.length
    min_area = 0.5*(rescaled_meshsize/2)**2
    k = np.round(length/(rescaled_meshsize)*2).astype(int)
    equi_points = np.transpose([polygon.exterior.interpolate(t).xy for t in np.linspace(0,polygon.length,k,False)])
    equi_points = equi_points.squeeze().T
    equi_points[:,0] -= 20.5
    tissue_equi_points = equi_points
    minx = np.min(equi_points[:,0])
    maxx = np.max(equi_points[:,0])
    tissue_equi_points[tissue_equi_points[:,0] - minx < rescaled_meshsize/2, 0] = minx
    tissue_equi_points[tissue_equi_points[:,0] - maxx > -rescaled_meshsize/2, 0] = maxx

    maxy = np.max(tissue_equi_points[:,1])
    marker = np.isclose(tissue_equi_points[:,0], minx) | np.isclose(tissue_equi_points[:,0], maxx) | np.isclose(tissue_equi_points[:,1], maxy)

    smooth_points = np.array(taubin_smooth(tissue_equi_points))
    tissue_equi_points[~marker, :] = smooth_points[~marker, :]


    # # Plot tissue_equi_points to visualize the boundary
    # plt.figure(figsize=(10, 5))
    # plt.plot(tissue_equi_points[:, 1], tissue_equi_points[:, 0], 'r.-', label='Tissue Boundary')
    # plt.title('Tissue Boundary Points')
    # plt.xlabel('Y')
    # plt.ylabel('X')
    # plt.gca().set_aspect('equal')
    # plt.legend()
    # plt.show()

    holes = []
    len_holes = []
    for boundary in boundaries:

        boundary = np.append(boundary, boundary[0,None], axis=0)
        polygon = sg.Polygon(boundary)
        if polygon.area < min_area:
            continue
        length = polygon.length
        k = np.round(length/rescaled_meshsize).astype(int)
        if k < 4:
            continue
        equi_points = np.transpose([polygon.exterior.interpolate(t).xy for t in np.linspace(0,polygon.length,k,False)])
        equi_points = equi_points.squeeze().T
        equi_points[:,0] -= 20.5
        equi_points[:,1] -= 10.5
        # equi_points = np.append(equi_points, equi_points[0,None], axis=0)
        holes.append(equi_points)
        len_holes.append(len(equi_points))

    # Smooth holes
    xmin = np.min(tissue_equi_points[:,0])
    xmax = np.max(tissue_equi_points[:,0])
    holes_smooth = []
    for hole in holes:
        h = np.vstack((hole, hole[0]))
        h = np.array(taubin_smooth(h))[:-1]
        h[h[:,0] < xmin, 0] = xmin
        h[h[:,0] > xmax, 0] = xmax

        poly_hole = sg.Polygon(h)
        if poly_hole.area < min_area:
            continue

        holes_smooth.append(h)
        
    holes = holes_smooth

    # Order holes based on len_holes
    holes = [hole for _, hole in sorted(zip(len_holes, holes), key=lambda x: x[0])][::-1]

    # Plot tissue_equi_points and holes
    fig, ax = plt.subplots(figsize=(40, 8))
    ax.plot(tissue_equi_points[:, 1], tissue_equi_points[:, 0], 'r.-', linewidth=2, label='Tissue Boundary')
    for hole in holes:
        ax.plot(hole[:, 1], hole[:, 0], 'b.-', linewidth=1, label='Hole')
    ax.set_aspect('equal')

    # mesh
    plt.figure(1)
    with pygmsh.occ.Geometry() as geom:

        borders = []
        for hole in tqdm(holes):
            borders.append(geom.add_polygon(hole,
                mesh_size=rescaled_meshsize
            ))

        # main surface
        tissue = geom.add_polygon(tissue_equi_points,
            mesh_size=rescaled_meshsize
        )

        # # Cutout the fiber boundary
        cuts = []
        for border in tqdm(borders):
            p = np.array([p.x for p in border.points])
            plt.plot(p[:, 1], p[:, 0], 'b.-', linewidth=1)
            cuts.append(geom.boolean_intersection([tissue, border], delete_first=False)[0])

        geom.boolean_fragments(tissue, cuts)

        mesh = geom.generate_mesh(verbose=True)

    points = mesh.points[:,0:2]
    tri_mesh = io.Mesh(points, {'triangle': mesh.cells_dict['triangle']})
    xyz = tri_mesh.points
    ien = tri_mesh.cells_dict['triangle']
    
    # Obtaining mask of fiber elements
    midpoints = np.mean(xyz[ien], axis=1)
    inholes = np.zeros(len(midpoints), dtype=int)
    spoints = [sg.Point(point) for point in midpoints]

    print('Finding fiber elems')
    for hole in tqdm(holes):
        poly = sg.polygon.Polygon(hole)
        marker = poly.contains(spoints)
        inholes[marker] = 1

    elem_fiber_mask = 1-inholes

    print('Removing disconnected cells')
    disconnected_cells = find_disconnected_cells(ien[elem_fiber_mask==1])
    cells = np.arange(len(ien))
    cells = cells[elem_fiber_mask==1]
    elem_fiber_mask[cells[disconnected_cells]] = 0

    # Check for isolated cells
    print('Finding fiber_mesh neighbors')
    fiber_elems = np.where(elem_fiber_mask==1)[0]
    elem_neigh = get_elem_neighbors(tri_mesh.cells[0].data[fiber_elems])
    non_neigh = np.sum(elem_neigh==-1, axis=1)
    isolated_cells = fiber_elems[np.where(non_neigh==3)[0]]

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

    tri_mesh.cell_data['fiber mask'] = [elem_fiber_mask]
    fiber_mesh, _, _ = create_submesh(tri_mesh, np.where(elem_fiber_mask==1)[0])
    xyz = fiber_mesh.points
    ien = fiber_mesh.cells_dict['triangle']

    elem_fiber_mask = np.ones(len(fiber_mesh.cells[0].data), dtype=int)
    belems, bfaces = get_surface_mesh(fiber_mesh)

    # Getting boundaries to fix bad elements
    print('Deleting bad elements on the boundaries')
    xmin = np.min(fiber_mesh.points[:,0])
    xmax = np.max(fiber_mesh.points[:,0])
    post1_faces = np.where(np.all(np.isclose(xyz[bfaces][:,:,0], xmin), axis=1))[0]
    post2_faces = np.where(np.all(np.isclose(xyz[bfaces][:,:,0], xmax), axis=1))[0]
    post_nodes = np.union1d(bfaces[post1_faces], bfaces[post2_faces])

    # Loop until all bad elements are fixed
    del_elems = 100
    it = 1
    while del_elems > 0:
        xyz, ien, elem_fiber_mask, del_elems = fix_bad_post_elements(xyz, ien, post_nodes, elem_fiber_mask)
        print('Iteration:', it, 'Deleted elements:', del_elems)
        it += 1

    xyz = xyz/rescale

    # Check if bad elements only have one neighbor
    fiber_mesh = io.Mesh(xyz, {'triangle': ien})
    belems, bfaces = get_surface_mesh(fiber_mesh)
    xyz, ien = fix_quality_one_neigh_elems(xyz, ien, belems)

    tri_mesh = io.Mesh(xyz, {'triangle': ien})
    _, bfaces = get_surface_mesh(tri_mesh)
    io.write('check.vtu', tri_mesh)
    
    # Subdivide fiber elements if needed
    if subdivide_fibers:
        print('Subdividing fiber elements')
        tri_mesh = subdivide_mesh_fibers(tri_mesh, elem_fiber_mask, bfaces)

    # Dummy process to get rid of isolated points
    fiber_mesh, _, _ = create_submesh(tri_mesh, np.arange(len(tri_mesh.cells[0].data)))

    # Create tissue mesh

    with pygmsh.occ.Geometry() as geom:
        # main surface
        tissue = geom.add_polygon(tissue_equi_points,
            mesh_size=rescaled_meshsize*2
        )

        mesh = geom.generate_mesh(verbose=True)

    points = mesh.points[:,0:2]
    tissue_mesh = io.Mesh(points, {'triangle': mesh.cells_dict['triangle']})
    tissue_mesh.points = tissue_mesh.points/rescale


    return fiber_mesh, tissue_mesh

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


def find_disconnected_cells(ien):
    num_cells = len(ien)
    visited = np.zeros(num_cells, dtype=bool)
    stack = [0]  # Start from the first cell

    cont = 0
    while stack:
        cell = stack.pop()
        cont += 1
        if not visited[cell]:
            visited[cell] = True
            neighbors = np.where(np.sum(np.isin(ien, ien[cell]), axis=1) > 0)[0]
            stack.extend(neighbors[~visited[neighbors]])
        if cont < 100 and len(stack) == 0:  # Making sure the first 100 cells are disconnected
            visited[cell] = False
            stack = [cont]

    disconnected_cells = np.where(~visited)[0]
    return disconnected_cells


def check_mesh_correctness():
    pass

def fix_bad_post_elements(xyz, ien, post_nodes, elem_fiber_mask):
    quality = tri_quality_aspect_ratio(xyz, ien)

    to_fix = np.where(quality > 7)[0]
    to_fix = np.union1d(to_fix, np.where(np.isnan(quality))[0])

    x0 = np.min(xyz[:,0])
    xl = np.max(xyz[:,0])
    tol = (xl-x0)/2000

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

            xyz[np.isclose(xyz[:,0], x0, atol=tol),0] = x0
            xyz[np.isclose(xyz[:,0], xl, atol=tol),0] = xl
        else:
            print(i)


    ien = np.delete(ien, elem_del, axis=0)
    elem_fiber_mask = np.delete(elem_fiber_mask, elem_del)

    # Check ien. Sometimes elements end up having same nodes
    elems1 = np.where(np.any((ien[:,1:]-ien[:,0][:,None] == 0), axis=1))[0]
    elems2 = np.where(np.any((ien[:,np.array([0,2])]-ien[:,1][:,None] == 0), axis=1))[0]
    elems3 = np.where(np.any((ien[:,np.array([1,2])]-ien[:,0][:,None] == 0), axis=1))[0]
    elem_del2 = np.union1d(np.union1d(elems1, elems2), elems3)

    if len(elem_del2) > 0:
        print(elem_del2)
        ien = np.delete(ien, elem_del2, axis=0)
        elem_fiber_mask = np.delete(elem_fiber_mask, elem_del2)

    io.write_points_cells('check.vtu', xyz, {'triangle': ien})
    return xyz, ien, elem_fiber_mask, len(elem_del)+len(elem_del2)


def fix_quality_one_neigh_elems(xyz, ien, belems):
    quality = tri_quality_radius_ratio(np.column_stack((xyz, np.zeros(len(xyz)))), ien)
    
    bad_elems = np.where(quality > 5)[0]

    list_bdry = np.zeros(len(ien), dtype=bool)
    list_bdry[belems] = True

    elem_del = [1,2]
    while len(elem_del) > 0:
        elem_neigh = get_elem_neighbors(ien)            # TODO: do this without recomputing
        elem_neigh_count = np.sum(elem_neigh != -1, axis=1)

        # sanity check
        zero_neigh_elems = np.where(elem_neigh_count == 0)[0]

        # Find elements with one neighbor
        one_neigh_elems = np.where(elem_neigh_count == 1)[0]

        elem_del = np.intersect1d(bad_elems, one_neigh_elems)
        elem_del = np.setdiff1d(elem_del, belems)
        elem_del = np.union1d(elem_del, zero_neigh_elems)

        ien = np.delete(ien, elem_del, axis=0)
        quality = np.delete(quality, elem_del)
        list_bdry = np.delete(list_bdry, elem_del, axis=0)
        belems = np.where(list_bdry)[0]

    return xyz, ien

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
