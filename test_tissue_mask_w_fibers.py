#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/07/07 14:33:32

@author: Javiera Jilberto Vallejos
'''

import numpy as np
from matplotlib import pyplot as plt
from skimage import io as skio
from img2mesh import clean_mask
from skimage import measure, morphology, transform, filters
from imregistration.utils import normalize_image

import shapely.geometry as sg
import meshio as io
import pygmsh

from tqdm import tqdm

tissue_fldr = '../DSP/Tissues/dataset2/gem02/exp/'
meshsize=5
print(meshsize)

rescale = 4
meshsize *= rescale

fiber_mask = skio.imread(tissue_fldr + 'pre_fiber_mask.tif')
tissue_mask = skio.imread(tissue_fldr + 'pre_tissue_mask.tif')

# Get contour using tissue_mask
mesh_tissue_mask = tissue_mask
mesh_tissue_mask = normalize_image(transform.rescale(mesh_tissue_mask, rescale))
mesh_tissue_mask = morphology.binary_closing(mesh_tissue_mask, footprint=morphology.disk(2))

# Generate mesh and grab node coordinates
mask = clean_mask(mesh_tissue_mask)
mask[0:10] = 0
mask[-10:] = 0

# Find boundary polygon
mask = np.pad(mask, pad_width=((20,20),(0,0)))
mask = filters.gaussian(mask) > 0.75
tissue_mask = mask
print(np.sum(tissue_mask))

# Find holes
mesh_fiber_mask = fiber_mask
mesh_fiber_mask = normalize_image(transform.rescale(mesh_fiber_mask, rescale))
# mesh_fiber_mask = morphology.binary_closing(mesh_fiber_mask, footprint=morphology.disk(2))

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
print(len(tissue_equi_points), np.sum(tissue_equi_points))

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
print(len(holes))

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


#%%

points = mesh.points[:,0:2]
tri_mesh = io.Mesh(points, {'triangle': mesh.cells_dict['triangle']})
line_mesh = io.Mesh(points, {'line': mesh.cells_dict['line']})
print(tri_mesh)

#%% quality
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


def fix_bad_elements(xyz, ien):
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

from scipy.spatial.distance import cdist

xyz = tri_mesh.points
ien = tri_mesh.cells_dict['triangle']
belem, bfaces = get_surface_mesh(tri_mesh)
bnodes = np.unique(bfaces)

post1_faces = np.where(np.all(xyz[bfaces][:,:,0] == 12, axis=1))[0]
post2_faces = np.where(np.all(xyz[bfaces][:,:,0] == 3934, axis=1))[0]
post_nodes = np.union1d(bfaces[post1_faces], bfaces[post2_faces])

del_elems = 100

it = 1
while del_elems > 0:
    xyz, ien, del_elems = fix_bad_elements(xyz, ien)
    print('Iteration:', it, 'Deleted elements:', del_elems)
    it += 1

print(ien.shape)

midpoints = np.mean(xyz[ien], axis=1)
inholes = np.zeros(len(midpoints))
spoints = [sg.Point(point) for point in midpoints]

for hole in tqdm(holes):
    poly = sg.polygon.Polygon(hole)
    marker = poly.contains(spoints)
    inholes[marker] = 1



tri_mesh = io.Mesh(xyz, {'triangle': ien})
print(tri_mesh)

io.write('mesh.vtu', tri_mesh)


#%%
xyz = xyz/rescale
plt.figure(1, clear=True, figsize=(25,15))
plt.imshow(tissue_mask.T, cmap='binary_r')
# plt.plot(tissue_equi_points[:,0], tissue_equi_points[:,1], 'k-')
# # for poly in holes:
# #     plt.plot(poly[:,0], poly[:,1], '-')
# # # plt.plot(boundary[:,0], boundary[:,1], 'o-')
# # # plt.plot(equi_points[:,0], equi_points[:,1], 'o')
plt.triplot(tri_mesh.points[:,0], tri_mesh.points[:,1], tri_mesh.cells_dict['triangle'])
# plt.gca().set_aspect('equal')
# # # plt.savefig('mesh.png', dpi=180)