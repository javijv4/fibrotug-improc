#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:37:46 2024

@author: Javiera Jilberto Vallejos
"""

import os
from glob import glob

from matplotlib import pyplot as plt
import numpy as np
import cheartio as chio
from skimage import io, filters, morphology, measure, exposure, transform, img_as_uint
import improcessing.fibers as fibproc
import improcessing.actin as actproc
import improcessing.dsp as dspproc
from imregistration.ITKTransform import elastix_simple_transformation, apply_transform
from imregistration.utils import rotate_interactive, crop_interactive, normalize_image
from img2mesh import mask2mesh, mask2mesh_with_fibers, mask2mesh_only_fibers, find_boundary



class DSPProtocolTissue:
    def __init__(self, folder, pre_tissue, post_tissue, fixed_image, out_fldr=None, flip180=False, recompute_crop=False) -> None:
        self.folder = folder
        if out_fldr is None:
            self.out_fldr = folder
        else:
            self.out_fldr = out_fldr
        self.mesh_folder = out_fldr + 'mesh/'
        self.data_folder = out_fldr + 'data/'

        self.pre_tissue = pre_tissue
        self.post_tissue = post_tissue

        # Getting tissue masks
        pre_tissue.grab_all_data()
        post_tissue.grab_all_data()

        # Flip post in 180 degrees if needed
        if flip180:
            for name in self.post_tissue.images.keys():
                if self.post_tissue.images[name] is not None:
                    self.post_tissue.images[name] = transform.rotate(self.post_tissue.images[name], 180)

        post_tissue_mask = self.post_tissue.tissue_mask
        pre_tissue_mask = self.pre_tissue.tissue_mask

        pre_fiber = pre_tissue.fiber_image
        post_fiber = post_tissue.fiber_image

        # Obtaining rotation and translation parameters
        self.fixed_image = fixed_image
        if fixed_image == 'pre':
            self.rot_crop_params = self.find_rotate_and_crop(pre_fiber, pre_tissue_mask, fixed_image, force_compute=recompute_crop)
        else:
            self.rot_crop_params = self.find_rotate_and_crop(post_fiber, post_tissue_mask, fixed_image, force_compute=recompute_crop)

        # Initializing image dictionaries
        self.pre_images = {}
        self.post_images = {}


    def generate_tissue_mesh(self, meshsize=5, pixel_size=0.390*1e-3, use_fiber_mask=False):
        if self.fixed_image == 'pre':
            tissue_mask = self.pre_images['tissue_mask']
        else:
            tissue_mask = self.post_images['tissue_mask']

        self.mesh_tissue_mask = tissue_mask

        # Generate mesh and grab node coordinates
        if use_fiber_mask:
            fiber_mask = self.pre_images['fiber_mask']
            mesh, elem_fiber_mask, _ = mask2mesh_with_fibers(tissue_mask, fiber_mask, meshsize=meshsize)
        else:
            fiber_mask = np.zeros_like(tissue_mask)
            mesh, elem_fiber_mask, _ = mask2mesh_with_fibers(tissue_mask, fiber_mask, meshsize=meshsize)
            # mesh = mask2mesh(tissue_mask, meshsize)
        print(np.min(mesh.points,axis=0), np.max(mesh.points,axis=0))
        ij_nodes = np.floor(mesh.points).astype(int)

        # Project data to mesh
        for field in self.pre_images.keys():
            if 'mask' in field:
                mesh.point_data['pre_' + field] = self.pre_images[field][ij_nodes[:,0], ij_nodes[:,1]].astype(int)
            else:
                mesh.point_data['pre_' + field] = self.pre_images[field][ij_nodes[:,0], ij_nodes[:,1]]
        for field in self.post_images.keys():
            if 'mask' in field:
                mesh.point_data['post_' + field] = self.post_images[field][ij_nodes[:,0], ij_nodes[:,1]].astype(int)
            else:
                mesh.point_data['post_' + field] = self.post_images[field][ij_nodes[:,0], ij_nodes[:,1]]


        # Calculate fiber and myofibril vectors
        pre_actin_vector = np.array([np.cos(mesh.point_data['pre_actin_angle']), np.sin(mesh.point_data['pre_actin_angle'])]).T
        post_actin_vector = np.array([np.cos(mesh.point_data['post_actin_angle']), np.sin(mesh.point_data['post_actin_angle'])]).T

        mesh.point_data['pre_actin_vector'] = np.hstack([pre_actin_vector, np.zeros((pre_actin_vector.shape[0], 1))])
        mesh.point_data['post_actin_vector'] = np.hstack([post_actin_vector, np.zeros((post_actin_vector.shape[0], 1))])

        # Fiber density to elements
        midpoints = np.mean(mesh.points[mesh.cells[0].data], axis=1)
        ij_midnodes = np.floor(midpoints).astype(int)

        # Remove any fiber density outside the fiber mask
        if use_fiber_mask:
            pre_elem_fib_rho = elem_fiber_mask
        else:
            pre_elem_fib_rho = self.pre_images['fiber_density'][ij_midnodes[:,0], ij_midnodes[:,1]]
        mesh.cell_data['pre_fiber_density_elem'] = [pre_elem_fib_rho]
        post_elem_fib_rho = self.post_images['fiber_density'][ij_midnodes[:,0], ij_midnodes[:,1]]
        mesh.cell_data['post_fiber_density_elem'] = [post_elem_fib_rho]

        # Adjust to real dimensions
        mesh.points = mesh.points * pixel_size

        # Find boundary
        bdata = find_boundary(mesh)

        # Compute width at boundary
        nodes_b1 = np.unique(bdata[bdata[:,-1]==1, 1:-1])
        nodes_b2 = np.unique(bdata[bdata[:,-1]==2, 1:-1])

        w1 = np.max(mesh.points[nodes_b1,1]) - np.min(mesh.points[nodes_b1,1])
        w2 = np.max(mesh.points[nodes_b2,1]) - np.min(mesh.points[nodes_b2,1])

        # Save mesh
        chio.write_mesh(self.mesh_folder + 'tissue', mesh.points, mesh.cells[0].data)
        chio.write_bfile(self.mesh_folder + 'tissue', bdata)

        # Save data points
        print('Saving tissue data...')
        for field in  mesh.point_data:
            name = field
            if 'fiber' in field:
                name = name + '_t'
            print(field)
            if 'vector' in field:
                aux = np.array([mesh.point_data[field][:,1], -mesh.point_data[field][:,0]]).T
                save = np.hstack([mesh.point_data[field][:,0:2], aux])
                chio.write_dfile(self.data_folder + name + '.FE', save)
            else:
                chio.write_dfile(self.data_folder + name + '.FE', mesh.point_data[field])
        for field in  mesh.cell_data:
            name = field
            if 'fiber' in field:
                name = name + '_t'
            print(field)
            chio.write_dfile(self.data_folder + name + '.FE', mesh.cell_data[field][0])

        chio.write_dfile(self.data_folder + 'w1.FE', np.array([w1]))
        chio.write_dfile(self.data_folder + 'w2.FE', np.array([w2]))

        self.mesh = mesh
        return mesh
    
    def generate_fiber_mesh(self, meshsize=3, pixel_size=0.390*1e-3):
        if self.fixed_image == 'pre':
            tissue_mask = self.pre_images['tissue_mask']
        else:
            tissue_mask = self.post_images['tissue_mask']

        self.mesh_tissue_mask = tissue_mask

        # Generate mesh and grab node coordinates
        fiber_mask = self.pre_images['fiber_mask']
        mesh = mask2mesh_only_fibers(tissue_mask, fiber_mask, meshsize=meshsize, subdivide_fibers=True)
        ij_nodes = np.floor(mesh.points).astype(int)

        # Project data to mesh
        for field in self.pre_images.keys():
            if 'mask' in field:
                mesh.point_data['pre_' + field] = self.pre_images[field][ij_nodes[:,0], ij_nodes[:,1]].astype(int)
            else:
                mesh.point_data['pre_' + field] = self.pre_images[field][ij_nodes[:,0], ij_nodes[:,1]]
        for field in self.post_images.keys():
            if 'mask' in field:
                mesh.point_data['post_' + field] = self.post_images[field][ij_nodes[:,0], ij_nodes[:,1]].astype(int)
            else:
                mesh.point_data['post_' + field] = self.post_images[field][ij_nodes[:,0], ij_nodes[:,1]]


        # Calculate fiber and myofibril vectors
        pre_fiber_vector = np.array([np.cos(mesh.point_data['pre_fiber_angle']), np.sin(mesh.point_data['pre_fiber_angle'])]).T
        post_fiber_vector = np.array([np.cos(mesh.point_data['post_fiber_angle']), np.sin(mesh.point_data['post_fiber_angle'])]).T

        mesh.point_data['pre_fiber_vector'] = np.hstack([pre_fiber_vector, np.zeros((pre_fiber_vector.shape[0], 1))])
        mesh.point_data['post_fiber_vector'] = np.hstack([post_fiber_vector, np.zeros((post_fiber_vector.shape[0], 1))])

        # Fiber density to elements
        midpoints = np.mean(mesh.points[mesh.cells[0].data], axis=1)
        ij_midnodes = np.floor(midpoints).astype(int)

        # Remove any fiber density outside the fiber mask
        pre_elem_fib_rho = np.ones(mesh.cells[0].data.shape[0])
        mesh.cell_data['pre_fiber_density_elem'] = [pre_elem_fib_rho]
        post_elem_fib_rho = self.post_images['fiber_density'][ij_midnodes[:,0], ij_midnodes[:,1]]
        mesh.cell_data['post_fiber_density_elem'] = [post_elem_fib_rho]

        # Adjust to real dimensions
        mesh.points = mesh.points * pixel_size

        # Find boundary
        bdata = find_boundary(mesh)

        # Save mesh
        chio.write_mesh(self.mesh_folder + 'fiber', mesh.points, mesh.cells[0].data)
        chio.write_bfile(self.mesh_folder + 'fiber', bdata)

        # Save data points
        print('Saving fiber data...')
        for field in  mesh.point_data:
            if 'fiber' not in field:
                continue
            print(field)
            if 'vector' in field:
                aux = np.array([mesh.point_data[field][:,1], -mesh.point_data[field][:,0]]).T
                save = np.hstack([mesh.point_data[field][:,0:2], aux])
                chio.write_dfile(self.data_folder + field + '.FE', save)
            else:
                chio.write_dfile(self.data_folder + field + '.FE', mesh.point_data[field])
        for field in  mesh.cell_data:
            chio.write_dfile(self.data_folder + field + '.FE', mesh.cell_data[field][0])

        self.mesh = mesh
        return mesh


    def generate_mesh(self, meshsize=5, pixel_size=0.390*1e-3, use_fiber_mask=False, add_posts=False, subdivide_fibers=False):
        if self.fixed_image == 'pre':
            tissue_mask = self.pre_images['tissue_mask']
        else:
            tissue_mask = self.post_images['tissue_mask']

        self.mesh_tissue_mask = tissue_mask

        # Generate mesh and grab node coordinates
        if use_fiber_mask:
            fiber_mask = self.pre_images['fiber_mask']
            mesh, elem_fiber_mask, fiber_mesh = mask2mesh_with_fibers(tissue_mask, fiber_mask, meshsize=meshsize, add_posts=add_posts, subdivide_fibers=subdivide_fibers)
        else:
            mesh = mask2mesh(tissue_mask, meshsize)
        ij_nodes = np.floor(mesh.points).astype(int)

        # Project data to mesh
        for field in self.pre_images.keys():
            if 'mask' in field:
                mesh.point_data['pre_' + field] = self.pre_images[field][ij_nodes[:,0], ij_nodes[:,1]].astype(int)
            else:
                mesh.point_data['pre_' + field] = self.pre_images[field][ij_nodes[:,0], ij_nodes[:,1]]
        for field in self.post_images.keys():
            if 'mask' in field:
                mesh.point_data['post_' + field] = self.post_images[field][ij_nodes[:,0], ij_nodes[:,1]].astype(int)
            else:
                mesh.point_data['post_' + field] = self.post_images[field][ij_nodes[:,0], ij_nodes[:,1]]


        # Calculate fiber and myofibril vectors
        pre_fiber_vector = np.array([np.cos(mesh.point_data['pre_fiber_angle']), np.sin(mesh.point_data['pre_fiber_angle'])]).T
        pre_actin_vector = np.array([np.cos(mesh.point_data['pre_actin_angle']), np.sin(mesh.point_data['pre_actin_angle'])]).T
        post_fiber_vector = np.array([np.cos(mesh.point_data['post_fiber_angle']), np.sin(mesh.point_data['post_fiber_angle'])]).T
        post_actin_vector = np.array([np.cos(mesh.point_data['post_actin_angle']), np.sin(mesh.point_data['post_actin_angle'])]).T

        mesh.point_data['pre_fiber_vector'] = np.hstack([pre_fiber_vector, np.zeros((pre_fiber_vector.shape[0], 1))])
        mesh.point_data['post_fiber_vector'] = np.hstack([post_fiber_vector, np.zeros((post_fiber_vector.shape[0], 1))])
        mesh.point_data['pre_actin_vector'] = np.hstack([pre_actin_vector, np.zeros((pre_actin_vector.shape[0], 1))])
        mesh.point_data['post_actin_vector'] = np.hstack([post_actin_vector, np.zeros((post_actin_vector.shape[0], 1))])

        # Fiber density to elements
        midpoints = np.mean(mesh.points[mesh.cells[0].data], axis=1)
        ij_midnodes = np.floor(midpoints).astype(int)

        # Remove any fiber density outside the fiber mask
        if use_fiber_mask:
            pre_elem_fib_rho = elem_fiber_mask
        else:
            pre_elem_fib_rho = self.pre_images['fiber_density'][ij_midnodes[:,0], ij_midnodes[:,1]]
        mesh.cell_data['pre_fiber_density_elem'] = [pre_elem_fib_rho]
        post_elem_fib_rho = self.post_images['fiber_density'][ij_midnodes[:,0], ij_midnodes[:,1]]
        mesh.cell_data['post_fiber_density_elem'] = [post_elem_fib_rho]

        # Adjust to real dimensions
        mesh.points = mesh.points * pixel_size

        # Find boundary
        bdata = find_boundary(mesh)
        if use_fiber_mask:
            fiber_mesh.points = fiber_mesh.points * pixel_size
            fiber_bdata = find_boundary(fiber_mesh)

        # Compute width at boundary
        nodes_b1 = np.unique(bdata[bdata[:,-1]==1, 1:-1])
        nodes_b2 = np.unique(bdata[bdata[:,-1]==2, 1:-1])

        w1 = np.max(mesh.points[nodes_b1,1]) - np.min(mesh.points[nodes_b1,1])
        w2 = np.max(mesh.points[nodes_b2,1]) - np.min(mesh.points[nodes_b2,1])

        # Save mesh
        chio.write_mesh(self.mesh_folder + 'tissue', mesh.points, mesh.cells[0].data)
        chio.write_bfile(self.mesh_folder + 'tissue', bdata)

        if use_fiber_mask:
            chio.write_mesh(self.mesh_folder + 'fiber', fiber_mesh.points, fiber_mesh.cells[0].data)
            chio.write_bfile(self.mesh_folder + 'fiber', fiber_bdata)

        # Save data points
        for field in  mesh.point_data:
            if 'vector' in field:
                aux = np.array([mesh.point_data[field][:,1], -mesh.point_data[field][:,0]]).T
                save = np.hstack([mesh.point_data[field][:,0:2], aux])
                chio.write_dfile(self.data_folder + field + '.FE', save)
            else:
                chio.write_dfile(self.data_folder + field + '.FE', mesh.point_data[field])
        for field in  mesh.cell_data:
            chio.write_dfile(self.data_folder + field + '.FE', mesh.cell_data[field][0])

        chio.write_dfile(self.data_folder + 'w1.FE', np.array([w1]))
        chio.write_dfile(self.data_folder + 'w2.FE', np.array([w2]))

        self.mesh = mesh
        return mesh



    def find_rotate_and_crop(self, image, mask, fixed_image, force_compute=False):
        if not force_compute:
            try:
                params = np.load(self.folder + fixed_image + 'rotation_crop_params.npy')
                return params
            except:
                pass

        # Rotation
        rot = np.zeros(1)
        rot = rotate_interactive(image, rot)[0]

        # Crop
        box = crop_interactive(image, rot)

        params = np.array([rot, *box])
        np.save(self.folder + fixed_image + 'rotation_crop_params.npy', params)
        return params


    def rotate_and_crop(self, image):
        rot = self.rot_crop_params[0]
        crop = self.rot_crop_params[1:].astype(int)

        image = transform.rotate(image, rot)
        image = image[crop[2]:crop[3], crop[0]:crop[1]]
        return image


    def register_to_fixed(self, fixed_image, mode='affine'):
        pre_mask = self.pre_tissue.images['tissue_mask'].astype(np.float32)
        post_mask = self.post_tissue.images['tissue_mask'].astype(np.float32)

        # Elastix transform
        if fixed_image == 'pre':
            transform_params = elastix_simple_transformation(pre_mask, post_mask, mode=mode)
            post_tissue_mask = self.warp_mask(post_mask, transform_params)
            pre_tissue_mask = self.rot_crop_mask(pre_mask)

            # Find secondary warping
            secondary_warping = self.find_secondary_warping(pre_tissue_mask, post_tissue_mask)

            # Apply secondary warping
            post_tissue_mask = self.apply_secondary_warping(post_tissue_mask, secondary_warping)

            self.post_images = self.warp_tissue_images(self.post_tissue, transform_params, secondary_warping=secondary_warping)
            self.pre_images = self.rot_crop_tissue_images(self.pre_tissue)
            self.pre_images['tissue_mask'] = pre_tissue_mask
            self.post_images['tissue_mask'] = post_tissue_mask

        else:
            transform_params = elastix_simple_transformation(post_mask, pre_mask, mode=mode)
            pre_tissue_mask = self.warp_mask(pre_mask, transform_params)
            post_tissue_mask = self.rot_crop_mask(post_mask)

            self.pre_images = self.warp_tissue_images(self.pre_tissue, transform_params)
            self.post_images = self.rot_crop_tissue_images(self.post_tissue)
            self.pre_images['tissue_mask'] = pre_tissue_mask
            self.post_images['tissue_mask'] = post_tissue_mask


    def warp_mask(self, mask, transform_params):
        warped_mask = apply_transform(mask, transform_params)
        warped_mask = self.rotate_and_crop(warped_mask)
        warped_mask = normalize_image(warped_mask, binary=True)
        return warped_mask


    def rot_crop_mask(self, mask):
        mask = self.rotate_and_crop(mask)
        mask = normalize_image(mask, binary=True)
        return mask


    def find_secondary_warping(self, fixed_mask, moving_mask):
        from scipy.interpolate import interp1d

        warping_width = np.zeros(fixed_mask.shape)
        x_coord = np.arange(fixed_mask.shape[1])

        fixed_lims = np.zeros([fixed_mask.shape[0], 2])
        moving_lims = np.zeros([moving_mask.shape[0], 2])
        for i in range(fixed_mask.shape[0]):
            fixed_lims[i] = np.array([np.min(np.where(fixed_mask[i]>0)), np.max(np.where(fixed_mask[i]>0))])
            moving_lims[i] = np.array([np.min(np.where(moving_mask[i]>0)), np.max(np.where(moving_mask[i]>0))])

        midline = np.mean(fixed_lims, axis=1)
        disp = fixed_lims - moving_lims

        mov_points = np.array([moving_lims[:,0], midline, moving_lims[:,1]]).T
        disp_vals = np.array([disp[:,0], np.zeros(len(disp)), disp[:,1]]).T

        sizex = moving_mask.shape[1]
        for i in range(fixed_mask.shape[0]):
            if np.isclose(mov_points[i,0], 0):
                mpoints = np.append(mov_points[i], sizex)
                dvals = np.append(disp_vals[i], disp_vals[i,-1])
            elif np.isclose(mov_points[i,-1], sizex-1):
                mpoints = np.append(0, mov_points[i])
                dvals = np.append(disp_vals[i,0], disp_vals[i])
            else:
                mpoints = np.append(0, mov_points[i])
                mpoints = np.append(mpoints, sizex)
                dvals = np.append(disp_vals[i,0], disp_vals[i])
                dvals = np.append(dvals, disp_vals[i,-1])

            f = interp1d(mpoints, dvals, fill_value='extrapolate')
            warping_width[i] = f(x_coord)

        warping_width = filters.gaussian(warping_width, sigma=(20,2))

        def warp_func(ij):
            ij = ij.astype(int)
            dispx = np.zeros(len(ij))
            dispy = -warping_width[ij[:,1], ij[:,0]]
            disp = np.stack([dispy, dispx], axis=-1)
            return ij + disp

        return warp_func


    def apply_secondary_warping(self, image, warp_func):
        warped_image = transform.warp(image, warp_func)
        return warped_image



    def warp_tissue_images(self, tissue, transform_params, secondary_warping=None):
        dict_images = {}

        # warp mask
        img = tissue.images['tissue_mask']
        warped_img = apply_transform(img, transform_params)
        tissue_mask = normalize_image(warped_img, binary=True)

        for name in tissue.images.keys():

            img = tissue.images[name]
            if img is None:
                continue
            if 'mask' in name:
                if img.dtype == bool:
                    img = img.astype(int)
                else:
                    img = normalize_image(img)
                    img = img > 0.5
                    img = img.astype(int)

            # Normalize data
            vmin = img.min()
            vmax = img.max()

            img = normalize_image(img)

            # warp
            warped_img = apply_transform(img, transform_params)
            warped_img = normalize_image(warped_img, mask=tissue_mask)

            warped_img = warped_img*(vmax-vmin) + vmin

            # rotate and crop
            warped_img = self.rotate_and_crop(warped_img)

            # Apply secondary warping
            if secondary_warping is not None:
                warped_img = self.apply_secondary_warping(warped_img, secondary_warping)


            if 'mask' in name:
                img = normalize_image(img)
                img = img > 0.5
                img = img.astype(int)

            dict_images[name] = warped_img


        return dict_images


    def rot_crop_tissue_images(self, tissue):
        dict_images = {}
        # Rotate and crop mask
        img = tissue.images['tissue_mask']
        img = self.rotate_and_crop(img)
        tissue_mask = normalize_image(img, binary=True)


        for name in tissue.images.keys():
            # if 'mask' in name: continue
            img = tissue.images[name]
            if img is None:
                continue
            if 'mask' in name:
                if img.dtype == bool:
                    img = img.astype(int)
                else:
                    img = normalize_image(img)
                    img = img > 0.5
                    img = img.astype(int)

            # Normalize data
            vmin = img.min()
            vmax = img.max()

            img = normalize_image(img)

            img = self.rotate_and_crop(img)
            img = img*(vmax-vmin) + vmin

            if 'mask' in name:
                img = normalize_image(img)
                img = img > 0.5
                img = img.astype(int)

            dict_images[name] = img

            if name == 'fibers' or name == 'actin'  or name == 'dsp':
                dict_images[name] = normalize_image(img, mask = tissue_mask)

        return dict_images


    def plot_warped_mask(self):
        pre_mask = self.pre_images['tissue_mask'].astype(float)
        post_mask = self.post_images['tissue_mask'].astype(float)

        pre_mask[pre_mask==0] = np.nan
        post_mask[post_mask==0] = np.nan

        fig, axs = plt.subplots(1, 3, clear=True, num=1)
        axs[0].imshow(self.pre_tissue.images['tissue_mask'], cmap='Reds', vmin=0, vmax=1)
        axs[1].imshow(self.post_tissue.images['tissue_mask'], cmap='Blues', vmin=0, vmax=np.max(self.post_tissue.images['tissue_mask']))
        axs[2].imshow(pre_mask, cmap='Reds', alpha=0.8, vmin=0, vmax=1)
        axs[2].imshow(post_mask, cmap='Blues', alpha=0.8, vmin=0, vmax=1)

        for ax in axs:
            ax.axis('off')


    def plot_images(self, which=['tissue_mask', 'fibers', 'actin', 'dsp']):
        nimages = len(which)
        _, axes = plt.subplots(2, nimages, figsize=(15, 10), clear=True)

        cont = 0
        for _, name in enumerate(which):
            image = self.pre_images[name]
            if image is None:
                continue
            lims = [0, 1]
            if 'density' in name:
                cmap='viridis'
            elif 'angle' in name:
                cmap='RdBu'
                lims = [-np.pi/4, np.pi/4]
            elif 'dispersion' in name:
                cmap='magma'
                lims = [0,0.5]
            else:
                cmap='gray'

            axes[0, cont].imshow(image, cmap=cmap, vmin=lims[0], vmax=lims[1])
            axes[0, cont].set_title(f'Pre {name}')
            axes[0, cont].axis('off')
            cont += 1
        cont = 0
        for _, name in enumerate(which):
            image = self.post_images[name]
            if image is None:
                continue
            lims = [0, 1]
            if 'density' in name:
                cmap='viridis'
            elif 'angle' in name:
                cmap='RdBu'
                lims = [-np.pi/4, np.pi/4]
            elif 'dispersion' in name:
                cmap='magma'
                lims = [0,0.5]
            else:
                cmap='gray'
            axes[1, cont].imshow(image, cmap=cmap, vmin=lims[0], vmax=lims[1])
            axes[1, cont].set_title(f'Post {name}')
            axes[1, cont].axis('off')
            cont += 1
        plt.tight_layout()


    def save_images(self, folder):
        for name in self.pre_images.keys():
            image = self.pre_images[name]
            if 'mask' in name:
                image = image.astype(np.uint8)
                io.imsave(folder + 'pre_' + name + '.tif', image, check_contrast=False)
            elif ('density' in name) or ('angle' in name) or ('dispersion' in name):
                io.imsave(folder + 'pre_' + name + '.tif', image, check_contrast=False)
            else:
                io.imsave(folder + 'pre_' + name + '.tif', img_as_uint(image), check_contrast=False)
        for name in self.post_images.keys():
            image = self.post_images[name]
            if 'mask' in name:
                image = image.astype(np.uint8)
                io.imsave(folder + 'post_' + name + '.tif', image, check_contrast=False)
            elif ('density' in name) or ('angle' in name) or ('dispersion' in name):
                io.imsave(folder + 'post_' + name + '.tif', image, check_contrast=False)
            else:
                io.imsave(folder + 'post_' + name + '.tif', img_as_uint(image), check_contrast=False)


class FtugTissue:
    def __init__(self, tissue_fldr, images) -> None:
        self.tissue_fldr = tissue_fldr
        self.fiber_image = None
        self.actin_image = None
        self.dsp_image = None
        self.dapi_image = None

        # If fiber, actin, dsp, or dapi images are provided, set them
        if 'fibers' in images:
            self.fiber_image = io.imread(tissue_fldr + images['fibers'])
        if 'actin' in images:
            self.actin_image = io.imread(tissue_fldr + images['actin'])
        if 'dsp' in images:
            self.dsp_image = io.imread(tissue_fldr + images['dsp'])
        if 'dapi' in images:
            self.dapi_image = io.imread(tissue_fldr + images['dapi'])

        # if self.fiber_image is None:
        #     raise ValueError('Fiber image is required')

        self.tissue_mask = None
        self.fiber_mask = None

        self.images = {'fibers': self.fiber_image, 'actin': self.actin_image, 'dsp': self.dsp_image, 'dapi': self.dapi_image}


    def get_tissue_mask(self, tissue_mask_type='both', force_compute=False):
        if not force_compute:
            try:
                self.tissue_mask = io.imread(self.tissue_fldr + '/tissue_mask.tif')
                self.images['tissue_mask'] = self.tissue_mask
                print('Loaded tissue mask from file ' + self.tissue_fldr + '/tissue_mask.tif')
                return self.tissue_mask
            except:
                pass

        print('Computing tissue mask')        
        if tissue_mask_type == 'both':
            if self.fiber_image is None:
                raise ValueError('Fiber image is required')
            if self.actin_image is None:    
                raise ValueError('Actin image is required')
    
            tissue_mask_from_actin = self.get_tissue_mask_from_image(self.actin_image)
            tissue_mask_from_fibers = self.get_tissue_mask_from_image(self.fiber_image)
            tissue_mask = tissue_mask_from_actin + tissue_mask_from_fibers
            
        elif tissue_mask_type == 'fibers':
            if self.fiber_image is None:
                raise ValueError('Fiber image is required')
            tissue_mask = self.get_tissue_mask_from_image(self.fiber_image)
            tissue_mask = morphology.binary_opening(tissue_mask, morphology.disk(15))

        elif tissue_mask_type == 'actin':
            if self.actin_image is None:    
                raise ValueError('Actin image is required')
            tissue_mask = self.get_tissue_mask_from_image(self.actin_image)

        self.tissue_mask = tissue_mask
        io.imsave(self.tissue_fldr + '/tissue_mask_init.tif', tissue_mask.astype(np.int8), check_contrast=False)
        self.images['tissue_mask'] = self.tissue_mask

        return tissue_mask


    def get_tissue_mask_from_image(self, image, block_size=None, remove_size=5, closing_block_size=5):
        # Generating initial mask
        if block_size is None:
            block_size = np.floor(np.min(image.shape)/10)*2+1
        thresh = filters.threshold_local(image, block_size=block_size)
        mask = image > thresh

        # Removing noise due to local thresholding
        mask = morphology.remove_small_objects(mask, remove_size)
        mask = morphology.binary_closing(mask)

        # get largest cluster
        labelled = measure.label(mask, connectivity=1)
        rp = measure.regionprops(labelled)
        sizes = ([i.area for i in rp])
        mask = labelled == (np.argmax(sizes)+1)

        # Trying to fill any gap in the boundary
        mask = morphology.binary_closing(mask, footprint=morphology.disk(closing_block_size))

        # Reconstruct mask
        # https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_holes_and_peaks.html#sphx-glr-auto-examples-features-detection-plot-holes-and-peaks-py
        seed = np.copy(mask)
        seed[1:-1, 1:-1] = mask.max()
        mask = morphology.reconstruction(seed, mask, method='erosion')

        mask_size = np.sum(mask)

        # Fill any holes in the mask
        mask = mask.astype(bool)
        mask = morphology.remove_small_holes(mask, area_threshold=mask_size//8)

        return mask


    """
    FIBER PROCESSING
    """
    def get_fiber_mask(self, force_compute=False):
        if not force_compute:
            try:
                self.fiber_mask = io.imread(self.tissue_fldr + '/fiber_mask.tif')
                print('Loaded fiber mask from file ' + self.tissue_fldr + '/fiber_mask.tif')
                return self.fiber_mask
            except:
                pass

        print('Computing fiber mask')
        image = self.fiber_image.copy()
        image[self.tissue_mask==0] = 0
        image_hist = filters.gaussian(image, sigma=1)

        hist = np.histogram(image_hist.ravel(), bins=128)

        thresh = filters.threshold_minimum(image_hist[self.tissue_mask==1].ravel(), hist=hist)

        image[image<thresh] = 0

        resize_image = transform.rescale(image, 2)
        resize_image = exposure.equalize_adapthist(resize_image, kernel_size=11, clip_limit=0.05)
        resize_mask = resize_image > filters.threshold_otsu(resize_image)
        resize_mask = morphology.binary_opening(resize_mask, morphology.disk(1))

        mask = transform.rescale(resize_mask.astype(float), 0.5, order=3) > 0.5
        mask = filters.gaussian(mask, sigma=1) > 0.5

        # Extract largest connected component
        labelled = measure.label(mask, connectivity=1)
        rp = measure.regionprops(labelled)
        sizes = ([i.area for i in rp])
        mask = labelled == (np.argmax(sizes)+1)

        io.imsave(self.tissue_fldr + '/fiber_mask_init.tif', mask.astype(np.int8), check_contrast=False)

        mask[self.tissue_mask==0] = 0
        self.fiber_mask = mask

        return mask


    def process_fibers(self, force_compute = False, sigma_density = 1, averaging_window = 7):
        if not force_compute:
            try:
                data = np.load(self.tissue_fldr + 'improc_fiber.npz')
                self.fiber_angle = data['angles']
                self.fiber_mask = data['mask']
                self.fiber_dispersion = data['dispersion']
                self.fiber_density = data['density']
                self.images['fiber_density'] = self.fiber_density
                self.images['fiber_angle'] = self.fiber_angle
                self.images['fiber_dispersion'] = self.fiber_dispersion
                self.images['fiber_mask'] = self.fiber_mask

                print('Processed fibers loaded from file' + self.tissue_fldr + 'improc_fiber.npz')
                return
            except:
                pass

        print('Processing fibers')
        mask = self.fiber_mask.copy()
        self.fiber_density = fibproc.compute_local_density(mask, sigma_density)

        # Display the processed image
        theta = fibproc.compute_local_orientation(mask)

        # First smooth angles
        self.fiber_angle = fibproc.smooth_fiber_angles(theta, mask, window_size=averaging_window)

        # Compute dispersion
        smooth_theta_disp = fibproc.smooth_fiber_angles(theta, mask, window_size=5)
        self.fiber_dispersion = fibproc.compute_fiber_dispersion(smooth_theta_disp, mask, window_size=averaging_window)

        self.images['fiber_density'] = self.fiber_density
        self.images['fiber_angle'] = self.fiber_angle
        self.images['fiber_dispersion'] = self.fiber_dispersion
        self.images['fiber_mask'] = self.fiber_mask

        np.savez(self.tissue_fldr + 'improc_fiber', angles=self.fiber_angle, mask=self.fiber_mask,
                 dispersion=self.fiber_dispersion, density=self.fiber_density)


    """
    ACTIN PROCESSING
    """
    def get_actin_blob_mask(self, force_compute=False, eq_block_size=49, blob_threshold=0.9,
                              eccentricity_threshold=0.9, blob_min_size=100):
        image = self.actin_image.copy()
        image = actproc.prepare_image(image, eq_block_size=eq_block_size)

        # Try loading the image
        if not force_compute:
            try:
                self.blobs_mask = io.imread(self.tissue_fldr + 'blob_mask.tif')
                print('Loaded actin blob mask from file' + self.tissue_fldr + 'blob_mask.tif')
            except:
                pass

            try:
                self.interpolate_mask = io.imread(self.tissue_fldr + 'interpolate_mask.tif')
                print('Loaded interpolate mask from file' + self.tissue_fldr + 'interpolate_mask.tif')
            except:
                pass

            self.actin_image_filtered = image
            if hasattr(self, 'blobs_mask') and hasattr(self, 'interpolate_mask'):
                return

        print('Computing actin blob mask')
        # If it doesn't exist, generate it
        if not hasattr(self, 'blobs_mask'):
            blobs_mask = actproc.clean_blobs_mask(image,
                                                blob_threshold=blob_threshold,
                                                eccentricity_threshold=eccentricity_threshold,
                                                blob_min_size=blob_min_size)
            blobs_mask = transform.rescale(blobs_mask, 0.5)

            # Saving mask in folder
            io.imsave(self.tissue_fldr + 'blob_mask_init.tif',
            blobs_mask.astype(np.int8), check_contrast=False)
            io.imsave(self.tissue_fldr + 'interpolate_mask_init.tif',
            blobs_mask.astype(np.int8), check_contrast=False)

            self.blobs_mask = blobs_mask
        
        if not hasattr(self, 'interpolate_mask'):
            self.interpolate_mask = self.blobs_mask.copy()
        self.actin_image_filtered = image

    def get_actin_mask(self, force_compute=False, thresholds=[0.9,0.8,0.7,0.6,0.5,0.4], dilation=[4,4,3,3,2,2]):
        if not force_compute:
            try:
                self.actin_mask = io.imread(self.tissue_fldr + 'actin_mask.tif')
                self.actin_angle = np.load(self.tissue_fldr + 'actin_angle.npy')
                print('Loaded actin mask from file' + self.tissue_fldr + 'actin_mask.tif')
                return
            except:
                pass

        print('Computing actin mask')
        image = self.actin_image_filtered.copy()
        blobs_mask = transform.rescale(self.blobs_mask, 2) > 0

        # Iterative thresholding process
        results, remove_mask = actproc.actin_iterative_thresholding(image, thresholds, dilation=dilation, blobs_mask=blobs_mask)

        # Local thresholding
        self.actin_angle, remove_mask = actproc.actin_local_thresholding(image, results, remove_mask, blobs_mask=blobs_mask)
        np.save(self.tissue_fldr + 'actin_angle.npy', self.actin_angle)

        # Compute Sarcomere Mask
        actin_mask = actproc.compute_myofibril_mask(self.actin_angle)
        actin_mask = transform.rescale(actin_mask, 0.5) > 0
        actin_mask[self.tissue_mask==0] = 0

        # Saving mask in folder
        io.imsave(self.tissue_fldr + 'actin_mask_init.tif',
                actin_mask.astype(np.int8), check_contrast=False)
        self.actin_mask = actin_mask


    def process_actin(self, force_compute=False):
        if not force_compute:
            try:
                data = np.load(self.tissue_fldr + 'improc_actin.npz')
                self.actin_angle = data['angles']
                self.actin_mask = data['mask']
                self.actin_dispersion = data['dispersion']
                self.actin_density = data['density']
                self.images['actin_density'] = self.actin_density
                self.images['actin_angle'] = self.actin_angle
                self.images['actin_dispersion'] = self.actin_dispersion
                self.images['actin_mask'] = self.actin_mask
                print('Processed actin loaded from file' + self.tissue_fldr + 'improc_actin.npz')
                return
            except:
                pass

        print('Processing actin')
        interpolate_mask = transform.rescale(self.interpolate_mask, 2) > 0
        actin_mask = transform.rescale(self.actin_mask, 2) > 0

        # Nearest neighbor interpolation
        print('Interpolating actin mask')
        angles = actproc.nearest_interpolation(interpolate_mask, actin_mask, self.actin_angle)
        actin_mask = actin_mask + interpolate_mask
        actin_mask[actin_mask>1] = 1

        # Mask results to tissue mask
        print('Masking actin results')
        angles, actin_mask = actproc.mask_actin_results(angles, actin_mask, self.tissue_mask)

        # Smooth myofibril results
        self.actin_angle = actproc.smooth_actin_angles(angles, actin_mask, window_size = 11)

        # Compute dispersion
        print('Computing actin dispersion')
        self.actin_dispersion = actproc.compute_dispersion(angles, actin_mask, window_size = 35)
        actin_mask = transform.rescale(actin_mask, 0.5) > 0

        # Density
        print('Computing actin density')
        self.actin_mask = actin_mask
        self.actin_density = filters.gaussian(actin_mask, sigma=2)

        self.images['actin_density'] = self.actin_density
        self.images['actin_angle'] = self.actin_angle
        self.images['actin_dispersion'] = self.actin_dispersion
        self.images['actin_mask'] = self.actin_mask

        np.savez(self.tissue_fldr + 'improc_actin', angles=self.actin_angle, mask=self.actin_mask,
                 dispersion=self.actin_dispersion, density=self.actin_density)


    def create_cell_mask(self):
        actin_mask = self.actin_mask
        cell_mask = morphology.remove_small_objects(actin_mask, min_size=1000)
        cell_mask = morphology.remove_small_holes(cell_mask, area_threshold=1000)
        cell_mask = morphology.binary_dilation(cell_mask, morphology.disk(10))
        cell_mask = filters.gaussian(cell_mask, sigma=5) > 0.5
        cell_density = filters.gaussian(cell_mask, sigma=2)

        self.cell_mask = cell_mask
        self.cell_density = cell_density

        self.images['cell_mask'] = cell_mask
        self.images['cell_density'] = cell_density

        io.imsave(self.tissue_fldr + 'cell_mask.tif', cell_mask.astype(np.int8), check_contrast=False)

        np.savez(self.tissue_fldr + 'improc_actin', angles=self.actin_angle, mask=self.actin_mask,
                 dispersion=self.actin_dispersion, density=self.actin_density,
                 cell_mask=self.cell_mask, cell_density=self.cell_density)


    """
    DSP PROCESSING
    """
    def process_dsp(self, method='window', mask_method=1, force_compute=False):
        if not force_compute:
            try:
                data = np.load(self.tissue_fldr + 'improc_dsp.npz')
                self.dsp_density = data['density']
                self.dsp_mask = data['mask']
                self.images['dsp_density'] = self.dsp_density
                self.images['dsp_mask'] = self.dsp_mask
                return
            except:
                pass

        print('Processing DSP')
        if mask_method == 1:
            self.dsp_mask = dspproc.get_dsp_mask(self.dsp_image, self.tissue_mask)
        elif mask_method == 2:
            self.dsp_mask = dspproc.get_dsp_mask(self.dsp_image, self.tissue_mask, self.fiber_mask)


        self.dsp_density = dspproc.process_dsp(self.dsp_mask, method=method)
        self.dsp_density[self.tissue_mask==0] = 0
        self.dsp_mask[self.tissue_mask==0] = 0

        self.images['dsp_density'] = self.dsp_density
        self.images['dsp_mask'] = self.dsp_mask

        np.savez(self.tissue_fldr + 'improc_dsp', mask=self.dsp_mask, density=self.dsp_density)


    """
    UTILS
    """
    def grab_all_data(self):
        self.tissue_mask = io.imread(self.tissue_fldr + '/tissue_mask.tif')
        self.images['tissue_mask'] = self.tissue_mask

        data = np.load(self.tissue_fldr + 'improc_actin.npz')
        self.actin_angle = data['angles']
        self.actin_mask = data['mask']
        self.actin_dispersion = data['dispersion']
        self.actin_density = data['density']
        self.cell_mask = data['cell_mask']
        self.cell_density = data['cell_density']
        self.images['actin_density'] = self.actin_density
        self.images['actin_angle'] = self.actin_angle
        self.images['actin_dispersion'] = self.actin_dispersion
        self.images['actin_mask'] = self.actin_mask
        self.images['cell_mask'] = self.cell_mask
        self.images['cell_density'] = self.cell_density

        data = np.load(self.tissue_fldr + 'improc_fiber.npz')
        self.fiber_angle = data['angles']
        self.fiber_mask = data['mask']
        self.fiber_dispersion = data['dispersion']
        self.fiber_density = data['density']
        self.images['fiber_density'] = self.fiber_density
        self.images['fiber_angle'] = self.fiber_angle
        self.images['fiber_dispersion'] = self.fiber_dispersion
        self.images['fiber_mask'] = self.fiber_mask

        data = np.load(self.tissue_fldr + 'improc_dsp.npz')
        self.images['dsp_density'] = data['density']
        self.images['dsp_mask'] = data['mask']



    """
    PLOTS
    """

    def plot_images(self, folder):
        if self.fiber_image is not None:
            plt.figure(1, clear=True)
            plt.imshow(self.fiber_image, cmap='gray')
            plt.axis('off')
            plt.savefig(folder + 'fiber.png', bbox_inches='tight')

        if self.actin_image is not None:
            plt.figure(1, clear=True)
            plt.imshow(self.actin_image, cmap='gray')
            plt.axis('off')
            plt.savefig(folder + 'actin.png', bbox_inches='tight')

        if self.dsp_image is not None:
            plt.figure(1, clear=True)
            plt.imshow(self.dsp_image, cmap='gray')
            plt.axis('off')
            plt.savefig(folder + 'dsp.png', bbox_inches='tight')


    def plot_tissue_mask(self, folder):
        plt.figure(1, clear=True)
        if self.fiber_image is not None:
            plt.imshow(self.fiber_image, cmap='gray')
        else:
            plt.imshow(self.actin_image, cmap='gray')
        plt.imshow(self.tissue_mask, alpha=0.5, cmap='RdBu')
        plt.axis('off')
        plt.savefig(folder + 'tissue_mask.png', bbox_inches='tight')


    def plot_fiber_processing(self, folder):
        plt.figure(2, clear=True)
        plt.imshow(self.fiber_image, cmap='gray')
        plt.imshow(self.fiber_mask, alpha=0.5, cmap='RdBu')
        plt.axis('off')
        plt.savefig(folder + 'fiber_density.png', bbox_inches='tight', dpi=180)

        plt.figure(3, clear=True)
        plt.imshow(self.fiber_density, cmap='viridis')
        plt.axis('off')
        plt.savefig(folder + 'fiber_density.png', bbox_inches='tight', dpi=180)

        plt.figure(4, clear=True)
        plt.imshow(self.fiber_angle, cmap='RdBu', vmin=-np.pi/4, vmax=np.pi/4)
        plt.axis('off')
        plt.savefig(folder + 'fiber_angle.png', bbox_inches='tight', dpi=180)

        plt.figure(5, clear=True)
        plt.imshow(self.fiber_dispersion, alpha=1, cmap='magma', vmin=0, vmax=0.5)
        plt.axis('off')
        plt.savefig(folder + 'fiber_dispersion.png', bbox_inches='tight', dpi=180)


    def plot_actin_processing(self, folder):
        plt.figure(1, clear=True)
        plt.imshow(self.actin_image, cmap='gray')
        plt.imshow(self.blobs_mask, alpha=0.5, cmap='RdBu')
        plt.axis('off')
        plt.savefig(folder + 'blobs_mask.png', bbox_inches='tight', dpi=180)

        plt.figure(2, clear=True)
        plt.imshow(self.actin_image, cmap='viridis')
        plt.imshow(self.actin_mask, alpha=0.5, cmap='RdBu')
        plt.axis('off')
        plt.savefig(folder + 'actin_mask.png', bbox_inches='tight', dpi=180)

        plt.figure(3, clear=True)
        plt.imshow(self.actin_density, cmap='viridis', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(folder + 'actin_density.png', bbox_inches='tight', dpi=180)

        plt.figure(4, clear=True)
        plt.imshow(self.actin_angle, cmap='RdBu', vmin=-np.pi/4, vmax=np.pi/4)
        plt.axis('off')
        plt.savefig(folder + 'actin_angle.png', bbox_inches='tight', dpi=180)

        plt.figure(5, clear=True)
        plt.imshow(self.actin_dispersion, alpha=1, cmap='magma', vmin=0, vmax=0.5)
        plt.axis('off')
        plt.savefig(folder + 'actin_dispersion.png', bbox_inches='tight', dpi=180)


    def plot_dsp_processing(self, folder, zoom=[]):
        if len(zoom) > 0:
            dsp_image = self.dsp_image[zoom[0]:zoom[1], zoom[2]:zoom[3]]
            dsp_mask = self.dsp_mask[zoom[0]:zoom[1], zoom[2]:zoom[3]]
        else:
            dsp_image = self.dsp_image
            dsp_mask = self.dsp_mask

        dsp_mask = dsp_mask.astype(float)
        dsp_mask[dsp_mask==0] = np.nan

        plt.figure(1, clear=True)
        plt.imshow(dsp_image, cmap='gray')
        plt.imshow(dsp_mask, alpha=0.5, cmap='viridis', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(folder + 'dsp_mask.png', bbox_inches='tight', dpi=180)

        plt.figure(2, clear=True)
        plt.imshow(dsp_image, cmap='gray')
        plt.imshow(self.dsp_density, alpha=0.5, cmap='viridis')
        plt.axis('off')
        plt.savefig(folder + 'dsp_density.png', bbox_inches='tight', dpi=180)


    def plot_dsp_processing_zoom(self, folder, zoom):
        dsp_image = self.dsp_image[zoom[0]:zoom[1], zoom[2]:zoom[3]]
        dsp_mask = self.dsp_mask[zoom[0]:zoom[1], zoom[2]:zoom[3]]

        dsp_mask = dsp_mask.astype(float)
        dsp_mask[dsp_mask==0] = np.nan

        fig, axs = plt.subplots(1, 2, figsize=(10, 5), clear=True, num=3)
        axs[0].imshow(dsp_image, cmap='gray')
        axs[0].axis('off')
        axs[0].set_title('DSP Image')
        axs[1].imshow(dsp_image, cmap='gray')
        axs[1].imshow(dsp_mask, cmap='viridis', vmin=0, vmax=1)
        axs[1].axis('off')
        axs[1].set_title('DSP Mask')
        plt.savefig(folder + 'dsp_mask_zoom.png', bbox_inches='tight', dpi=180)




def find_images(tissue_fldr, dataset=2, load=False):
    images = {}
    tif_files = glob(tissue_fldr + '*.tif')
    for tif_file in tif_files:
        if dataset == 1:
            if ('day7' in tissue_fldr) or ('pre' in tissue_fldr):
                if 'c1+2+3' in tif_file:
                    continue
                elif 'c1' in tif_file:
                    images['dsp'] = os.path.basename(tif_file)
                elif 'c2' in tif_file:
                    images['actin'] = os.path.basename(tif_file)
            elif ('day9' in tissue_fldr) or ('post' in tissue_fldr):
                if 'c1+2+3' in tif_file:
                    continue
                elif 'c2' in tif_file:
                    images['dsp'] = os.path.basename(tif_file)
                elif 'c3' in tif_file:
                    images['fibers'] = os.path.basename(tif_file)
                elif 'c4' in tif_file:
                    images['actin'] = os.path.basename(tif_file)
        elif dataset == 2:
            if ('day7' in tissue_fldr) or ('pre' in tissue_fldr):
                if 'c1+2+3' in tif_file:
                    continue
                elif 'c1' in tif_file:
                    images['dsp'] = os.path.basename(tif_file)
                elif 'c2' in tif_file:
                    images['fibers'] = os.path.basename(tif_file)
                elif 'c3' in tif_file:
                    images['actin'] = os.path.basename(tif_file)
            elif ('day9' in tissue_fldr) or ('post' in tissue_fldr):
                if 'c1+2+3' in tif_file:
                    continue
                elif 'c2' in tif_file:
                    images['dsp'] = os.path.basename(tif_file)
                elif 'c3' in tif_file:
                    images['fibers'] = os.path.basename(tif_file)
                elif 'c4' in tif_file:
                    images['actin'] = os.path.basename(tif_file)

    if load:
        for key in images.keys():
            images[key] = io.imread(tissue_fldr + images[key])

    return images