#!/usr/bin/env python
# coding: utf-8

# ## 4. Image Registration with initial transform and/or multiple threads

# In this notebook 2 other options of the elastix algorithm are shown: initial transformation and multithreading.
# They're shown together just to reduce the number of example notebooks and
# thus can be used independently as well as in combination with whichever other functionality
# of the elastix algorithm.
#
# Initial transforms are transformations that are done on the moving image before the registration is started.
#
# Multithreading spreaks for itself and can be used in similar fashion in the transformix algorithm.
#
#

# ### Registration

import itk
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.exposure import rescale_intensity
from mpl_toolkits.axes_grid1 import make_axes_locatable

def imageToArray(imageFileName, mask=None):
    """
    image_to_array transform png images to arrays to be used in elastix transformations

    :param imageFileName1: original image of the image before contraction/movement
    :param mask1: masks for image1 and image2
    :return: the array versions of image1 and image 2 with the masks applied as a list of two values
             list[0] = imageArray1 and list[1] = imageArray2
    """
    image = imread(imageFileName, as_gray=True)

    #transforms images into arrays
    image_array = np.asarray(image).astype(np.float32)

    #converts masks into arrays
    if mask is not None:
        mask = np.asarray(imread(mask, as_gray=True))
        image_array[mask==0] = 0

    return image_array

def elastix_simple_transformation(originalArray, movingArray, mode):
    """
    elastix_transformation utalizes the elastix functionality to attempt to transform
    moving images back to it's original image

    :param originalArray: original array of the image before contraction/movement
    :param movingArray: array of image during contraction/movement
    :param mode: str, 'translation', 'rigid', 'affine'
    :return: the transform parameters of the image that returns the image to it's original state
    """
    fixed_array = originalArray.astype(np.float32)
    moving_array = movingArray.astype(np.float32)

    fixed_image = itk.GetImageFromArray(fixed_array)
    moving_image = itk.GetImageFromArray(moving_array)

    # Import Default Parameter Map
    parameter_object = itk.ParameterObject.New()

    # Import Default Parameter Map
    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap(mode)
    default_rigid_parameter_map['FinalBSplineInterpolationOrder'] = ['0']
    default_rigid_parameter_map['NumberOfResolutions'] = ['6']
    parameter_object.AddParameterMap(default_rigid_parameter_map)

    # Load Elastix Image Filter Object
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetParameterObject(parameter_object)

    # Set additional options
    elastix_object.SetLogToConsole(False)

    # Update filter object (required)
    elastix_object.UpdateLargestPossibleRegion()

    return elastix_object.GetTransformParameterObject()


def apply_transform(movingArray, resultParameters):
    movingArray = movingArray.astype(np.float32)
    movingImage = itk.GetImageFromArray(movingArray)
    defArray = itk.transformix_filter(movingImage,
            transform_parameter_object = resultParameters,
            log_to_console=False)
    defArray = itk.GetArrayFromImage(defArray).astype(float)

    # Rescale intensity to 0-1 range
    defArray = rescale_intensity(defArray)
    return defArray


def rescale_image_intensity(defArray):
    # Just a dummy way of doing this so I don't have to import the package in the main file
    defArray = rescale_intensity(defArray)
    return defArray

def display_save_Image(image, vmin1, vmax1, save):
    """
    display_save_Image displays whatever image you want

    :param image: the image itself
    :param vmin1, vmax1: min and max values of the image to show
    :param save: true or false to save the image
    """

    plt.figure()
    plt.imshow(image, vmin = vmin1, vmax = vmax1)
    plt.axis('off')
    plt.title(image, 'results')
    if save:
        plt.savefig('output/', image, '.png')

def displacement_field_elastix(originalArray, movingArray, saveImage, outfldr, pix_vid, maskFileName = None):
    """
    displacement_field_elastix creates a displacement field of the two images brought in

    :param originalArray: original array of the image before contraction/movement
    :param movingArray: array of image during contraction/movement
    :param parameterFileName: name of the file with the ideal parameters for elastix transformation
    :return: the displacement field array
    """
    resultParameters = elastix_transformation(originalArray, movingArray, True, outfldr)
    movingImage = itk.GetImageFromArray(movingArray)
    deformation_field = itk.transformix_deformation_field(movingImage, resultParameters)
    defArray = itk.GetArrayFromImage(deformation_field).astype(float)*pix_vid

    if maskFileName is not None:
        mask_1 = np.asarray(imread(maskFileName, as_gray=True))
        mask_1 = mask_1[9:-10]
        Mask_1 = np.zeros((244, 420))
        Mask_1[0:mask_1.shape[0],11:mask_1.shape[1]+11] = mask_1
        Mask_1 = np.flipud(Mask_1)
        defArray[Mask_1==0] = np.nan
    return defArray




def display_save_displacement(defArray, tri_quad, disp, name, save):
    #Plot images
    fig, axs = plt.subplots(2, 2, sharey = "row", figsize=[30,30])

    axs[0, 1].invert_yaxis()
    axs[0, 0].invert_yaxis()

    v_min1 = -16
    v_max1 = 12

    v_min2 = -16
    v_max2 = 16

    levels_x = np.linspace(v_min2, v_max2, 256)
    levels_y = np.linspace(v_min1, v_max1, 256)


    dispx = defArray[:,:,0]
    # dispx = np.flipud(dispx)
    im3 = axs[0, 1].imshow(dispx, vmin = v_min2, vmax = v_max2)
    axs[0, 1].invert_yaxis()
    divider1 = make_axes_locatable(axs[0,1])
    cax = divider1.new_vertical(size='5%', pad=0.6, pack_start = True)
    fig.add_axes(cax)
    cbar = fig.colorbar(im3, cax = cax, orientation = 'horizontal')
    cbar.set_label('displacement (pixels)', fontsize = 25)
    cbar.ax.tick_params(labelsize=20)

    dispy = defArray[:,:,1]
    # dispy = np.flipud(dispy)

    im2 = axs[0, 0].imshow(dispy,vmin = v_min1, vmax = v_max1)
    axs[0, 1].invert_yaxis()

    divider = make_axes_locatable(axs[0,0])
    cax = divider.new_vertical(size='5%', pad=0.6, pack_start = True)
    fig.add_axes(cax)
    cbar2 = fig.colorbar(im2, cax = cax, orientation = 'horizontal')
    cbar2.set_label('displacement (pixels)', fontsize = 25)
    cbar2.ax.tick_params(labelsize=20)

    axs[0,0].axis('off')
    axs[0,1].axis('off')
    axs[0,0].set_title('Elastix Displacement Y', fontsize=30)
    axs[0,1].set_title('Elastix Displacement X', fontsize=30)

    im4 = axs[1, 1].tricontourf(tri_quad, disp[:,0] , levels=levels_x )

    axs[1,1].set_aspect("equal")
    divider1 = make_axes_locatable(axs[1,1])
    cax = divider1.new_vertical(size='5%', pad=0.6, pack_start = True)
    fig.add_axes(cax)
    cbar3 = fig.colorbar(im4, cax = cax, orientation = 'horizontal')
    cbar3.set_label('displacement (pixels)', fontsize = 25)
    cbar3.ax.tick_params(labelsize=20)

    im5 = axs[1, 0].tricontourf(tri_quad, disp[:,1] , levels=levels_y)
    divider = make_axes_locatable(axs[1,0])
    axs[1,0].set_aspect("equal")
    cax = divider.new_vertical(size='5%', pad=0.6, pack_start = True)
    fig.add_axes(cax)
    cbar4 = fig.colorbar(im5, cax = cax, orientation = 'horizontal')
    cbar4.set_label('displacement (pixels)', fontsize = 25)
    cbar4.ax.tick_params(labelsize=20)

    axs[1,0].axis('off')
    axs[1,1].axis('off')
    axs[1,0].set_title('Analytical Displacement Y', fontsize=30)
    axs[1,1].set_title('Analytical Displacement X', fontsize=30)


    if save:
        plt.savefig(name + '.png', dpi = 200)


    # np.save('comparisionPlots/displacement_x.npy', defArray[:,:,0])
    # np.save('comparisionPlots/displacement_y.npy', defArray[:,:,1]*-1)

    #as a note next time save the above file with the colorbar being the same for both

