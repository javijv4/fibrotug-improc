#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/27 13:24:49

@author: Javiera Jilberto Vallejos 
'''

from matplotlib import pyplot as plt
import numpy as np
from skimage import io, filters, measure, draw, morphology, transform, segmentation, exposure
import cv2
from scipy.optimize import minimize
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, windows
from matplotlib.widgets import PolygonSelector, RectangleSelector
from matplotlib.widgets import Slider

def estimate_tissue_rectangle(img, plot=False):

    portion = 1
    mult = 1

    # Threshold image using otsu
    while portion > 0.1:
        thresh = filters.threshold_otsu(img)*mult
        binary = img > thresh
        binary = 1-binary
        binary = morphology.binary_opening(binary, morphology.disk(1))

        # Label connected regions of the binary image
        label_image = measure.label(binary)

        # Find the two largest connected components
        regions = measure.regionprops(label_image)
        regions = sorted(regions, key=lambda r: r.area, reverse=True)[:2]

        # Create a mask for the two largest connected components
        largest_objects_mask = np.zeros_like(label_image, dtype=bool)
        for region in regions:
            largest_objects_mask |= label_image == region.label
        binary = largest_objects_mask
        portion = np.sum(binary)/binary.size
        mult -= 0.1
        if mult < 0:
            return #ValueError('Could not find a good threshold')



    # Find contours
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [np.vstack(contours).squeeze()]

    # Get the minimum area rectangle for each contour
    min_area_rects = [cv2.minAreaRect(contour) for contour in contours]

    # Draw the rectangles on the original image
    for rect in min_area_rects:
        box = cv2.boxPoints(rect)
        box = np.intp(box)

    # I need to make sure the box is oriented in a consistent way
    length = np.linalg.norm(box[3] - box[0])
    width = np.linalg.norm(box[1] - box[0])
    if length < width:
        length = np.linalg.norm(box[1] - box[0])
        width = np.linalg.norm(box[2] - box[1])
        aux_box = np.array([box[0], box[3], box[2], box[1]])
        box = aux_box


    if plot:
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.plot(box[:, 0], box[:, 1], 'r-')
        for contour in contours:
            contour = contour.squeeze()
            plt.plot(contour[:, 0], contour[:, 1], 'b-')
        plt.axis('off')

    # Sanity checks
    aspect_ratio = length / width
    if aspect_ratio < 1.5:
        return
    if length < 500 or length > 700:
        return


    return box


def get_box_grid(box, rotation=0, center=(0, 0), img=None):
    length = np.linalg.norm(box[3] - box[0])
    width = np.linalg.norm(box[1] - box[0])
    if length < width:
        length = np.linalg.norm(box[1] - box[0])
        width = np.linalg.norm(box[2] - box[1])

    length = float(length*1.1)
    width = float(width*1.1)

    i = np.arange(0, length, 1)
    j = np.arange(0, width, 1)
    ii, jj = np.meshgrid(i, j)

    # Rotate the grid
    if rotation != 0:
        R = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        rotated_grid = np.dot(R, np.array([ii.flatten(), jj.flatten()]))
        ii = rotated_grid[0].reshape(ii.shape)
        jj = rotated_grid[1].reshape(jj.shape)

    # Center the grid at the centroid of the mask
    if np.linalg.norm(center) != 0:
        ii -= np.mean(ii)- center[0]
        jj -= np.mean(jj) - center[1]


    # Making sure the grid is within the image
    if img is not None:
        mask = np.zeros(ii.shape, dtype=bool)
        mask[ii < 0] = 1
        mask[jj < 0] = 1
        mask[ii >= img.shape[0]] = 1
        mask[jj >= img.shape[1]] = 1
        mask = morphology.binary_dilation(mask, morphology.disk(2))
        ii[ii < 0] = 0
        jj[jj < 0] = 0
        ii[ii >= img.shape[0]] = img.shape[0]-1
        jj[jj >= img.shape[1]] = img.shape[1]-1

        ii = ii.astype(int)
        jj = jj.astype(int)
        return ii, jj, mask

    return ii, jj


def rotate_and_evaluate(box, zero_frame, centroid, angle):
    if not isinstance(angle, float):
        angle = angle[0]

    ii, jj = get_box_grid(box, angle, centroid)

    # Evaluate frame_zero at the grid points
    zero_frame = (zero_frame - np.min(zero_frame)) / (np.max(zero_frame) - np.min(zero_frame))

    values = evaluate_image_in_grid(ii, jj, zero_frame)

    return ii, jj, values, np.sum(values, axis=0)


def evaluate_image_in_grid(ii, jj, img):
    # Making sure the grid is within the image
    ii[ii < 0] = 0
    jj[jj < 0] = 0
    ii[ii >= img.shape[0]] = img.shape[0]-1
    jj[jj >= img.shape[1]] = img.shape[1]-1

    ii = ii.astype(int)
    jj = jj.astype(int)

    values = img[jj,ii]

    return values


def find_tissue_rotation(box, zero_frame):
    # Calculate the centroid of the mask
    centroid = np.mean(box, axis=0)

    # Find initial rotation
    vector = box[3] - box[0]
    initial_angle = np.arctan2(vector[1], vector[0])

    # Define the objective function for minimization
    def objective_function(angle):
        ii, _, _, sum_values = rotate_and_evaluate(box, zero_frame, centroid, angle)
        min_val_left = np.min(sum_values[:ii.shape[1] // 2])
        min_val_right = np.min(sum_values[ii.shape[1] // 2:])
        mean_min_val = (min_val_left + min_val_right) / 2
        return mean_min_val

    # Use a minimization algorithm to find the best angle
    result = minimize(objective_function, initial_angle, method='Nelder-Mead', bounds=[(initial_angle - np.pi / 4, initial_angle + np.pi / 4)])
    best_angle = result.x[0]

    return best_angle

def interactive_tissue_rotation(zero_frame, initial_angle):
    zero_frame = exposure.equalize_hist(zero_frame)
    fig, ax = plt.subplots()
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
    
    ax.vlines(np.arange(0, zero_frame.shape[1], 100), 0, zero_frame.shape[0], color='r', lw=1)
    ax.hlines(np.arange(0, zero_frame.shape[0], 100), 0, zero_frame.shape[1], color='r', lw=1)

    val = np.mean(zero_frame)
    rotated_img = transform.rotate(zero_frame, np.rad2deg(initial_angle), cval=val)
    img_display = ax.imshow(rotated_img, cmap='gray')
    ax.set_title('Adjust the rotation angle')
    ax.axis('off')

    slider = Slider(ax_slider, 'Angle', initial_angle - np.pi / 4, initial_angle + np.pi / 4, valinit=initial_angle)

    def update(val):
        angle = slider.val
        rotated_img = transform.rotate(zero_frame, np.rad2deg(angle), cval=val)
        img_display.set_data(rotated_img)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    best_angle = slider.val

    return best_angle



def get_mask(diff, xlim_l, xlim_r, rescale=3):
    frangi = filters.frangi(diff, sigmas=range(rescale,rescale*3,rescale*2))
    x = np.linspace(0,1,frangi.shape[1])
    frangi[:,(x>xlim_l[0])*(x<xlim_l[1])] = 0
    frangi[:,(x>xlim_r[0])*(x<xlim_r[1])] = 0
    mask = frangi > filters.threshold_triangle(frangi)

    # Get the two largest objects
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    areas = [prop.area for prop in props]
    areas = np.array(areas)
    largest = np.argmax(areas)
    areas[largest] = 0
    second_largest = np.argmax(areas)

    mask_1 = labels == largest + 1
    mask_2 = labels == second_largest + 1
    mask = mask_1 + mask_2

    # Check that both objects have non-zero values at the beggining and at the end
    if not np.sum(mask_1[0]) or not np.sum(mask_2[0]):
        raise ValueError('The two largest objects do not have non-zero values at the beggining')
    if not np.sum(mask_1[-1]) or not np.sum(mask_2[-1]):
        raise ValueError('The two largest objects do not have non-zero values at the end')
    
    return mask, mask_1, mask_2


def correct_by_rest_position(points, left_or_right, bins=128):
    maxv = np.max(points)
    minv = np.min(points)
    mag = maxv - minv
    if left_or_right == 'left':
        maxv = maxv - mag/2
    else:
        minv = minv + mag/2
    counts, pos = np.histogram(points, bins=bins, range=(minv, maxv))
    rest_val = pos[np.argmax(counts)]
    close_val = points[np.argmin(np.abs(points - rest_val))]

    if left_or_right == 'left':
        points[points < close_val] = close_val
    else:
        points[points > close_val] = close_val

    return points


def get_post_boundaries(vals, xlim_l, xlim_r, rescale=4, which='left'):
    nframes = vals.shape[0]
    vals = transform.rescale(vals, rescale, order=1, mode='reflect', anti_aliasing=False)
    try: 
        diff = np.gradient(vals, axis=1)
        mask, mask_1, mask_2 = get_mask(diff, xlim_l, xlim_r, rescale=rescale)
    except ValueError as e:
        diff = np.gradient(vals[:,::-1], axis=1)[:,::-1]
        mask, mask_1, mask_2 = get_mask(diff, xlim_l, xlim_r, rescale=rescale)
    
        
    mask_1 = filters.gaussian(mask_1, sigma=1)
    mask_2 = filters.gaussian(mask_2, sigma=1)

    i = np.arange(0, mask.shape[0], 1, dtype=float)
    j = np.arange(0, mask.shape[1], 1, dtype=float)
    _, ii = np.meshgrid(i, j)
    ii = ii.T

    point_mask_1 = np.sum(mask_1*ii, axis=1)/np.sum(mask_1, axis=1)
    point_mask_2 = np.sum(mask_2*ii, axis=1)/np.sum(mask_2, axis=1)

    frame = np.arange(0, nframes, 1/rescale)
    f1 = interp1d(frame, point_mask_1)
    f2 = interp1d(frame, point_mask_2)

    point_mask_1 = f1(np.arange(0, nframes, 1)) / rescale
    point_mask_2 = f2(np.arange(0, nframes, 1)) / rescale
    
    point_mask_1 = correct_by_rest_position(point_mask_1, which, bins=256)
    point_mask_2 = correct_by_rest_position(point_mask_2, which, bins=256)

    return point_mask_1, point_mask_2


def get_displacements(all_frame_vals):
    xlim_l, xlim_r = select_post_area(all_frame_vals)

    point_1_left, point_2_left = get_post_boundaries(all_frame_vals[:, :all_frame_vals.shape[1] // 3], xlim_l, xlim_r)
    point_1_right, point_2_right = get_post_boundaries(all_frame_vals[:, all_frame_vals.shape[1] // 3 * 2:], xlim_l, xlim_r, which='right')
    point_1_right += all_frame_vals.shape[1] // 3 * 2
    point_2_right += all_frame_vals.shape[1] // 3 * 2 

    return (point_1_left, point_2_left, point_1_right, point_2_right)


def get_displacements_3(all_frame_vals, zero_frame, rescale=4):
    xlim_l, xlim_r = select_post_area(zero_frame, all_frame_vals)
    # xlim_l = (0.14674735249621784, 0.264750378214826) 
    # xlim_r = (0.8169440242057489, 0.9576399394856279)

    # Apply a low-pass filter to all_frame_vals    
    arr = filters.sobel(filters.unsharp_mask(all_frame_vals, radius=1, amount=5), axis=1)
    # arr = np.gradient(all_frame_vals, axis=1)
    arr = np.abs(arr)
    half = arr.shape[1]//2*rescale

    arr = transform.rescale(arr, rescale, order=3, mode='reflect', anti_aliasing=False)

    frame_peaks = np.zeros([arr.shape[0], 4])
    x = np.linspace(0, 1, arr.shape[1])
    weights = np.zeros(len(x))
    weights[(x>xlim_l[0])*(x<xlim_l[1])] = 1
    weights[(x>xlim_r[0])*(x<xlim_r[1])] = 1
    weights = gaussian_filter1d(weights, 1*rescale)
    # weights = ((np.tanh((x-0.8)*40)+1)/2)*((np.tanh((-x+0.95)*40)+1)/2) + ((np.tanh((-x+0.2)*40)+1)/2)*(np.tanh((x-0.05)*40)+1)/2



    disp_array = np.repeat(np.arange(-50, 51, 1), 4).reshape(-1, 4)
    disp_array[:,-2:] = -disp_array[:,-2:]
    disp_array = np.arange(-50, 51, 1)

    gaussian = windows.gaussian(disp_array.shape[0], 3*rescale)

    def get_new_peaks(disp_l, disp_r, peaks):
        new_peaks = np.zeros_like(peaks)
        new_peaks[:2] = peaks[:2] + disp_l
        new_peaks[2:] = peaks[2:] + disp_r
        return new_peaks
    
    disps = []
    for i in range(arr.shape[0]):
        aux = arr[i]*weights
        if i == 0:
            peaks = find_peaks(aux, distance=30*rescale)[0]
            peaks_left = peaks[peaks < half]
            peaks_right = peaks[peaks > half]

            order = np.argsort(aux[peaks_left])
            peaks_left = peaks_left[order[-2:]]
            peaks_left = np.sort(peaks_left)

            order = np.argsort(aux[peaks_right])
            peaks_right = peaks_right[order[-2:]]
            peaks_right = np.sort(peaks_right)

            peaks = np.append(peaks_left, peaks_right)

        else:
            aux_func = interp1d(np.arange(0, len(aux), 1), aux, fill_value=0, bounds_error=False)
            vals_l = np.sum(aux_func(peaks[None, :2] + disp_array[:,None]), axis=1)
            vals_r = np.sum(aux_func(peaks[None, 2:] + disp_array[:,None]), axis=1)
            disp_l = disp_array[np.argmax(vals_l)]
            disp_r = disp_array[np.argmax(vals_r)]
            disps.append((disp_l, disp_r))
            peaks = get_new_peaks(disp_l, disp_r, peaks)

        frame_peaks[i] = peaks


    # Define the Butterworth filter
    displacements = get_displacements_from_traces(frame_peaks)
    b, a = butter(N=4, Wn=0.08, btype='low', analog=False)

    smooth_curves = np.zeros_like(frame_peaks)
    for i in range(4):
        curve = frame_peaks[:, i]
        side = 'left' if i < 2 else 'right'

        # Apply the filter to the curve
        smooth_curve = filtfilt(b, a, curve)
        smooth_curve = correct_by_rest_position(smooth_curve, side, bins=256)

        smooth_curves[:, i] = smooth_curve

    smooth_curves = smooth_curves / rescale

    # Resample the curves
    f = interp1d(np.linspace(0, 1, smooth_curves.shape[0]), smooth_curves, axis=0)
    smooth_curves = f(np.linspace(0, 1, all_frame_vals.shape[0]))

    return smooth_curves[:,0], smooth_curves[:,1], smooth_curves[:,2], smooth_curves[:,3]




def get_displacements_4(all_frame_vals, xlim_l, xlim_r, rescale=4):
    print(xlim_l, xlim_r)
    # Apply a low-pass filter to all_frame_vals    
    arr = transform.rescale(all_frame_vals, rescale, order=3, mode='reflect', anti_aliasing=False)
    arr = filters.sobel(filters.unsharp_mask(arr, radius=2, amount=20), axis=1)
    arr = filters.gaussian(arr, sigma=2)
    # arr = np.gradient(all_frame_vals, axis=1)
    arr = np.abs(arr)
    half = arr.shape[1]//2

    # arr = transform.rescale(arr, rescale, order=3, mode='reflect', anti_aliasing=False)

    g = windows.gaussian(1001,5*rescale)
    gfunc = interp1d(np.arange(0, len(g), 1)-len(g)//2, g, fill_value=0, bounds_error=False)

    frame_peaks = np.zeros([arr.shape[0], 2])
    x = np.linspace(0, 1, arr.shape[1])
    weights = np.zeros(len(x))
    weights[(x>xlim_l[0])*(x<xlim_l[1])] = 1
    weights[(x>xlim_r[0])*(x<xlim_r[1])] = 1
    weights = gaussian_filter1d(weights, 1*rescale)
    
    x = np.arange(0, arr.shape[1], 1)

    for i in range(arr.shape[0]):
        aux = arr[i]*weights
        
        peaks = find_peaks(aux, distance=300*rescale)[0]
        peaks_left = peaks[peaks < half]
        peaks_left_value = aux[peaks_left]
        peaks_right = peaks[peaks > half]
        peaks_right_value = aux[peaks_right]

        peaks_left = peaks_left[np.argmax(peaks_left_value)]
        peaks_right = peaks_right[np.argmax(peaks_right_value)]
        peaks = np.array([peaks_left, peaks_right])
        frame_peaks[i] = peaks
        weights = gfunc(x-peaks[0]) + gfunc(x-peaks[1])
        # plt.plot(aux)
        # plt.plot(peaks, aux[peaks], 'ro')

    # Define the Butterworth filter
    displacements = get_displacements_from_traces(frame_peaks.T)
    if np.max(displacements) < 2*rescale:
        Wn = 0.03
    else:
        Wn = 0.08

    b, a = butter(N=4, Wn=Wn, btype='low', analog=False)

    smooth_curves = np.zeros_like(frame_peaks)
    for i in range(2):
        curve = frame_peaks[:, i]
        side = 'left' if i == 0 else 'right'

        # Apply the filter to the curve
        smooth_curve = filtfilt(b, a, curve)
        smooth_curve = correct_by_rest_position(smooth_curve, side, bins=256)

        smooth_curves[:, i] = smooth_curve

    smooth_curves = smooth_curves / rescale

    # Resample the curves
    f = interp1d(np.linspace(0, 1, smooth_curves.shape[0]), smooth_curves, axis=0)
    smooth_curves = f(np.linspace(0, 1, all_frame_vals.shape[0]))
    # smooth_curves = frame_peaks / rescale

    return smooth_curves[:,0], smooth_curves[:,1]


def find_traces(arr, rescale, half, xlim_l, xlim_r, reverse=False):
    # Generate a gaussian function for wieghting the peaks
    g = windows.gaussian(1001, 3*rescale)
    gfunc = interp1d(np.arange(0, len(g), 1)-len(g)//2, g, fill_value=0, bounds_error=False)

    frame_peaks = np.zeros([arr.shape[0], 4])
    x = np.linspace(0, 1, arr.shape[1])
    weights = np.zeros(len(x))
    weights[(x>xlim_l[0])*(x<xlim_l[1])] = 1
    weights[(x>xlim_r[0])*(x<xlim_r[1])] = 1
    weights = gaussian_filter1d(weights, 1*rescale)
    weights0 = weights.copy()
    x = np.arange(0, arr.shape[1], 1)

    if reverse:
        arr = arr[::-1]

    for i in range(arr.shape[0]):
        aux = arr[i]*weights
        peaks = find_peaks(aux, distance=30*rescale)[0]
        peaks_left = peaks[peaks < half]
        peaks_right = peaks[peaks > half]

        order = np.argsort(aux[peaks_left])
        peaks_left = peaks_left[order[-2:]]
        peaks_left = np.sort(peaks_left)

        order = np.argsort(aux[peaks_right])
        peaks_right = peaks_right[order[-2:]]
        peaks_right = np.sort(peaks_right)

        peaks = np.append(peaks_left, peaks_right)
        frame_peaks[i] = peaks
        weights = gfunc(x-peaks[0]) + gfunc(x-peaks[1]) + gfunc(x-peaks[2]) + gfunc(x-peaks[3])*weights0

    regularity = np.zeros(4)
    for i in range(4):
        individual_peaks = find_individual_peaks(frame_peaks[:, i])
        if len(individual_peaks) == 0:
            regularity[i] = np.nan
            continue
        regularity[i] = np.mean(np.std(np.array(individual_peaks), axis=0))

    if reverse:
        frame_peaks = frame_peaks[::-1]

    plt.figure(figsize=(10, 6))
    plt.imshow(arr, aspect='auto', cmap='viridis')
    plt.plot(frame_peaks[:, 0], np.arange(arr.shape[0]), color='r', label='Peak 1 Left')
    plt.plot(frame_peaks[:, 1], np.arange(arr.shape[0]), color='r', label='Peak 2 Left')
    plt.plot(frame_peaks[:, 2], np.arange(arr.shape[0]), color='b', label='Peak 1 Right')
    plt.plot(frame_peaks[:, 3], np.arange(arr.shape[0]), color='b', label='Peak 2 Right')
    plt.legend()
    plt.xlabel('Pixel Position')
    plt.ylabel('Frame')
    plt.title('Frame Peaks and Array')
    plt.show()

    return frame_peaks


def check_peak_traces(frame_peaks, frame_peaks_r):
    disp = frame_peaks - frame_peaks[0]
    disp[:,2:] = -disp[:,2:]
    disp_r = frame_peaks_r - frame_peaks_r[0]
    disp_r[:,2:] = -disp_r[:,2:]

    std_disp = np.std(disp, axis=1)
    std_disp_r = np.std(disp_r, axis=1)

    if np.mean(std_disp) < np.mean(std_disp_r):
        return frame_peaks
    else:
        return frame_peaks_r

def select_post_area(img, all_frame_vals):
    img = exposure.equalize_hist(img)
    img_aux = np.vstack([img, 1-all_frame_vals])
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_aux, cmap='gray')
    ax.axis('off')
    ax.set_title('Select left post area and press Enter')

    lines = []

    def onselect(eclick, erelease):
        nonlocal lines
        for line in lines:
            line.remove()
        lines = []
        line1 = ax.axvline(eclick.xdata, color='r')
        line2 = ax.axvline(erelease.xdata, color='r')
        lines.extend([line1, line2])
        fig.canvas.draw()

    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)

    rect_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    x_min1, x_max1 = int(rect_selector.extents[0]), int(rect_selector.extents[1])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_aux, cmap='gray')
    ax.axis('off')
    ax.set_title('Select right post area and press Enter')

    lines = []

    def onselect(eclick, erelease):
        nonlocal lines
        for line in lines:
            line.remove()
        lines = []
        line1 = ax.axvline(eclick.xdata, color='r')
        line2 = ax.axvline(erelease.xdata, color='r')
        lines.extend([line1, line2])
        fig.canvas.draw()

    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)

    rect_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    x_min2, x_max2 = int(rect_selector.extents[0]), int(rect_selector.extents[1])

    # Normalize the values
    xmax = len(all_frame_vals[1])
    x_min1 = x_min1/xmax
    x_max1 = x_max1/xmax
    x_min2 = x_min2/xmax
    x_max2 = x_max2/xmax

    return (x_min1, x_max1), (x_min2, x_max2)

def get_displacements_2(all_frame_vals, rescale=4):
    # Get area of interest
    xlim_l, xlim_r = select_post_area(all_frame_vals)
    # xlim_l = (0.15965166908563136, 0.3018867924528302) 
    # xlim_r = (0.7039187227866474, 0.8490566037735849)


    # Apply a low-pass filter to all_frame_vals    
    arr = filters.sobel(filters.unsharp_mask(all_frame_vals, radius=1, amount=5), axis=1)
    # arr = np.gradient(all_frame_vals, axis=1)
    arr = np.abs(arr)
    half = arr.shape[1]//2*rescale

    arr = transform.rescale(arr, rescale, order=3, mode='reflect', anti_aliasing=False)

    # Find traces
    frame_peaks = find_traces(arr, rescale, half, xlim_l, xlim_r)
    frame_peaks_r = find_traces(arr, rescale, half, xlim_l, xlim_r, reverse=True)
    frame_peaks = check_peak_traces(frame_peaks, frame_peaks_r)

    # Define the Butterworth filter
    b, a = butter(N=4, Wn=0.05, btype='low', analog=False)

    smooth_curves = np.zeros_like(frame_peaks)
    for i in range(4):
        curve = frame_peaks[:, i]
        side = 'left' if i < 2 else 'right'

        # Apply the filter to the curve
        smooth_curve = filtfilt(b, a, curve)
        smooth_curve = correct_by_rest_position(smooth_curve, side, bins=256)

        smooth_curves[:, i] = smooth_curve

    smooth_curves = smooth_curves / rescale

    # Resample the curves
    f = interp1d(np.linspace(0, 1, smooth_curves.shape[0]), smooth_curves, axis=0)
    smooth_curves = f(np.linspace(0, 1, all_frame_vals.shape[0]))

    return smooth_curves[:,0], smooth_curves[:,1], smooth_curves[:,2], smooth_curves[:,3]


def check_traces(traces):
    strikes = np.zeros(len(traces), dtype=int)

    point_1_left, point_2_left, point_1_right, point_2_right = traces

    point_1_disp_left = point_1_left - np.min(point_1_left)
    point_2_disp_left = point_2_left - np.min(point_2_left)
    point_1_disp_right = -(point_1_right - np.max(point_1_right))
    point_2_disp_right = -(point_2_right - np.max(point_2_right))

    # First check: post width should be somewhat constant
    width_left = np.abs(point_2_disp_left - point_1_disp_left)
    width_right = np.abs(point_2_disp_right - point_1_disp_right)

    if np.std(width_left) > 1:
        strikes[:2] += 1
    if np.std(width_right) > 1:
        strikes[2:] += 1


    # Second check: the variation of the traces should be small
    error_1l_2l = np.linalg.norm(point_1_disp_left - point_2_disp_left)
    error_1l_1r = np.linalg.norm(point_1_disp_left - point_1_disp_right)
    error_1l_2r = np.linalg.norm(point_1_disp_left - point_2_disp_right)
    error_2l_1r = np.linalg.norm(point_2_disp_left - point_1_disp_right)
    error_2l_2r = np.linalg.norm(point_2_disp_left - point_2_disp_right)
    error_1r_2r = np.linalg.norm(point_1_disp_right - point_2_disp_right)

    # Error leaving 1l out
    error_1l_out = np.mean([error_2l_1r, error_2l_2r, error_1r_2r])
    error_2l_out = np.mean([error_1l_1r, error_1l_2r, error_1r_2r])
    error_1r_out = np.mean([error_1l_2l, error_2l_2r, error_1l_2r])
    error_2r_out = np.mean([error_1l_2l, error_2l_1r, error_1l_1r])

    mean_1l_out = np.mean([error_2l_out, error_1r_out, error_2r_out])
    std_1l_out = np.std([error_2l_out, error_1r_out, error_2r_out])
    mean_2l_out = np.mean([error_1l_out, error_1r_out, error_2r_out])
    std_2l_out = np.std([error_1l_out, error_1r_out, error_2r_out])
    mean_1r_out = np.mean([error_1l_out, error_2l_out, error_2r_out])
    std_1r_out = np.std([error_1l_out, error_2l_out, error_2r_out])
    mean_2r_out = np.mean([error_1l_out, error_2l_out, error_1r_out])
    std_2r_out = np.std([error_1l_out, error_2l_out, error_1r_out])


    if error_1l_out < (mean_1l_out - 3*std_1l_out):
        strikes[0] += 1
    if error_2l_out < (mean_2l_out - 3*std_2l_out):
        strikes[1] += 1
    if error_1r_out < (mean_1r_out - 3*std_1r_out):
        strikes[2] += 1
    if error_2r_out < (mean_2r_out - 3*std_2r_out):
        strikes[3] += 1

    if strikes[0] == 2:
        point_1_left[:] = np.nan
    if strikes[1] == 2:
        point_2_left[:] = np.nan
    if strikes[2] == 2:
        point_1_right[:] = np.nan
    if strikes[3] == 2:
        point_2_right[:] = np.nan

    return point_1_left, point_2_left, point_1_right, point_2_right


def check_traces_2(traces):
    ntraces = len(traces)
    displacements = get_displacements_from_traces(traces)

    # Normalize the displacements
    for i in range(ntraces):
        displacements[i] = (displacements[i] - np.min(displacements[i])) / (np.max(displacements[i]) - np.min(displacements[i]))

    irregularity = np.zeros(ntraces)
    # Check how regular the traces are
    for i in range(ntraces):
        trace = traces[i]
        if np.max(trace)-np.min(trace) < 0.3:
            irregularity[i] = 1e6
            continue
            
        individual_peaks = find_individual_peaks(displacements[i])
        if len(individual_peaks) == 0:
            irregularity[i] = 1e6
            continue
        
        if np.min(individual_peaks) > 0.3:
            irregularity[i] = 1e6
            continue
        if np.max(individual_peaks) < 0.7:
            irregularity[i] = 1e6
            continue

        individual_peaks = np.array(individual_peaks)
        if len(individual_peaks) > 5:
            individual_peaks = individual_peaks[1:-1]
        temporal_std = np.std(individual_peaks, axis=0)
        irregularity[i] = np.mean(temporal_std)

        # plt.figure()
        # plt.plot(individual_peaks.T)

    # Return the traces with the lowest irregularity
    return traces[np.argmin(irregularity)], np.argmin(irregularity)
    

def find_individual_peaks(trace):
    aux_trace = trace[len(trace)//4:3*len(trace)//4]
    prominence = (np.max(aux_trace) - np.min(aux_trace))*0.4
    peaks, props = find_peaks(trace, prominence=prominence, width=5)
    if len(peaks) < 3:
        return []
    rate = np.mean(np.diff(props['left_ips']))
    width = int(rate)
    left_side = props['left_ips'] - props['left_bases']
    left_side = np.mean(left_side)

    # Split mean trace into individual peaks centered in the peaks
    individual_peaks = []
    for i in range(len(peaks)):
        start = int(props['left_ips'][i]) - int(left_side)
        end = start + width
        if start < 0: continue
        if end > len(trace): break
        peak_segment = trace[start:end]
        individual_peaks.append(peak_segment)

    return individual_peaks


def get_best_trace(traces):
    # Grab best trace
    trace, which = check_traces_2(traces)
    if which == 0:
        displacement = trace - np.min(trace)
    elif which == 1:
        displacement = -(trace - np.max(trace))

    prominence = (np.max(displacement) - np.min(displacement))*0.5
    peaks_mean, prop_mean = find_peaks(displacement, prominence=prominence, width=5, distance=10)   
    if len(peaks_mean) < 1:
        return np.zeros_like(displacement), np.zeros_like(displacement), 0, np.zeros_like(displacement)
    rate_mean = np.mean(np.diff(peaks_mean))
    width = int(rate_mean)

    # Split mean displacement into individual peaks centered in the peaks
    individual_peaks = []
    for i in range(len(peaks_mean)):
        start = int(prop_mean['left_ips'][i]) - int(width/4)
        end = start + width
        if start < 0: continue
        if end > len(displacement): break
        peak_segment = displacement[start:end]
        individual_peaks.append(peak_segment)

    # Get a mean individual peak
    mean_individual_peak = np.mean(individual_peaks, axis=0)

    # Making sure the curve is zero at the beggining and at the end
    x=np.linspace(0,1,len(mean_individual_peak))
    weight = ((np.tanh((x-0.1)*40)+1)/2)*((np.tanh((-x+0.9)*40)+1)/2)

    return mean_individual_peak*weight, traces[which], which, individual_peaks


def get_mean_trace(traces):
    point_1_left, point_2_left, point_1_right, point_2_right = traces

    point_1_disp_left = point_1_left - np.min(point_1_left)
    point_2_disp_left = point_2_left - np.min(point_2_left)
    point_1_disp_right = -(point_1_right - np.max(point_1_right))
    point_2_disp_right = -(point_2_right - np.max(point_2_right))

    mean_trace = np.nanmean([point_1_disp_left, point_2_disp_left, point_1_disp_right, point_2_disp_right], axis=0)

    prominence = (np.max(mean_trace) - np.min(mean_trace))*0.5
    peaks_mean, prop_mean = find_peaks(mean_trace, prominence=prominence, width=10)
    if len(peaks_mean) < 3:
        return np.zeros_like(mean_trace), np.zeros_like(mean_trace), np.zeros_like(mean_trace)
    rate_mean = np.mean(np.diff(peaks_mean))
    width = int(rate_mean)

    # Split mean trace into individual peaks centered in the peaks
    individual_peaks = []
    for i in range(len(peaks_mean)):
        start = int(prop_mean['left_ips'][i]) - int(width/4)
        end = start + width
        if start < 0: continue
        if end > len(mean_trace): break
        peak_segment = mean_trace[start:end]
        individual_peaks.append(peak_segment)

    # Get a mean individual peak
    mean_individual_peak = np.mean(individual_peaks, axis=0)

    # Making sure the curve is zero at the beggining and at the end
    x=np.linspace(0,1,len(mean_individual_peak))
    weight = ((np.tanh((x-0.1)*40)+1)/2)*((np.tanh((-x+0.9)*40)+1)/2)

    return mean_individual_peak*weight, mean_trace, individual_peaks


def get_tissue_width(values, traces, ls, plot=False):
    if len(traces) == 4:
        point_1_left, point_2_left, point_1_right, point_2_right = traces

        if np.any(np.isnan(point_1_left)):
            middle_tissue = np.nanmean([point_2_left, point_1_right], axis=0)
        elif np.any(np.isnan(point_2_left)):
            middle_tissue = np.nanmean([point_1_left, point_2_right], axis=0)
        elif np.any(np.isnan(point_1_right)):
            middle_tissue = np.nanmean([point_1_left, point_2_right], axis=0)
        elif np.any(np.isnan(point_2_right)):
            middle_tissue = np.nanmean([point_2_left, point_1_right], axis=0)
        else:
            middle_tissue = np.nanmean(traces, axis=0)
        middle_tissue = int(np.mean(middle_tissue))

    elif len(traces) == 2:
        middle_tissue = int(np.mean([traces[0], traces[1]]))

    mid_values = values[:, :, middle_tissue]
    mid_values = (mid_values - mid_values.min()) / (mid_values.max() - mid_values.min())

    arr = filters.sobel(filters.unsharp_mask(mid_values, radius=5, amount=10), axis=1)
    arr = filters.gaussian(np.abs(arr), sigma=2)
    arr = np.abs(filters.sobel(arr))

    mult = 1.0
    mask_size = 0
    while mask_size < arr.size//6:
        mask = arr > filters.threshold_otsu(arr)*mult
        mask = morphology.binary_closing(mask, footprint=morphology.disk(10))
        mask_size = np.sum(mask)
        mult -= 0.1
        if mult < 0:
            raise ValueError('Could not find a good threshold')

    # Label connected regions of the mask
    labeled_mask = measure.label(mask)

    # Find the largest connected component
    regions = measure.regionprops(labeled_mask)
    largest_region = max(regions, key=lambda r: r.area)

    # Create a mask for the largest connected component
    mask = labeled_mask == largest_region.label
    mask = morphology.binary_closing(mask, footprint=morphology.disk(10))

    mask = filters.gaussian(mask, 5) > 0.5

    # Need to close all the holes in the mask
    aux_mask = np.pad(mask, 1, mode='constant')
    aux_mask[0] = 1
    
    seed = np.copy(aux_mask)
    seed[1:-1, 1:-1] = aux_mask.max()
    aux_mask = morphology.reconstruction(seed, aux_mask, method='erosion')
    mask = aux_mask[1:-1, 1:-1]

    mask = mask.astype(int) - morphology.binary_erosion(mask, footprint=np.ones((1,3), dtype=int))
    points = np.where(mask==1)[1]
    points = points.reshape([-1,2])
    width = points[:,1] - points[:,0]

    # Get the width in microns
    width = width / ls

    if plot:
        plt.figure()
        plt.imshow(mid_values, cmap='gray')
        mask = morphology.binary_dilation(mask, morphology.disk(1))
        mask = mask.astype(float)
        mask[mask == 0] = np.nan
        plt.imshow(mask, cmap='bwr', vmin=0, vmax=1)
        plt.axis('off')
        
    return width

def analyze_post_displacement(trace, ls):
    time, force = trace[:,0], trace[:,1]
    dt = time[1] - time[0]
    # Find max force
    max_disp = np.max(force)
    # Find peaks
    prominence = (np.max(force) - np.min(force))*0.4
    peaks, _ = find_peaks(force, prominence=prominence, width=5, distance=10)

    # Find peak times
    peak_times = peaks*dt

    cycle_times = np.diff(peak_times)
    # Find irregularity
    irregularity = np.std(cycle_times)
    # Find bpm
    bpm = 60/np.mean(cycle_times)
    
    return max_disp / ls, irregularity, bpm, peaks

"""
PLOTS
"""
def plot_tissue_rotation(zero_frame, angle, sum_values):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10),  gridspec_kw={'height_ratios': [3, 1]})

    ax[0].imshow(transform.rotate(zero_frame, angle=np.rad2deg(angle)), cmap='gray')
    ax[0].set_title('Rotated Zero Frame')
    ax[0].axis('off')

    # Bottom subplot: Sum of the values along axis=0
    ax[1].plot(sum_values, 'k')
    ax[1].set_xlim(0, len(sum_values))
    ax[1].set_title('Sum of Values along Axis=0')
    ax[1].set_xlabel('Column Index')
    ax[1].set_ylabel('Sum of Values')

    plt.tight_layout()


def get_displacements_from_traces(traces, which=None):
    if len(traces) > 4:
        traces = np.array(traces).reshape(1, -1)

    ntraces = len(traces)
    if ntraces == 1:
        if which == 0:
            displacements = traces[0] - np.min(traces[0])
        else:
            displacements = -(traces[0] - np.max(traces[0]))
    elif ntraces == 2:
        displacements = np.zeros_like(traces)
        for i in range(2):
            if i == 0:
                displacements[i] = traces[i] - np.min(traces[i])
            else:
                displacements[i] = -(traces[i] - np.max(traces[i]))
        
        lss = ['r', 'b']
    else:
        displacements = np.zeros_like(traces)
        for i in range(4):
            if i < 2:
                displacements[i] = traces[i] - np.min(traces[i])
            else:
                displacements[i] = -(traces[i] - np.max(traces[i]))
    
    if which is not None and ntraces > 1:
        displacements = displacements[which]
    return displacements


def plot_time_traces(traces, zero_frame, box, angle, all_frame_values):
    ntraces = len(traces)
    displacements = get_displacements_from_traces(traces)
    if ntraces == 2:
        lss = ['r', 'b']
    else:
        lss = ['r', 'r--', 'b', 'b--']
            
    ii, jj = get_box_grid(box, angle, center=box.mean(axis=0))
    values = evaluate_image_in_grid(ii, jj, zero_frame)
    values = (values - values.min()) / (values.max() - values.min())
    ii, jj = get_box_grid(box, 0, center=box.mean(axis=0))

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax_big = fig.add_subplot(gs[:, 1])

    ax0.scatter(ii, jj, c=values, s=1, cmap='viridis')
    ax0.axis('off')
    ax0.set_aspect('equal')

    # Big subplot
    ax_big.imshow(1-all_frame_values, aspect='auto', cmap='viridis')
    for i in range(ntraces):
        ax_big.plot(traces[i], np.arange(0, traces[i].shape[0], 1), lss[i])
    ax_big.axis('off')

    # Second subplot
    for i in range(ntraces):
        ax1.plot(displacements[i], lss[i], label=f'Trace {i}')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Displacement (pix)')

    plt.tight_layout()

# def plot_time_traces(traces, zero_frame, box, angle, all_frame_values):
#     point_1_left, point_2_left, point_1_right, point_2_right = traces

#     point_1_disp_left = point_1_left - np.min(point_1_left)
#     point_2_disp_left = point_2_left - np.min(point_2_left)
#     point_1_disp_right = -(point_1_right - np.max(point_1_right))
#     point_2_disp_right = -(point_2_right - np.max(point_2_right))

#     ii, jj = get_box_grid(box, angle, center=box.mean(axis=0))
#     values = evaluate_image_in_grid(ii, jj, zero_frame)
#     values = (values - values.min()) / (values.max() - values.min())
#     ii, jj = get_box_grid(box, 0, center=box.mean(axis=0))

#     fig = plt.figure(figsize=(10, 6))
#     gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])

#     ax0 = fig.add_subplot(gs[0, 0])
#     ax1 = fig.add_subplot(gs[1, 0])
#     ax_big = fig.add_subplot(gs[:, 1])

#     ax0.scatter(ii, jj, c=values, s=1, cmap='viridis')
#     ax0.axis('off')
#     ax0.set_aspect('equal')

#     # Big subplot
#     ax_big.imshow(1-all_frame_values, aspect='auto', cmap='viridis')
#     ax_big.plot(point_1_left, np.arange(0, point_1_left.shape[0], 1), 'r')
#     ax_big.plot(point_2_left, np.arange(0, point_2_left.shape[0], 1), 'r--')
#     ax_big.plot(point_1_right, np.arange(0, point_1_right.shape[0], 1), 'b')
#     ax_big.plot(point_2_right, np.arange(0, point_2_right.shape[0], 1), 'b--')
#     ax_big.axis('off')

#     # Second subplot
#     ax1.plot(point_1_disp_left, 'r', label='Point 1 Left')
#     ax1.plot(point_2_disp_left, 'r--', label='Point 2 Left')
#     ax1.plot(point_1_disp_right, 'b', label='Point 1 Right')
#     ax1.plot(point_2_disp_right, 'b--', label='Point 2 Right')
#     ax1.set_xlabel('Frame')
#     ax1.set_ylabel('Displacement (pix)')

#     plt.tight_layout()

def plot_stack_traces(traces, best_trace, which_best, all_frame_vals, box, angle, peaks, tif_stack):
    ntraces = len(traces)
    if which_best == 0:
        curve = best_trace - np.min(best_trace)
    else:
        curve = -(best_trace - np.max(best_trace))

    if len(peaks) < 2:
        rest_idx = 0
        peak_idx = 0
    else:
        rest_frames= np.where(curve == 0)[0]
        rest_frame = np.argmin(np.abs(rest_frames - (peaks[0] + peaks[1])//2))
        rest_idx = rest_frames[rest_frame]
        peak_idx = peaks[1]

    rest_frame = tif_stack[rest_idx]
    peak_frame = tif_stack[peak_idx]

    ii, jj = get_box_grid(box, angle, center=box.mean(axis=0))
    peak_values = evaluate_image_in_grid(ii, jj, peak_frame)
    # peak_values = (values - values.min()) / (values.max() - values.min())
    rest_values = evaluate_image_in_grid(ii, jj, rest_frame)
    # rest_values = (values - values.min()) / (values.max() - values.min())

    ii, jj = get_box_grid(box, 0, center=box.mean(axis=0))

    arr = np.vstack((rest_values, peak_values))
    max_val = np.max(arr)
    width = arr.shape[0]
    arr2 = (1-all_frame_vals)*max_val
    arr2 = transform.resize(arr2, (arr2.shape[0]*2, arr2.shape[1]), order=1)
    arr2 = np.vstack((arr, arr2)).T
    aspect_ratio = arr2.shape[1]/arr2.shape[0]

    funcs = []
    for i in range(len(traces)):
        funcs.append(interp1d(np.arange(0, traces[i].shape[0], 1), traces[i]))
        
    aux = np.arange(0, all_frame_vals.shape[0]-0.5, 0.5)
    aux2 = np.arange(0, all_frame_vals.shape[0]*2-1, 1)

    if ntraces == 2:
        lss = ['r', 'b']
    else:
        lss = ['r', 'r--', 'b', 'b--']
    plt.figure(figsize=(5*aspect_ratio,5))
    plt.imshow(arr2, cmap='gray')
    for i in range(len(traces)):
        plt.plot(aux2+width, funcs[i](aux), lss[i])
        plt.plot(np.arange(0, width//2, 1), np.full(width//2, traces[i][rest_idx]), lss[i])
        plt.plot(np.arange(0, width//2, 1)+width//2, np.full(width//2, traces[i][peak_idx]), lss[i])
    
    
    plt.plot(np.arange(0, width//2, 1), np.full(width//2, traces[which_best][rest_idx]), 'k')
    plt.plot(np.arange(0, width//2, 1)+width//2, np.full(width//2, traces[which_best][peak_idx]), 'k')

    plt.axis('off')


def plot_traces_and_mean(traces, mean_trace, which_best, mean_individual_trace, individual_traces):
    ntraces = len(traces)
    lss = ['r', 'r--', 'b', 'b--']  
    best_disp = get_displacements_from_traces(mean_trace, which=which_best)
    displacements = get_displacements_from_traces(traces)
    peaks = []
    for i in range(len(displacements)):
        prominence = (np.max(displacements[i]) - np.min(displacements[i]))*0.4
        peaks.append(find_peaks(displacements[i], prominence=prominence)[0])

    prominence = (np.max(best_disp) - np.min(best_disp))*0.4
    peaks_mean, prop_mean = find_peaks(best_disp, prominence=prominence, width=30)

    # Combine the two plots in one figure with two subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [2.5, 1]})

    frames = np.arange(0, len(best_disp), 1)
    # Right subplot: Displacement traces
    for i in range(ntraces):
        ax1.plot(frames, displacements[i], lss[i], label=f'Trace {i}', alpha=0.5)
    ax1.plot(frames, best_disp, 'k', label='Mean Trace', lw=2)

    # Add peaks to the plot
    for i in range(ntraces):
        ax1.plot(peaks[i], displacements[i][peaks[i]], 'rx')
        
    ax1.plot(peaks_mean, best_disp[peaks_mean], 'kx')

    ylim = ax1.get_ylim()
    ax1.vlines(prop_mean['left_ips'], ymin=ylim[0], ymax=ylim[1])

    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Displacement (pix)')
    ax1.set_xlim([frames[0], frames[-1]])
    ax1.set_ylim(ylim)

    # Left subplot: Individual peaks combined in one plot
    for i, peak_segment in enumerate(individual_traces):
        ax2.plot(peak_segment, label=f'Peak {i + 1}', alpha=0.5)

    ax2.plot(mean_individual_trace, 'k', lw=2, label='Mean Peak')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Displacement (pix)')
    ax2.set_xlim([0, len(mean_individual_trace) - 1])

    # Share y-limits between the two subplots
    ax2.set_ylim(ylim)

    plt.tight_layout()


class BoxSelector:

    def __init__(self, ax, img, box0=None):
        self.canvas = ax.figure.canvas
        self.img = img
        self.box = box0
        self.ax = ax
        self.verts = []

        self.poly = PolygonSelector(ax, self.onselect, props=dict(color='r', linestyle='-', linewidth=2, alpha=0.5))

        self.ax.imshow(self.img, cmap='gray')
        self.ax.set_title('Select the four vertices of the box and press Enter to confirm')
        if self.box is not None:
            self.verts = self.box
            self.lines = self.plot_box(self.box)

        self.canvas.mpl_connect('key_press_event', self.on_key)

    def onselect(self, verts):
        self.reset()
        self.verts = verts
        self.canvas.draw_idle()

    def plot_box(self, verts):
        verts = np.array(verts)
        l1 = self.ax.plot([verts[0, 0], verts[1, 0]], [verts[0, 1], verts[1, 1]], 'r-')
        l2 = self.ax.plot([verts[1, 0], verts[2, 0]], [verts[1, 1], verts[2, 1]], 'r-')
        l3 = self.ax.plot([verts[2, 0], verts[3, 0]], [verts[2, 1], verts[3, 1]], 'r-')
        l4 = self.ax.plot([verts[3, 0], verts[0, 0]], [verts[3, 1], verts[0, 1]], 'r-')
        return (l1, l2, l3, l4)

    def reset(self):
        if hasattr(self, 'lines'):
            for line in self.lines:
                for l in line:
                    l.remove()
        self.lines = []

    def disconnect(self):
        self.poly.disconnect_events()
        self.canvas.draw_idle()

    def get_box(self):
        return np.array(self.verts)

    def on_key(self, event):
        if event.key == 'enter':
            self.box = get_box_from_box(self.get_box())
            plt.close(self.ax.figure)




def interactive_box_selection(zero_frame, box):
    zero_frame = exposure.equalize_hist(zero_frame)

    _, ax = plt.subplots()
    selector = BoxSelector(ax, zero_frame, box)
    plt.show()

    return selector.box


def get_box_from_box(box_):
    rect = cv2.minAreaRect(box_.astype(int))
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    length = np.linalg.norm(box[3] - box[0])
    width = np.linalg.norm(box[1] - box[0])
    if length < width:
        length = np.linalg.norm(box[1] - box[0])
        width = np.linalg.norm(box[2] - box[1])
        aux_box = np.array([box[0], box[3], box[2], box[1]])
        box = aux_box
    return box
