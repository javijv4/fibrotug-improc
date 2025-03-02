#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/27 13:24:49

@author: Javiera Jilberto Vallejos 
'''

from matplotlib import pyplot as plt
import numpy as np
from skimage import io, filters, measure, draw, morphology, transform, segmentation
import cv2
from scipy.optimize import minimize
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.signal import find_peaks

def estimate_tissue_rectangle(img, plot=False):

    portion = 1
    mult = 1

    # Threshold image using otsu
    while portion > 0.1:
        thresh = filters.threshold_otsu(img)*mult
        binary = img > thresh
        binary = 1-binary

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
            raise ValueError('Could not find a good threshold')


    # Find contours
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [np.vstack(contours).squeeze()]

    # Get the minimum area rectangle for each contour
    min_area_rects = [cv2.minAreaRect(contour) for contour in contours]

    # Draw the rectangles on the original image
    for rect in min_area_rects:
        box = cv2.boxPoints(rect)
        box = np.int0(box)

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
        # plt.plot(np.append(box[:, 0], box[0, 0]), np.append(box[:, 1], box[0, 1]), 'r-')
        plt.plot(box[:, 0], box[:, 1], 'r-')
        plt.show()

    # Sanity checks
    aspect_ratio = length / width
    if aspect_ratio < 1.5:
        raise ValueError('The aspect ratio is too small')
    if length < 500 or length > 700:
        raise ValueError('The length of the box is out of bounds')


    return box


def get_box_grid(box, rotation=0, center=(0, 0)):
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
        ii -= np.mean(ii)- center[1]
        jj -= np.mean(jj) - center[0]

    return ii, jj


def rotate_and_evaluate(box, zero_frame, centroid, angle):
    if not isinstance(angle, float):
        angle = angle[0]

    ii, jj = get_box_grid(box, angle, centroid)

    # Evaluate frame_zero at the grid points
    zero_frame = (zero_frame - np.min(zero_frame)) / (np.max(zero_frame) - np.min(zero_frame))
    max_val = np.max(zero_frame)
    values = np.array([zero_frame[int(j), int(i)] 
                       if 0 <= int(i) < zero_frame.shape[1] and 0 <= int(j) < zero_frame.shape[0] 
                       else max_val for i, j in zip(ii.flatten(), jj.flatten())])
    values = values.reshape(ii.shape)

    return ii, jj, values, np.sum(values, axis=0)


def evaluate_image_in_grid(ii, jj, img):
    max_val = np.max(img)
    values = np.array([img[int(j), int(i)] 
                       if 0 <= int(i) < img.shape[1] and 0 <= int(j) < img.shape[0] 
                       else max_val for i, j in zip(ii.flatten(), jj.flatten())])
    values = values.reshape(ii.shape)
    
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
    result = minimize(objective_function, initial_angle, bounds=[(initial_angle - np.pi / 4, initial_angle + np.pi / 4)])
    best_angle = result.x[0]

    return best_angle


def get_mask(diff, rescale=3):
    frangi = filters.frangi(diff, sigmas=range(rescale,rescale*3,rescale*2))
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
    counts, pos = np.histogram(points, bins=bins)
    rest_val = pos[np.argmax(counts)]
    close_val = points[np.argmin(np.abs(points - rest_val))]

    if left_or_right == 'left':
        points[points < close_val] = close_val
    else:
        points[points > close_val] = close_val

    return points


def get_post_boundaries(vals, rescale=4, which='left'):
    nframes = vals.shape[0]
    vals = transform.rescale(vals, rescale, order=1, mode='reflect', anti_aliasing=False)
    try: 
        diff = np.gradient(vals, axis=1)
        mask, mask_1, mask_2 = get_mask(diff, rescale=rescale)
    except ValueError as e:
        diff = np.gradient(vals[:,::-1], axis=1)[:,::-1]
        mask, mask_1, mask_2 = get_mask(diff, rescale=rescale)
    
        
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
    point_1_left, point_2_left = get_post_boundaries(all_frame_vals[:, :all_frame_vals.shape[1] // 3])
    point_1_right, point_2_right = get_post_boundaries(all_frame_vals[:, all_frame_vals.shape[1] // 3 * 2:], which='right')
    point_1_right += all_frame_vals.shape[1] // 3 * 2
    point_2_right += all_frame_vals.shape[1] // 3 * 2 


    return point_1_left, point_2_left, point_1_right, point_2_right


def get_mean_trace(traces):
    point_1_left, point_2_left, point_1_right, point_2_right = traces

    point_1_disp_left = point_1_left - np.min(point_1_left)
    point_2_disp_left = point_2_left - np.min(point_2_left)
    point_1_disp_right = -(point_1_right - np.max(point_1_right))
    point_2_disp_right = -(point_2_right - np.max(point_2_right))

    mean_trace = (point_1_disp_left + point_2_disp_left + point_1_disp_right + point_2_disp_right) / 4

    point_1_disp_left = point_1_left - np.min(point_1_left)
    prominence = (np.max(point_1_disp_left) - np.min(point_1_disp_left))*0.6
    peaks_mean, prop_mean = find_peaks(mean_trace, prominence=prominence, width=10)
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

    return mean_individual_peak, mean_trace, individual_peaks


def get_tissue_width(values, traces, ls, plot=False):
    middle_tissue = np.mean(traces, axis=0)
    middle_tissue = int(np.mean(middle_tissue))

    mid_values = values[:, :, middle_tissue]
    mid_values = (mid_values - mid_values.min()) / (mid_values.max() - mid_values.min())

    mask_1 = segmentation.flood_fill(mid_values, (0, 0), -1, tolerance=0.1)
    mask_2 = segmentation.flood_fill(mid_values, (mid_values.shape[0] - 1, mid_values.shape[1] - 1), -1, tolerance=0.1)

    mask = 1 - ((mask_1 == -1) + (mask_2 == -1))

    # Label connected regions of the mask
    labeled_mask = measure.label(mask)

    # Find the largest connected component
    regions = measure.regionprops(labeled_mask)
    largest_region = max(regions, key=lambda r: r.area)

    # Create a mask for the largest connected component
    mask = labeled_mask == largest_region.label
    mask = mask.astype(int) - morphology.binary_erosion(mask, footprint=np.ones((1,3), dtype=int))
    points = np.where(mask==1)[1]
    points = points.reshape([-1,2])
    width = points[:,1] - points[:,0]

    # Get the width in microns
    width = width / ls

    if plot:
        plt.figure()
        plt.imshow(mid_values, cmap='gray')
        plt.imshow(mask, cmap='viridis', alpha=0.5)

    return width

def analyze_post_displacement(trace, ls):
    time, force = trace[:,0], trace[:,1]
    dt = time[1] - time[0]
    # Find max force
    max_disp = np.max(force)
    # Find peaks
    peaks, _ = find_peaks(force, height=max_disp/10)
    # Find peak times
    peak_times = peaks*dt
    # Find irregularity
    irregularity = np.std(np.diff(peak_times))
    # Find bpm
    bpm = 60/irregularity
    
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


def plot_time_traces(traces, zero_frame, box, angle, all_frame_values):
    point_1_left, point_2_left, point_1_right, point_2_right = traces

    point_1_disp_left = point_1_left - np.min(point_1_left)
    point_2_disp_left = point_2_left - np.min(point_2_left)
    point_1_disp_right = -(point_1_right - np.max(point_1_right))
    point_2_disp_right = -(point_2_right - np.max(point_2_right))

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
    arr = np.vstack((1-values, 1-all_frame_values))
    ax_big.imshow(arr, aspect='auto', cmap='viridis')
    ax_big.plot(point_1_left, np.arange(0, point_1_left.shape[0], 1), 'r')
    ax_big.plot(point_2_left, np.arange(0, point_2_left.shape[0], 1), 'r--')
    ax_big.plot(point_1_right, np.arange(0, point_1_right.shape[0], 1), 'b')
    ax_big.plot(point_2_right, np.arange(0, point_2_right.shape[0], 1), 'b--')
    ax_big.axis('off')

    # Second subplot
    ax1.plot(point_1_disp_left, 'r', label='Point 1 Left')
    ax1.plot(point_2_disp_left, 'r--', label='Point 2 Left')
    ax1.plot(point_1_disp_right, 'b', label='Point 1 Right')
    ax1.plot(point_2_disp_right, 'b--', label='Point 2 Right')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Displacement (pix)')

    plt.tight_layout()

def plot_stack_traces(traces, all_frame_vals, box, angle, peaks, tif_stack):
    point_1_left, point_2_left, point_1_right, point_2_right = traces
    point_1_disp_left = point_1_left - np.min(point_1_left)

    rest_frames= np.where(point_1_disp_left == 0)[0]
    rest_frame = np.argmin(np.abs(rest_frames - (peaks[0] + peaks[1])//2))
    rest_idx = rest_frames[rest_frame]
    rest_frame = tif_stack[rest_idx]

    peak_idx = peaks[1]
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

    f1 = interp1d(np.arange(0, point_1_left.shape[0], 1), point_1_left)
    f2 = interp1d(np.arange(0, point_2_left.shape[0], 1), point_2_left)
    f3 = interp1d(np.arange(0, point_1_right.shape[0], 1), point_1_right)
    f4 = interp1d(np.arange(0, point_2_right.shape[0], 1), point_2_right)
    aux = np.arange(0, all_frame_vals.shape[0]-0.5, 0.5)
    aux2 = np.arange(0, all_frame_vals.shape[0]*2-1, 1)

    plt.figure(figsize=(5*aspect_ratio,5))
    plt.imshow(arr2, cmap='gray')
    plt.plot(aux2 + width, f1(aux), 'r')
    plt.plot(aux2 + width, f2(aux), 'r--')
    plt.plot(aux2 + width, f3(aux), 'b')
    plt.plot(aux2 + width, f4(aux), 'b--')
    plt.plot(np.arange(0, width//2, 1), np.full(width//2, point_1_left[rest_idx]), 'r')
    plt.plot(np.arange(0, width//2, 1), np.full(width//2, point_2_left[rest_idx]), 'r--')
    plt.plot(np.arange(0, width//2, 1), np.full(width//2, point_1_right[rest_idx]), 'b')
    plt.plot(np.arange(0, width//2, 1), np.full(width//2, point_2_right[rest_idx]), 'b--')
    plt.plot(np.arange(0, width//2, 1)+width//2, np.full(width//2, point_1_left[peak_idx]), 'r')
    plt.plot(np.arange(0, width//2, 1)+width//2, np.full(width//2, point_2_left[peak_idx]), 'r--')
    plt.plot(np.arange(0, width//2, 1)+width//2, np.full(width//2, point_1_right[peak_idx]), 'b')
    plt.plot(np.arange(0, width//2, 1)+width//2, np.full(width//2, point_2_right[peak_idx]), 'b--')
    plt.axis('off')


def plot_traces_and_mean(traces, mean_trace, mean_individual_trace, individual_traces):
    point_1_left, point_2_left, point_1_right, point_2_right = traces

    point_1_disp_left = point_1_left - np.min(point_1_left)
    point_2_disp_left = point_2_left - np.min(point_2_left)
    point_1_disp_right = -(point_1_right - np.max(point_1_right))
    point_2_disp_right = -(point_2_right - np.max(point_2_right))

    prominence = (np.max(point_1_disp_left) - np.min(point_1_disp_left))*0.4
    peaks_1_left = find_peaks(point_1_disp_left, prominence=prominence)[0]
    peaks_2_left = find_peaks(point_2_disp_left, prominence=prominence)[0]
    peaks_1_right = find_peaks(point_1_disp_right, prominence=prominence)[0]
    peaks_2_right = find_peaks(point_2_disp_right, prominence=prominence)[0]
    peaks_mean, prop_mean = find_peaks(mean_trace, prominence=prominence, width=5)


    # Combine the two plots in one figure with two subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [2.5, 1]})

    frames = np.arange(0, len(mean_trace), 1)
    # Right subplot: Displacement traces
    ax1.plot(frames, point_1_disp_left, 'r', label='Point 1 Left', alpha=0.5)
    ax1.plot(frames, point_2_disp_left, 'r--', label='Point 2 Left', alpha=0.5)
    ax1.plot(frames, point_1_disp_right, 'b', label='Point 1 Right', alpha=0.5)
    ax1.plot(frames, point_2_disp_right, 'b--', label='Point 2 Right', alpha=0.5)
    ax1.plot(frames, mean_trace, 'k', label='Mean Trace', lw=2)

    # Add peaks to the plot
    ax1.plot(peaks_1_left, point_1_disp_left[peaks_1_left], 'rx')
    ax1.plot(peaks_2_left, point_2_disp_left[peaks_2_left], 'rx')
    ax1.plot(peaks_1_right, point_1_disp_right[peaks_1_right], 'bx')
    ax1.plot(peaks_2_right, point_2_disp_right[peaks_2_right], 'bx')
    ax1.plot(peaks_mean, mean_trace[peaks_mean], 'kx')

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
