#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/27 11:56:11

@author: Javiera Jilberto Vallejos 
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from vidprocessing import vidutils as vut
import cheartio as chio
import glob
import time as tm

dataset_fldr = '/home/jilberto/Dropbox (University of Michigan)/Projects/fibroTUG/DSP/Tissues/dataset2_2/'
tif_files = glob.glob(f'{dataset_fldr}/**/*corrected.tif', recursive=True)

# Movie params
fps = 65
ls = 1/0.908
tissue_depth = 12
pillar_stiffnes = 0.41

nfailed = 0
nsuccess = 0

#%%
# tif_files = ['/home/jilberto/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/fibroTUG/DSP/Tissues/dataset2_2/elk02/post/A_0.41_0.1_ELK_s1-day9-ET1-02_corrected.tif']
for tif_stack_path in tif_files[:1]:
    # try:
        start = tm.time()
        print(f'Processing {tif_stack_path}')
        tif_stack = io.imread(tif_stack_path)
        zero_frame = tif_stack[0]

        # Deal with path
        fldr = os.path.dirname(tif_stack_path) + '/'
        png_fldr = fldr + 'png_dump/'
        if not os.path.exists(png_fldr):
            os.makedirs(png_fldr)

        if ('day9' in fldr) or ('post' in fldr)  or ('3days' in tif_stack_path):
            name = 'day9'
        else:
            if '6hrs' in tif_stack_path:
                name = 'day7_6hrs'
            elif 'ET1' in tif_stack_path:
                name = 'day7_ET1'
            else:
                name = 'day7'

        # if os.path.exists(f'{fldr}/{name}_parameters\.txt'):
        #     continue

        # Get a box of the tissue
        box = vut.estimate_tissue_rectangle(zero_frame, plot=False)
        # box = vut.interactive_box_selection(zero_frame, box)        

        # Find best angle
        angle = vut.find_tissue_rotation(box, zero_frame)
        #TODO make this interactive

        # Center the grid at the centroid of the mask
        ii, jj, mask = vut.get_box_grid(box, angle, center=box.mean(axis=0), img=zero_frame)

        # Evaluate grid in all frames
        all_frame_vals = []
        values_grid_time = []
        for frame in tif_stack:
            values = frame[jj,ii]
            values = (values-values.min())/(values.max()-values.min())
            values[mask] = np.median(values)
            sum_values = np.sum(values, axis=0)
            all_frame_vals.append(sum_values)
            values_grid_time.append(values)

        values_grid_time = np.array(values_grid_time)

        all_frame_vals = np.array(all_frame_vals)
        all_frame_vals = (all_frame_vals - all_frame_vals.min()) / (all_frame_vals.max() - all_frame_vals.min())
        all_frame_vals = 1 - all_frame_vals

        # Get the displacement traces 
        traces = vut.get_displacements_4(all_frame_vals, values_grid_time[0])
        vut.plot_time_traces(traces, zero_frame, box, angle, all_frame_vals)
        plt.show()
        plt.savefig(f'{png_fldr}/{name}_post_analysis.png', bbox_inches='tight')
        traces = vut.check_traces(traces)

        # Get mean
        mean_individual_trace, mean_trace, individual_traces = vut.get_mean_trace(traces)
        vut.plot_traces_and_mean(traces, mean_trace, mean_individual_trace, individual_traces)
        plt.savefig(f'{png_fldr}/{name}_post_mean_trace.png', bbox_inches='tight')
        plt.show()

        # Calculate quantities
        time = np.arange(len(mean_trace))/fps
        time_trace = np.column_stack((time, mean_trace))
        max_disp, irregularity, bpm, peaks = vut.analyze_post_displacement(time_trace, ls)
        max_force = max_disp * pillar_stiffnes

        vut.plot_stack_traces(traces, all_frame_vals, box, angle, peaks, tif_stack)
        plt.savefig(f'{png_fldr}/{name}_stack_traces.png', bbox_inches='tight')
        plt.show()

        # Calculate the width in the middle of the tissue
        width = vut.get_tissue_width(values_grid_time, traces, ls, plot=True)
        plt.savefig(f'{png_fldr}/{name}_width.png', bbox_inches='tight')
        width_at_peak = np.mean(width[peaks])

        max_stress = max_force / (width_at_peak * tissue_depth) * 1000 # kPa

        # Create dictionary with relevant values
        results = {'max_force': max_force, 'irregularity': irregularity, 'bpm': bpm, 'max_stress': max_stress, 'width': width_at_peak}
        chio.dict_to_pfile(f'{fldr}/{name}_parameters.txt', results)


        # Save mean individual trace
        disp_trace = mean_individual_trace / ls / 1000
        time = np.arange(len(disp_trace))/fps
        save = np.column_stack((time, disp_trace))
        chio.write_dfile(f'{fldr}/{name}_post_disp.INIT', save)

        nsuccess += 1
        print(f'Processed in {tm.time()-start} seconds')

    # except:
    #     print(f'Error processing {tif_stack_path}')
    #     nfailed += 1
    #     continue
print(f'Processed {nsuccess} files, failed {nfailed}')