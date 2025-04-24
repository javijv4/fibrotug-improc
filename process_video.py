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

dataset_fldr = '/Users/jjv/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/fibroTUG/DSP/Tissues/dataset3/wt07/'
tif_files = glob.glob(f'{dataset_fldr}/**/*corrected.tif', recursive=True)

# Movie params
fps = 65
ls = 1/0.908
tissue_depth = 12
pillar_stiffnes = 0.41

nfailed = 0
nsuccess = 0

#%%
st=144
tif_files = ['/Users/jjv/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/fibroTUG/DSP/Tissues/dataset1/wt05/pre/A_0.41_0.1_DSPWT-S2-05_corrected.tif']
for i, tif_stack_path in enumerate(tif_files):
    # try:
        if 'Javi DSP Videos' in tif_stack_path: continue
        # if not ('ET1' in tif_stack_path): continue #or ('day6' in tif_stack_path): continue
        # if not ('wt' in tif_stack_path): continue
        # if 'dataset1' in tif_stack_path: continue
        start = tm.time()
        print(f'Processing file {i+1+st}/{len(tif_files)}')
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

        # if os.path.exists(f'{fldr}/{name}_post_disp.INIT'):
        #     continue

        # Get a box of the tissue
        try:
            box = np.load(f'{fldr}/{name}_box.npy')
        except:
            box = vut.estimate_tissue_rectangle(zero_frame, plot=False)
            box = vut.interactive_box_selection(zero_frame, box)       
            np.save(f'{fldr}/{name}_box.npy', box)
        # box = vut.estimate_tissue_rectangle(zero_frame, plot=False)
        # box = vut.interactive_box_selection(zero_frame, box)       
        # np.save(f'{fldr}/{name}_box.npy', box)

        # Find best angle
        try:
            angle = np.load(f'{fldr}/{name}_angle.npy')
        except:
            angle = vut.find_tissue_rotation(box, zero_frame)
            angle = vut.interactive_tissue_rotation(zero_frame, angle)
            np.save(f'{fldr}/{name}_angle.npy', angle)
        # angle = vut.find_tissue_rotation(box, zero_frame)
        # angle = vut.interactive_tissue_rotation(zero_frame, angle)
        # np.save(f'{fldr}/{name}_angle.npy', angle)

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
        try:
            xlim_l, xlim_r = np.load(f'{fldr}/{name}_post_xlim.npy')
            if np.sum(xlim_l) == 0 or np.sum(xlim_r) == 0 or (xlim_l[1]-xlim_l[0] < 1e-6) or (xlim_r[1]-xlim_r[0] < 1e-6):
                raise ValueError('No xlims selected')
        except:
            xlim_l, xlim_r = vut.select_post_area(values_grid_time[0], all_frame_vals)
            print(f'xlim_l: {xlim_l}, xlim_r: {xlim_r}')
            if np.sum(xlim_l) == 0 and np.sum(xlim_r) == 0:
                raise ValueError('No xlims selected')
            else:
                np.save(f'{fldr}/{name}_post_xlim.npy', [xlim_l, xlim_r])
        # xlim_l, xlim_r = vut.select_post_area(values_grid_time[0], all_frame_vals)
        # np.save(f'{fldr}/{name}_post_xlim.npy', [xlim_l, xlim_r])

        traces = vut.get_displacements_4(all_frame_vals, xlim_l, xlim_r)
        vut.plot_time_traces(traces, zero_frame, box, angle, all_frame_vals)
        plt.savefig(f'{png_fldr}/{name}_post_analysis.png', bbox_inches='tight')



        # Get mean
        best_individual_trace, best_trace, which_best, individual_traces = vut.get_best_trace(traces)
        vut.plot_traces_and_mean(traces, best_trace, which_best, best_individual_trace, individual_traces)
        plt.savefig(f'{png_fldr}/{name}_post_best_trace.png', bbox_inches='tight')


        # Calculate quantities
        time = np.arange(len(best_trace))/fps
        disp_trace = vut.get_displacements_from_traces(best_trace, which_best)
        disp_trace = np.column_stack((time, disp_trace))
        max_disp, irregularity, bpm, peaks = vut.analyze_post_displacement(disp_trace, ls)
        max_force = max_disp * pillar_stiffnes

        vut.plot_stack_traces(traces, best_trace, which_best, all_frame_vals, box, angle, peaks, tif_stack)
        plt.savefig(f'{png_fldr}/{name}_stack_traces.png', bbox_inches='tight')

        # Calculate the width in the middle of the tissue
        width = vut.get_tissue_width(values_grid_time, traces, ls, plot=True)
        plt.savefig(f'{png_fldr}/{name}_width.png', bbox_inches='tight')
        if len(width) == 0:
            width_at_peak = 0
        else:
            width_at_peak = np.mean(width[peaks])

        max_stress = max_force / (width_at_peak * tissue_depth) * 1000 # kPa

        # Create dictionary with relevant values
        results = {'max_force': max_force, 'irregularity': irregularity, 'bpm': bpm, 'max_stress': max_stress, 'width': width_at_peak}
        chio.dict_to_pfile(f'{fldr}/{name}_parameters.txt', results)


        # Save mean individual trace
        disp_trace = best_individual_trace / ls / 1000
        time = np.arange(len(disp_trace))/fps
        save = np.column_stack((time, disp_trace))
        chio.write_dfile(f'{fldr}/{name}_post_disp.INIT', save)

        nsuccess += 1
        print(f'Processed in {tm.time()-start} seconds')

        plt.show()
        plt.close('all')

    # except:
    #     print(f'Error processing {tif_stack_path}')
    #     nfailed += 1
    #     continue

        # break   
print(f'Processed {nsuccess} files, failed {nfailed}')