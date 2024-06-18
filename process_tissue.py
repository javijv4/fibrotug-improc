#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 08:54:46 2024

@author: Javiera Jilberto Vallejos
"""

import os
import numpy as np
from FtugTissue import FtugTissue
import matplotlib.pyplot as plt

tissue_fldr = '../DSP/Tissues/dataset2/gem02/day7/'
png_dump = tissue_fldr + 'png_dump/'
if not os.path.exists(png_dump):
    os.makedirs(png_dump)

images = {'dsp': 'A_0.41_0.1_GEM12_s1_day7-02_MAX_c1_ORG.tif',
          'fibers': 'A_0.41_0.1_GEM12_s1_day7-02_MAX_c2_ORG.tif',
          'actin': 'A_0.41_0.1_GEM12_s1_day7-02_MAX_c3_ORG.tif'}


# images = {'dsp': 'A_0.41_0.1_GEM12_s1_day9-02_MAX_c2_ORG.tif',
#           'fibers': 'A_0.41_0.1_GEM12_s1_day9-02_MAX_c3_ORG.tif',
#           'actin': 'A_0.41_0.1_GEM12_s1_day9-02_MAX_c4_ORG.tif'}

tissue = FtugTissue(tissue_fldr, images)
tissue.plot_images(png_dump)

tissue.get_tissue_mask()
tissue.plot_tissue_mask(png_dump)

# Fiber processing
tissue.get_fiber_mask()
tissue.process_fibers()
tissue.plot_fiber_processing(png_dump)

# Actinin processing
tissue.get_actin_blob_mask()
tissue.get_actin_mask()
tissue.process_actin()
tissue.plot_actin_processing(png_dump)

# DSP processing
tissue.process_dsp(force_compute=True)
tissue.plot_dsp_processing(png_dump)
tissue.plot_dsp_processing_zoom(png_dump, [740, 840, 140, 240])
