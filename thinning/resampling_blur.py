# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:39:34 2024

@author: helioum
"""
import numpy as np
import scipy.ndimage as nd
import manage_data as md
import nrrd 

gr_truth = np.load('ground_truth_kesm.npy')
sample_gr = gr_truth[0:200, 200:400, 300:500]

#%%
# resample raw and ground truth volumes (z-axis x2)
resampled_gr = md.resample(sample_gr, (1, 1, 1), (0.3906, 0.3906, 0.1953))
#nrrd.write('gr_truth(resampled).nrrd', resampled_gr)
#%%
# blur the volumes
ground_blurred = nd.median_filter(resampled_gr, size=(7, 7, 7))
binary_volume = np.where(ground_blurred > 0.1, 1, 0)
#%%
# save nrrd files
nrrd.write('gr_truth(blurred)_3.nrrd', binary_volume)
