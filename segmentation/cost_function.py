# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:23:39 2024

cost function and optimization of Frangi method

@author: helioum
"""

import numpy as np
import frangi
from scipy.optimize import fmin_powell
import cthresholding as cp_th
import manage_data as md

# test for 3D volume - KESM
volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

# the cropped sizes can be changed
sample_vol = volume[200:400, 10:210, 100:300]
sample_gr = gr_truth[200:400, 10:210, 100:300]

#%%
# params = A, B, C
def cost_frangi(params, sample_vol, sample_gr):
    # test for 3D volume - KESM
    A, B, C = params
    print('a:', round(A, 3), 'b:', round(B, 3), 'c:', round(C, 3), end='\t')
    
    scale_range = np.arange(1.5, 4, 0.5)
    
    output = frangi.upgrade_vesselness(sample_vol, A, B, C, scale_range, 'white')
    
    predicted, best_thresh = cp_th.compute_otsu_img(output,  background='black')
    TP = np.logical_and(sample_gr, predicted).sum()
    FP = np.logical_and(np.logical_not(sample_gr), predicted).sum()
    FN = np.logical_and(sample_gr, np.logical_not(predicted)).sum()     
    
    dice = (2*TP) / float(2*TP + FP + FN)
    print('\tdice=', round(dice, 4), '\n')
    return 1 - dice


''' Normal Frangi '''
# Initial guess for parameters
scale = [1.5, 2, 2.5, 3, 3.5]
half_norm = frangi.max_norm(sample_vol, scale) / 2

# Set initial parameter and bounds for Beyond Frangi optimization
initial_params = [0.5, 0.5, half_norm]

# Run the optimization for Beyond Frangi
result_b = fmin_powell(cost_frangi, initial_params, args=(sample_vol, sample_gr), maxiter=1, maxfun=1)

print('A:\t', result_b[0])
print('B:\t', result_b[1])
print('C:\t', result_b[2])


