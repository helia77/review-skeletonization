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

# test for 3D volume - KESM
volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

# the cropped sizes can be changed
sample_vol = volume[350:400, 50:100, 150:200]
sample_gr = gr_truth[350:400, 50:100, 150:200]

#%%
# params = A, B, C
def cost_frangi(params, sample_vol, sample_gr):
    # test for 3D volume - KESM
    A, B, C = params
    print('a= ', round(A, 3), 'b= ', round(B, 3), 'c=', round(C, 3), end='\t')
    #volume = np.load('whole_volume_kesm.npy')
    #gr_truth = np.load('ground_truth_kesm.npy')

    # the cropped sizes can be changed
    #sample_vol = volume[350:400, 50:100, 150:200]
    #sample_gr = gr_truth[350:400, 50:100, 150:200]
    
    all_filters = []
    
    beta  = 2 * (B**2)
    c     = 2 * (C**2)
    alpha = 2 * (A**2)
    scale_range = np.arange(3, 5, 1)
    for scale in scale_range:
        filtered_vol = frangi.vesselness_3D(sample_vol, scale, alpha, beta, c, 'white')
        all_filters.append(filtered_vol)
    
    # pick the pixels with the highest vesselness value
    max_vol = all_filters[0]
    output_vol = np.zeros(sample_vol.shape)
    #print('Getting maximum pixels...')
    for x in range(sample_vol.shape[2]):
        for y in range(sample_vol.shape[1]):
            for z in range(sample_vol.shape[0]):
                max_value = max_vol[z, y, x]
                for vol in all_filters:
                    if (vol[z, y, x] > max_value):
                        max_value = vol[z, y, x]
                output_vol[z, y, x] = max_value
    
    output_vol = np.uint8(output_vol * 255)
    
    predicted, best_thresh = cp_th.compute_otsu_img(output_vol,  background='black')
    TP = np.logical_and(sample_gr, predicted).sum()
    FP = np.logical_and(np.logical_not(sample_gr), predicted).sum()
    FN = np.logical_and(sample_gr, np.logical_not(predicted)).sum()     
    
    dice = (2*TP) / float(2*TP + FP + FN)
    print('dice= ', round(dice, 4), '\n')
    return 1 - dice


''' Normal Frangi '''
# Initial guess for parameters
scale = [3, 4, 5, 6]
half_norm = frangi.max_norm(sample_vol, scale) / 2

# Set initial parameter and bounds for Beyond Frangi optimization
initial_params = [0.5, 0.5, half_norm]

# Run the optimization for Beyond Frangi
result_b = fmin_powell(cost_frangi, initial_params, args=(sample_vol, sample_gr), maxiter=1, maxfun=1)
#%%
print('A:\t', result_b[0])
print('B:\t', result_b[1])
print('C:\t', result_b[2])


