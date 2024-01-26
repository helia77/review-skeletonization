# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:23:39 2024

cost function and optimization of Frangi method

@author: helioum
"""

import numpy as np
import frangi
from scipy.optimize import fmin_powell
import thresholding as th
import metric as mt

# test for 3D volume - KESM
volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

# the cropped sizes can be changed
sample_vol = volume[0:100, 300:400, 300:500]
sample_gr = gr_truth[0:100, 300:400, 300:500]


# params = A, B, C
def cost_frangi(params, sample_vol, sample_gr):
    # test for 3D volume - KESM
    A, B, C = params
    print('a:', round(A, 3), 'b:', round(B, 3), 'c:', round(C, 3), end='\t')
    
    scale_range = np.arange(1.5, 4, 0.5)
    
    output = frangi.upgrade_vesselness(sample_vol, A, B, C, scale_range, 'white')
    
    predicted, _ = th.compute_otsu_img(output,  background='black')
    TP = np.logical_and(sample_gr, predicted).sum()
    FP = np.logical_and(np.logical_not(sample_gr), predicted).sum()
    FN = np.logical_and(sample_gr, np.logical_not(predicted)).sum()     
    
    dice = (2*TP) / float(2*TP + FP + FN)
    print('\tdice=', round(dice, 4), '\n')
    return 1 - dice

def cost_auc(params, sample_vol, sample_gr):
    A, B, C = params
    print('a:', round(A, 3), 'b:', round(B, 3), 'c:', round(C, 3), end='\t')
    scale_range = np.arange(1.5, 4, 0.5)
    
    predicted = frangi.upgrade_vesselness(sample_vol, A, B, C, scale_range, 'white')
    
    if(np.unique(predicted).size > 1):
        th_range = np.delete(np.unique(predicted), 0)
    else:
        th_range = np.unique(predicted)
    precision   = np.zeros((th_range.size))
    recall      = np.zeros((th_range.size))
    for i, t in enumerate(th_range):
        # global thresholding
        threshed = (predicted >= t)
        met = mt.metric(sample_gr, threshed)
        precision[i] = met.precision()
        recall[i] = met.TPR()
    
    indices = np.argsort(recall)
    sorted_recall = recall[indices]
    sorted_precision = precision[indices]
    
    auc = np.trapz(sorted_precision, sorted_recall)
    
    return 1 - auc
    
    
    
''' Normal Frangi '''
# Initial guess for parameters
scale = [1.5, 2, 2.5, 3, 3.5]
half_norm = frangi.max_norm(sample_vol, scale) / 2

# Set initial parameter and bounds for Beyond Frangi optimization
initial_params = [0.5, 0.5, half_norm]

# Run the optimization for Beyond Frangi
result_b = fmin_powell(cost_frangi, initial_params, args=(sample_vol, sample_gr))

print('A:\t', result_b[0])
print('B:\t', result_b[1])
print('C:\t', result_b[2])
