# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:27:29 2024

cost function and optimization of Beyond Frangi method
NOTE: the cost function for Beyond Frangi is written for KESM data only (background='white')

@author: helioum
"""

import numpy as np
import numpy.linalg as lin
from scipy.ndimage import filters
import frangi
from scipy.optimize import minimize
import matplotlib.pyplot as plt

volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

# the cropped sizes can be changed
sample_vol = volume[300:350, 50:100, 150:200]
sample_gr = gr_truth[300:350, 50:100, 150:200]

def cost_bfrangi(tau, sample_vol, sample_gr):
    print('tau=', tau)
    
    scale_range = np.arange(1.5, 4, 0.5)
    
    # pick the pixels with the highest vesselness value
    predicted = frangi.beyond_frangi_filter(sample_vol, scale_range, tau, 'white')
    
    #predicted, best_thresh = cp_th.compute_otsu_img(response,  background='black')
    TP = np.logical_and(sample_gr, predicted).sum()
    FP = np.logical_and(np.logical_not(sample_gr), predicted).sum()
    FN = np.logical_and(sample_gr, np.logical_not(predicted)).sum()     
    
    dice = (2*TP) / float(2*TP + FP + FN)
    print('dice= ', dice, '\n')
    return 1 - dice


''' Beyond Frangi '''

# Set initial parameter and bounds for Beyond Frangi optimization
initial_tau = [0.5]
tau_bound = [(0.4, 5)]

# Run the optimization for Beyond Frangi
result_b = minimize(cost_bfrangi, initial_tau, arg=(sample_vol, sample_gr), method='Powell', bounds=tau_bound)
optimized_tau = result_b.x
print('Best tau:\t', optimized_tau[0])
