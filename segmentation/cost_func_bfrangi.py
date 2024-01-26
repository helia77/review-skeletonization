# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:27:29 2024

cost function and optimization of Beyond Frangi method
NOTE: the cost function for Beyond Frangi is written for KESM data only (background='white')

@author: helioum
"""

import numpy as np
import frangi
from scipy.optimize import minimize
import metric as mt
import thresholding as th

#%%
volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

# the cropped sizes can be changed
sample_vol = volume[0:100, 300:400, 300:500]
sample_gr = gr_truth[0:100, 300:400, 300:500]

def cost_bfrangi(tau, sample_vol, sample_gr):
    
    scale_range = [0.9, 1.8, 2.7, 3.6]
    
    # pick the pixels with the highest vesselness value
    predicted = frangi.beyond_frangi_filter(sample_vol, tau, scale_range, 'white')
    thresh, _ = th.compute_otsu_img(predicted, 'black')
    met = mt.metric(sample_gr, thresh)
    auc = met.return_auc()
    print('tau:', tau, '\tAUC:', round(auc, 4))
    
    return 1 - auc


''' Beyond Frangi '''

# Set initial parameter and bounds for Beyond Frangi optimization
initial_tau = 0.5
tau_bound = [(0.1, 5)]

# Run the optimization for Beyond Frangi
result_b = minimize(cost_bfrangi, initial_tau, args=(sample_vol, sample_gr), method='Powell', bounds=tau_bound)
np.save(result_b, 'result_cost.npy')
optimized_tau = result_b.x
print('Best tau:\t', optimized_tau[0])
