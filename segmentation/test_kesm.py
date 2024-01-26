# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 12:37:35 2023

This code is written to plot the Precision-Recall curve of KESM data for Figure(1) - including global thresholding, 2D and 3D Otsu's points,
optimized 3-parameter Frangi, optimized 1-parameter Frangi, and published-parameters Frangi.

@author: helioum
"""

import numpy as np
import matplotlib.pyplot as plt
import frangi
import metric as mt
import thresholding as th

#%%
# Extract the test volume from raw data and corresponding ground truth
volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

sample_vol = volume[0:100, 300:400, 300:500]
sample_gr = gr_truth[0:100, 300:400, 300:500]

# The best scale range found for this data
scale_range = [0.9, 1.8, 2.7, 3.6]                              # s_min=0.9 s_max=3.6 step=0.9

#%%
# Global thresholding
th_range = np.delete(np.unique(sample_vol), 0)                  # first value is always zero => recall=1

global_PR   = np.zeros((2, th_range.size))                      # [0]: precision   [1]: recall
for i, t in enumerate(th_range):
    # global thresholding
    threshed_gl = (sample_vol <= t)
    met_gl = mt.metric(sample_gr, threshed_gl)

    global_PR[0][i] = met_gl.precision()
    global_PR[1][i] = met_gl.TPR()                                 # recall

#%%
# 2D  Otsu's point
_, best_thresh_2d = th.compute_otsu_img(sample_vol, background='white')
threshed_otsu2d = (sample_vol <= best_thresh_2d)
met_otsu2d = mt.metric(sample_gr, threshed_otsu2d)

otsu2d_PR = [met_otsu2d.precision(), met_otsu2d.TPR()]

#%%
# 3D  Otsu's point
_, best_thresh_3d = th.compute_otsu(sample_vol, background='white')
threshed_otsu3d = (sample_vol <= best_thresh_3d)
met_otsu3d = mt.metric(sample_gr, threshed_otsu3d)

otsu3d_PR = [met_otsu3d.precision(), met_otsu3d.TPR()]

#%%
# Optimized 3-parameter (alpha, beta, c) Frangi
alpha = 0.5
beta = 0.5
c = 105
opt_Frangi3p = frangi.upgrade_vesselness(sample_vol, alpha, beta, c, scale_range, 'white')

#%%
# Optimized 1-parameter (tau) Frangi
tau = 1.2
opt_BFrangi = frangi.beyond_frangi_filter(sample_vol, tau, scale_range, 'white')

#%%
# Frangi with published parameter values
half_norm = frangi.max_norm(sample_vol, scale_range) / 2
pub_Frangi = frangi.upgrade_vesselness(sample_vol, 0.5, 0.5, half_norm, scale_range, 'white')

#%%
# Save everything
np.save('Final npy Data/Figure 1/global_curve.npy', global_PR)
np.save('Final npy Data/Figure 1/otsu2d_point.npy', np.array(otsu2d_PR))
np.save('Final npy Data/Figure 1/otsu3d_point.npy', np.array(otsu3d_PR))
np.save('Final npy Data/Figure 1/opt_Frangi3p.npy', opt_Frangi3p)
np.save('Final npy Data/Figure 1/opt_bFrangi.npy', opt_BFrangi)
np.save('Final npy Data/Figure 1/pub_Frangi.npy', pub_Frangi)
 


