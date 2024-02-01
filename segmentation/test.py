# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:19:02 2023

@author: helioum
"""
import numpy as np
import matplotlib.pyplot as plt
import frangi
import metric as mt
import time
import thresholding as th
import plot_curves as pltc

#%%
# test for 3D volume - KESM
volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

sample_vol = volume[0:100, 300:400, 300:500]
sample_gr = gr_truth[0:100, 300:400, 300:500]
scale_range = [0.9, 1.8, 2.7, 3.6]
beta_range = np.linspace(0, 3, 50)
#np.save('Final npy Data/Figure 2/KESM/beta_range.npy', beta_range)
#%%
all_b_frangi = np.load('Final npy Data/Figure 2/KESM/all_frangi_brange.npy')
all_b_thresh = np.load('Final npy Data/Figure 2/KESM/all_threshed_brange.npy') #?
AUC = np.load('Final npy Data/Figure 2/KESM/AUC_betas.npy')
beta_range = np.load('Final npy Data/Figure 2/KESM/beta_range.npy')
    
#%%
plt.figure(1)
plt.plot(beta_range, AUC)
plt.xlabel('$\\beta$')
plt.ylabel('AUC')
idxmax = np.squeeze(np.where(AUC == max(AUC)))
plt.scatter(beta_range[idxmax], AUC[idxmax], marker='o', color='red', s=40)
plt.plot()

print('highest AUC occurs at $\\beta$ ', beta_range[idxmax])