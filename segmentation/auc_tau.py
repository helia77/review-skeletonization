# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:03:37 2024

@author: helioum
"""

import numpy as np
import frangi
import metric as mt
import matplotlib.pyplot as plt
import time
#%%
volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

# the cropped sizes can be changed
sample_vol = volume[50:100, 350:400, 450:500]
sample_gr = gr_truth[50:100, 350:400, 450:500]
scale_range = [0.9, 1.8]#, 2.7, 3.6]

#%%
taus = np.arange(0, 10, 0.1)
AUC = np.zeros_like(taus)
start = time.time()

for i, tau in enumerate(taus):
    print(i+1, end='\t')
    predicted = frangi.beyond_frangi_filter(sample_vol, tau, scale_range, 'white')
    met = mt.metric(sample_gr, predicted)
    auc = met.return_auc()

    print('AUC:', auc)
    AUC[i] = auc

np.save('auc_tau.npy', AUC)
print('Took ', (time.time() - start)/60.0, ' secs')

#%%
auc_tau = np.load('auc_tau.npy')
plt.figure(1)
idxmax_auc = np.argwhere(auc_tau == max(auc_tau))[0][0]
plt.plot(taus, auc_tau, color='g')
plt.scatter(taus[idxmax_auc], auc_tau[idxmax_auc], marker='x', c='r', s=30)
print('Max AUC at tau:', taus[idxmax_auc], ' , AUC:', auc_tau[idxmax_auc])

