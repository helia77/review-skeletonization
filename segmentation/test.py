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
half_norm = frangi.max_norm(sample_vol, scale_range) / 2

#%%
start = time.time()
predicted1 = frangi.upgrade_vesselness(sample_vol, 0.1, 0.5, 50, scale_range, background='white')
predicted2 = frangi.upgrade_vesselness(sample_vol, 0.1, 0.5, 75, scale_range, background='white')
predicted3 = frangi.upgrade_vesselness(sample_vol, 0.1, 0.5, 100, scale_range, background='white')
predicted4 = frangi.upgrade_vesselness(sample_vol, 0.1, 0.5, 125, scale_range, background='white')
print('Took ', time.time() - start, ' secs')
#%%
start = time.time()
threshed1, best_thresh1 = th.compute_otsu_img(predicted1, 'black')
threshed2, best_thresh2 = th.compute_otsu_img(predicted2, 'black')
threshed3, best_thresh3 = th.compute_otsu_img(predicted3, 'black')
threshed4, best_thresh4 = th.compute_otsu_img(predicted4, 'black')
print('Took ', time.time() - start, ' secs')

#%%
i = 35
fig, ax = plt.subplots(5, 2)

ax[0, 0].imshow(predicted1[i], cmap='gray')
ax[0, 1].imshow(threshed1[i], cmap='gray')
ax[1, 0].imshow(predicted2[i], cmap='gray')
ax[1, 1].imshow(threshed2[i], cmap='gray')
ax[2, 0].imshow(predicted3[i], cmap='gray')
ax[2, 1].imshow(threshed3[i], cmap='gray')
ax[3, 0].imshow(predicted4[i], cmap='gray')
ax[3, 1].imshow(threshed4[i], cmap='gray')
ax[4, 0].imshow(sample_vol[i], cmap='gray')
ax[4, 1].imshow(sample_gr[i], cmap='gray')

for i in range(5):
    for j in range(2):
        ax[i, j].axis('off')

#%%
# plot the precision-recall curves for the results of each tau
plt.figure(3)
plt.grid()
pltc.plot_pre_recall(predicted1, sample_gr, color='b', label='c=50', end=True)
pltc.plot_pre_recall(predicted2, sample_gr, color='r', label='c=75', end=True)
pltc.plot_pre_recall(predicted3, sample_gr, color='g', label='c=100', end=True)
pltc.plot_pre_recall(predicted4, sample_gr, color='m', label='c=125', end=True)
plt.legend(loc='lower left')
#%%
for pred in [predicted1, predicted2, predicted3, predicted4]:
    met = mt.metric(sample_gr, pred)
    print('Dice:', met.dice(), '\tJaccard:', met.jaccard(), '\tAUC:', met.return_auc())

