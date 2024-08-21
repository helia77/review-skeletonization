# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:24:55 2024

@author: helioum
"""
import numpy as np
import metric as mt
import thresholding as th
import matplotlib.pyplot as plt
import scipy.io as io

#%%
DATA = 'LSM'                     # change the dataset name
datasets = {"KESM": 'Data/KESM/sample_vol.npy', "LSM": 'Data/LSM/lsm_brain_8bit.npy', "Micro": 'Data/Micro-CT/sample_micro.npy'}
grounds = {"KESM": 'Data/KESM/sample_gr.npy', 
           "LSM": 'Data/LSM/lsm_brain_gr_truth.npy', 
           "Micro": 'Data/Micro-CT/sample_gr_micro.npy'}

volume = np.load(datasets[DATA])
gr_truth = np.load(grounds[DATA])

#%%
best = 0
best_params = [0, 0, 0, 0]
start = [0.06]
stop = [6]
threshs = [1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]
for i in start:
    for j in stop:
        print(i, ' ', j, end=' ')
        result = io.loadmat(f'./OOF/outputs/results_{i:.2f}_{j:.1f}.mat')['results']
        for thresh in threshs:
            new_result = result > thresh
            met = mt.metric(gr_truth, new_result)
            print(round(met.dice(), 3))
            if met.dice() > best:
                best = met.dice()
                best_params = [i, j, thresh, met.dice()]


print(i for i in best_params)
#%%
result = io.loadmat(f'./OOF/oof_lsm.mat')['results']
threshed = result > 2.4
met = mt.metric(gr_truth, result)
# print(round(met.f_score(), 4), end=', ')
plt.subplot(141).imshow(volume[25], 'gray')
plt.subplot(142).imshow(gr_truth[25], 'gray')
plt.subplot(143).imshow(result[25], 'gray')
plt.subplot(144).imshow(threshed[25], 'gray')
plt.show()

met = mt.metric(gr_truth, threshed)
print('\nDice:\t\t', round(met.dice(), 3))
print('Jaccard:\t', round(met.jaccard(), 3))
print('Precision:\t', round(met.precision(), 3))
print('Recall:\t\t', round(met.TPR(), 3))

#%%
best_params = [0, 0]
threshs = [-2, -2.1, -2.2, -2.3, -2.4]
for thresh in threshs:
    new_result = result < thresh
    met = mt.metric(gr_truth, new_result)
    print(round(met.f_score(), 3))
    if met.f_score() > best:
        best = met.f_score()
        best_params = [thresh, met.f_score()]

