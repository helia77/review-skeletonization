# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:44:49 2024

This code is written to plot a contour colomap of the Area Under the Precision-Recall Curve (AUC-PR) for a range of alpha and c values
in Frangi filter. The beta parameter is 

@author: helioum
"""

import numpy as np
import frangi
import matplotlib.pyplot as plt
import time
import metric as mt
import cthresholding as cp_th

volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

# the cropped sizes can be changed
sample_vol = volume[300:350, 50:100, 150:200]
sample_gr = gr_truth[300:350, 50:100, 150:200]
#%%
def return_auc(vol, gr_truth, A, B, C):
    # calculating the vesselness function
    scale_range = np.arange(1.5, 4, 0.5)
    predicted = frangi.upgrade_vesselness(vol, A, B, C, scale_range, 'white')
    
    # plt.figure(1)
    # plt.imshow(predicted[10], cmap='gray')
    # plt.figure(2)
    # plt.imshow(sample_vol[10], cmap='gray')
    
    # calculating the area under the curve (AUC)
    if(np.unique(predicted).size > 1):
        th_range = np.delete(np.unique(predicted), 0)
    else:
        th_range = np.unique(predicted)
    precision   = np.zeros((th_range.size))
    recall      = np.zeros((th_range.size))
    for i, t in enumerate(th_range):
        # global thresholding
        threshed = (predicted >= t)
        met = mt.metric(gr_truth, threshed)
        precision[i] = met.precision()
        recall[i] = met.TPR()
    
    indices = np.argsort(recall)
    sorted_recall = recall[indices]
    sorted_precision = precision[indices]
    
    auc = np.trapz(sorted_precision, sorted_recall)
    print('AUC is: ', round(auc, 4))
    return auc

#%%
# give ranges for alpha and c 
alpha = np.arange(0.05, 0.55, 0.05)
c = np.arange(10, 110, 10)
b = 0.5

A, C = np.meshgrid(alpha,  c) # grid of point

#%%
AUC = np.zeros(A.shape)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
            AUC[i, j] = return_auc(sample_vol, sample_gr, A[i, j], b, C[i, j])

np.save('AUC_file.npy', AUC)
#%%
im = plt.imshow(AUC,cmap='RdBu')
cset = plt.contour(AUC, np.arange(-1, 1.5, 0.2), linewidths=1.5,cmap='Set2')
plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
plt.colorbar(im) # adding the colobar on the right

round_alpha = [round(a, 2) for a in alpha]
plt.xticks(np.arange(len(alpha)), round_alpha, fontsize=10)
plt.yticks(np.arange(len(c)), c)
plt.xlabel('$\\alpha$')
plt.ylabel('c')
idxmax = np.unravel_index(np.argmax(AUC), AUC.shape)
plt.scatter(idxmax[1], idxmax[0], marker='x', color='red', s=100, label='Max AUC')
plt.legend(loc='upper right')
plt.show()

