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
import metric as mt
import time

volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

# the cropped sizes can be changed
sample_vol = volume[0:100, 300:400, 300:500]
sample_gr = gr_truth[0:100, 300:400, 300:500]
#%%
def return_auc(vol, gr_truth, A, B, C):
    # calculating the vesselness function
    scale_range = [0.9, 1.8, 2.7, 3.6]
    predicted = frangi.upgrade_vesselness(vol, A, B, C, scale_range, 'white')
    
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
alpha = np.arange(0, 1, 0.025)    # size 40
c = np.arange(0, 160, 4)          # size 40
b = 0.5

A, C = np.meshgrid(alpha,  c) # grid of point
#np.save('AUC_alpha.npy', alpha)
#np.save('AUC_c.npy', c)

#%%
AUC = np.zeros(A.shape)
start = time.time()
for i in range(2):
    for j in range(2):
        print(i*5 + j + 1, end=' ')
        AUC[i, j] = return_auc(sample_vol, sample_gr, A[i, j], b, C[i, j])

#np.save('AUC_file.npy', AUC)
print('Took ', (time.time() - start)/60.0, ' sec')
#%%
im = plt.imshow(AUC,cmap='RdBu')
cset = plt.contour(AUC, np.arange(-1, 2, 0.2), linewidths=1.5, cmap='Set2')
plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
plt.colorbar(im) # adding the colobar on the right

round_alpha = [round(a, 3) for a in alpha]
round_c =     [round(C) for C in c]
plt.xticks(np.arange(len(alpha)), round_alpha, fontsize=5, rotation=90)
plt.yticks(np.arange(len(c)), round_c, fontsize=5)
plt.xlabel('$\\alpha$')
plt.ylabel('C')
idxmax = np.unravel_index(np.argmax(AUC), AUC.shape)
plt.scatter(idxmax[1], idxmax[0], marker='x', color='red', s=100, label='Max AUC')
plt.legend(loc='upper right')
plt.show()

