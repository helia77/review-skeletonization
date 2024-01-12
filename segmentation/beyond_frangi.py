# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:19:02 2023

@author: helioum
"""

import numpy as np
import numpy.linalg as lin
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import frangi
import metric as mt
import time


#%%
def beyond_frangi_filter(src, scale_range, tau, background):
    
    all_filters = []
    # for each scale
    for s in scale_range:
        # convolving image with Gaussian derivatives - including Hxx, Hxy, Hyy
        Hxx = np.zeros((src.shape[0], src.shape[1], src.shape[2]))
        Hyy = np.zeros((src.shape[0], src.shape[1], src.shape[2]))
        Hzz = np.zeros((src.shape[0], src.shape[1], src.shape[2]))
        Hxy = np.zeros((src.shape[0], src.shape[1], src.shape[2]))
        Hxz = np.zeros((src.shape[0], src.shape[1], src.shape[2]))
        Hzy = np.zeros((src.shape[0], src.shape[1], src.shape[2]))
        
        filters.gaussian_filter(src, (s, s, s), (0, 0, 2), Hxx)
        filters.gaussian_filter(src, (s, s, s), (0, 1, 1), Hxy)
        filters.gaussian_filter(src, (s, s, s), (0, 2, 0), Hyy)
        filters.gaussian_filter(src, (s, s, s), (2, 0, 0), Hzz)
        filters.gaussian_filter(src, (s, s, s), (1, 0, 1), Hxz)
        filters.gaussian_filter(src, (s, s, s), (1, 1, 0), Hzy)
    
        # correct for scaling - normalization
        s3 = s * s * s
        Hxx *= s3; Hyy *= s3; Hzz *= s3
        Hxy *= s3; Hxz *= s3; Hzy *= s3
        
        # reduce computation by computing vesselness only where needed
        B1 = - (Hxx + Hyy + Hzz)
        B2 = (Hxx * Hyy) + (Hxx * Hzz) + (Hyy * Hzz) - (Hxy * Hxy) - (Hxz * Hxz) - (Hzy * Hzy)
        B3 = (Hxx * Hzy * Hzy) + (Hxy * Hxy * Hzz) + (Hxz * Hyy * Hxz) - (Hxx * Hyy * Hzz) - (Hxy * Hzy * Hxz) - (Hxz * Hxy * Hzy)
        
        T = np.ones_like(B1, dtype=np.uint8)
        
        if background == 'black':
            T[B1 <= 0] = 0
            T[(B2 <= 0) & (B3 == 0)] = 0
            T[(B1 > 0) & (B2 > 0) & (B1*B2 < B3)] = 0
        else:
            T[B1 >= 0] = 0
            T[(B2 >= 0) & (B3 == 0)] = 0
            T[(B1 < 0) & (B2 < 0) & ((-B1)*(-B2) < (-B3))] = 0
        
        del B1, B2, B3
        Hxx *= T; Hyy *= T; Hzz *= T
        Hxy *= T; Hxz *= T; Hzy *= T
        
        H = np.zeros((src.shape[0], src.shape[1], src.shape[2], 3, 3))
        H[:, :, :, 2, 2] = Hxx;     H[:, :, :, 1, 1] = Hyy;     H[:, :, :, 0, 0] = Hzz;
        H[:, :, :, 1, 2] = Hxy;     H[:, :, :, 0, 2] = Hxz;     H[:, :, :, 0, 1] = Hzy;
        H[:, :, :, 2, 1] = Hxy;     H[:, :, :, 2, 0] = Hxz;     H[:, :, :, 1, 0] = Hzy;
        
        del Hxx, Hyy, Hzz, Hxy, Hxz, Hzy
        
        # we only be needing lambda2 and lambda3
        lambdas = lin.eigvalsh(H)
        
        idx = np.argwhere(T == 1)
        
        V0 = np.zeros_like(src)
        for arg in idx:
            # sort the eigenvalues
            i, j, k = arg
            lambdas[i, j, k] = sorted(lambdas[i, j, k], key=abs)
            
        max_l3 = np.max(lambdas[:, :, :, 2])  
        for arg in idx:
            i, j, k = arg
            _, l2, l3 = sorted(lambdas[i, j, k], key=abs)
            if background == 'black':
                l2 = -l2
                l3 = -l3
        
            # calculating lambda rho
            reg_term = tau * max_l3            # regularized term
            if l3 > 0 and l3 < reg_term:
                l_rho = reg_term
            elif l3 <= 0:
                l_rho = 0
            else:
                l_rho = l3
            
            # modified vesselness function
            if l2 >= (l_rho/2) and l_rho > 0:
                V0[i, j, k] = 1
            elif l2 <= (l_rho/2):
                V0[i, j, k] = (l2**2) * (l_rho - l2) * 27 / ((l2 + l_rho) ** 3)
            
        all_filters.append(V0)
    
    # pick the highest vesselness values
    response = frangi.highest_pixel(all_filters)
    return response
#%%
# test for 3D volume - KESM
volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

sample_vol = volume[300:350, 50:100, 150:200]
sample_gr = gr_truth[300:350, 50:100, 150:200]

background = 'white'
scale_range = np.arange(3, 6, 1)
tau = 0.5
#%%
start = time.time()
output1 = beyond_frangi_filter(sample_vol, scale_range, tau, 'white')
print('Beyond Frangi Took: ', time.time() - start, ' secs')
#%%
start = time.time()
output2 = frangi.frangi_3D(sample_vol, 3, 0.2, 40, 3, 6, 1, 'white')
print('Frangi Took: ', time.time() - start, ' secs')

#%%
met1 = mt.metric(sample_gr, output1)
print('Beyond:\ndice: ', met1.dice())
print('jaccard: ', met1.jaccard())

met2 = mt.metric(sample_gr, output2)
print('Frangi:\ndice: ', met2.dice())
print('jaccard: ', met2.jaccard())

fig, ax = plt.subplots(1, 3)
ax[0].imshow(sample_vol[10], cmap='gray')
ax[1].imshow(output1[10], cmap='gray')
ax[2].imshow(output2[10], cmap='gray')
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')

