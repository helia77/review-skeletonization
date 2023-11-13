# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:09:28 2023

@author: helioum
"""

import numpy as np
import frangi as frg
import matplotlib.pyplot as plt
import time
import metric as mt
import cthresholding as cp_th
import thresholding as th
import plot_curves as pltc
#%%
# test for 3D volume - KESM
volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

sample_vol = volume[200:300, 0:100, 0:100]
sample_gr = gr_truth[200:300, 0:100, 0:100]
#%%
# preprocess the terms in vesselness function based on two parameters only - later change the other parameter
start = time.time()
scale = [3, 4, 5, 6]
beta = 2 * (1*1)
c = 2 * 45 * 45
alpha = 2 * (0.01 * 0.01)
start = time.time()

#alpha_terms = [frg.terms_alpha(volume, s, beta, c) for s in scale]
#beta_terms  = [frg.terms_beta(volume, s, alpha, c) for s in scale]
c_terms     = [frg.terms_c(sample_vol, s, alpha, beta) for s in scale]
    
print(f"Calculating terms took {time.time() - start} seconds")
    
#%%
# find the best c value
c_range = [0.01, 1, 45, 100, 1000]
start = time.time()
threshs_c = [frg.process_c(C, c_terms, sample_gr) for C in c_range]
print(f"\ntook {time.time() - start} seconds")
c_range = np.array(c_range)
#%%
plt.figure(1)
plt.grid()
for i in range(c_range.size):
    if(i == 3):
        pltc.plot_pre_recall(threshs_c[i][0], sample_gr, marker= 'x', label='c='+str(c_range[i]))
    elif(i == 4):
        pltc.plot_pre_recall(threshs_c[i][0], sample_gr, marker= 'o', label='c='+str(c_range[i]))
    else:
        pltc.plot_pre_recall(threshs_c[i][0], sample_gr, label='c='+str(c_range[i]))
#%%
i = 35
def ppp(i):
    fig, ax = plt.subplots(2, 3)
    fig.suptitle('c param - Image ' + str(i))
    ax[0, 0].imshow(threshs_c[0][0][i], cmap='gray')
    ax[0, 0].set_title('c = 0.01')
    ax[0, 0].axis('off')
    
    ax[0, 1].imshow(threshs_c[1][0][i], cmap='gray')
    ax[0, 1].set_title('c = 1')
    ax[0, 1].axis('off')
    
    ax[0, 2].imshow(threshs_c[2][0][i], cmap='gray')
    ax[0, 2].set_title('c = 45')
    ax[0, 2].axis('off')
    
    ax[1, 0].imshow(threshs_c[3][0][i], cmap='gray')
    ax[1, 0].set_title('c = 100')
    ax[1, 0].axis('off')
    
    ax[1, 1].imshow(threshs_c[4][0][i], cmap='gray')
    ax[1, 1].set_title('c = 1000')
    ax[1, 1].axis('off')
    
    ax[1, 2].imshow(sample_gr[i], cmap='gray')
    ax[1, 2].set_title('raw volume')
    ax[1, 2].axis('off')
################################################################################
#%%
# find the best beta value
beta_range = [0.001, 0.01, 0.1, 1, 10000]
start = time.time()
threshs_beta = [frg.process_beta(B, beta_terms, gr_truth) for B in beta_range]
print(f"\ntook {time.time() - start} seconds")
beta_range = np.array(beta_range)
#%%
plt.figure(2)
plt.grid()
for i in range(beta_range.size):
    pltc.plot_pre_recall(threshs_beta[i][0], gr_truth, label='b='+str(beta_range[i]))
################################################################################
#%%
i = 40
def pp(i):
    fig, ax = plt.subplots(2, 3)
    fig.suptitle('beta param - Image ' + str(i))
    ax[0, 0].imshow(threshs_beta[0][0][i], cmap='gray')
    ax[0, 0].set_title('b = 0.001')
    ax[0, 0].axis('off')
    
    ax[0, 1].imshow(threshs_beta[1][0][i], cmap='gray')
    ax[0, 1].set_title('b = 0.01')
    ax[0, 1].axis('off')
    
    ax[0, 2].imshow(threshs_beta[2][0][i], cmap='gray')
    ax[0, 2].set_title('b = 0.1')
    ax[0, 2].axis('off')
    
    ax[1, 0].imshow(threshs_beta[3][0][i], cmap='gray')
    ax[1, 0].set_title('b = 1')
    ax[1, 0].axis('off')
    
    ax[1, 1].imshow(threshs_beta[4][0][i], cmap='gray')
    ax[1, 1].set_title('b = 1000000000')
    ax[1, 1].axis('off')
    
    ax[1, 2].imshow(volume[i], cmap='gray')
    ax[1, 2].set_title('raw volume')
    ax[1, 2].axis('off')
#%%
# find the best beta value
alpha_range = [0.00001, 0.01, 1, 5, 10]
start = time.time()
threshs_alpha = [frg.process_alpha(A, alpha_terms, gr_truth) for A in alpha_range]
print(f"\ntook {time.time() - start} seconds")
alpha_range = np.array(alpha_range)
#%%
plt.figure(3)
plt.grid()
for i in range(alpha_range.size):
    if(i == 1):
        pltc.plot_pre_recall(threshs_alpha[i][0], gr_truth,  label='a='+str(np.round(alpha_range[i], 4)))
    elif(i == 4):
        pltc.plot_pre_recall(threshs_alpha[i][0], gr_truth, marker= 'o', label='a='+str(np.round(alpha_range[i], 4)))
    else:
        pltc.plot_pre_recall(threshs_alpha[i][0], gr_truth, marker= 'x',label='a='+str(np.round(alpha_range[i], 4)))
#%%
i = 96
def p(i):
    fig, ax = plt.subplots(2, 3)
    fig.suptitle('alpha param - Image ' + str(i))
    ax[0, 0].imshow(threshs_alpha[0][0][i], cmap='gray')
    ax[0, 0].set_title('a = 0.00000001')
    ax[0, 0].axis('off')
    
    ax[0, 1].imshow(threshs_alpha[1][0][i], cmap='gray')
    ax[0, 1].set_title('a = 0.01')
    ax[0, 1].axis('off')
    
    ax[0, 2].imshow(threshs_alpha[2][0][i], cmap='gray')
    ax[0, 2].set_title('a = 1')
    ax[0, 2].axis('off')
    
    ax[1, 0].imshow(threshs_alpha[3][0][i], cmap='gray')
    ax[1, 0].set_title('a = 5')
    ax[1, 0].axis('off')
    
    ax[1, 1].imshow(threshs_alpha[4][0][i], cmap='gray')
    ax[1, 1].set_title('a = 10')
    ax[1, 1].axis('off')
    
    ax[1, 2].imshow(volume[i], cmap='gray')
    ax[1, 2].set_title('raw volume')
    ax[1, 2].axis('off')

