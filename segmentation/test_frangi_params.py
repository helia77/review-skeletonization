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

sample_vol = volume[200:400, 0:200, 100:200]
sample_gr = gr_truth[200:400, 0:200, 100:200]
#%%
# preprocess the terms in vesselness function based on two parameters only - later change the other parameter
start = time.time()
scale = [3, 4, 5, 6]
beta = 2 * (1*1)
c = 2 * 45 * 45
alpha = 2 * (1 * 1)
start = time.time()

alpha_terms = [frg.terms_alpha(sample_vol, s, beta, c) for s in scale]
beta_terms  = [frg.terms_beta(sample_vol, s, alpha, c) for s in scale]
c_terms     = [frg.terms_c_only(sample_vol, s) for s in scale]

print(f"Calculating terms took {time.time() - start} seconds")
    
#%%
# find the best c value
#c_range = [50]
c_range = np.linspace(1, 100, 5)
start = time.time()
threshs_c = [frg.process_c(C, c_terms, sample_gr) for C in c_range]
print(f"\ntook {time.time() - start} seconds")
c_range = np.array(c_range)

#%%
met_frani_jac = [mt.metric(sample_gr, threshs_c[i][0]).jaccard() for i in range(len(c_range))]
met_frani_dice = [mt.metric(sample_gr, threshs_c[i][0]).dice() for i in range(len(c_range))]
met_otsu_jac = [threshs_c[i][2].jaccard() for i in range(len(c_range))]
met_otsu_dice = [threshs_c[i][2].dice() for i in range(len(c_range))]
#%%
plt.figure(1)
plt.grid()
pltc.plot_pre_recall(threshs_c[4][0], sample_gr, marker='>', label='c='+str(c_range[4]))
pltc.plot_pre_recall(threshs_c[3][0], sample_gr, marker='x', label='c='+str(c_range[3]))
pltc.plot_pre_recall(threshs_c[2][0], sample_gr, marker=',', label='c='+str(c_range[2]))
pltc.plot_pre_recall(threshs_c[1][0], sample_gr, label='c='+str(c_range[1]))
pltc.plot_pre_recall(threshs_c[0][0], sample_gr, marker='.', label='c='+str(c_range[0]))
#%%
frangis = [threshs_c[i][0] for i in range(len(threshs_c))]
pltc.plot_auc_pr(frangis, sample_gr, c_range)
#%%
i = 50
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

ax[1, 1].imshow(sample_vol[i], cmap='gray')
ax[1, 1].set_title('c = 1000')
ax[1, 1].axis('off')

ax[1, 2].imshow(sample_gr[i], cmap='gray')
ax[1, 2].set_title('raw volume')
ax[1, 2].axis('off')
################################################################################
#%%
# find the best beta value
beta_range = [0.001, 0.01, 0.1, 1, 100]
start = time.time()
threshs_beta = [frg.process_beta(B, beta_terms, sample_gr) for B in beta_range]
print(f"\ntook {time.time() - start} seconds")
beta_range = np.array(beta_range)

#%%
met_frani_jac = [mt.metric(sample_gr, threshs_beta[i][0]).jaccard() for i in range(len(beta_range))]
met_frani_dice = [mt.metric(sample_gr, threshs_beta[i][0]).dice() for i in range(len(beta_range))]
met_otsu_jac = [threshs_beta[i][2].jaccard() for i in range(len(beta_range))]
met_otsu_dice = [threshs_beta[i][2].dice() for i in range(len(beta_range))]
#%%
plt.figure(2)
plt.grid()
for i in range(beta_range.size):
    if(i == 1):
        pltc.plot_pre_recall(threshs_beta[i][0], sample_gr,  label='b='+str(np.round(beta_range[i], 4)))
    elif(i == 4):
        pltc.plot_pre_recall(threshs_beta[i][0], sample_gr, marker= 'o', label='b='+str(np.round(beta_range[i], 4)))
    else:
        pltc.plot_pre_recall(threshs_beta[i][0], sample_gr, marker= 'x',label='b='+str(np.round(beta_range[i], 4)))
################################################################################
#%%
i = 0

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

ax[1, 2].imshow(sample_vol[i], cmap='gray')
ax[1, 2].set_title('raw volume')
ax[1, 2].axis('off')
#%%
# find the best beta value
alpha_range = [0.01, 1, 5, 10, 20]
start = time.time()
threshs_alpha = [frg.process_alpha(A, alpha_terms, sample_gr) for A in alpha_range]
print(f"\ntook {time.time() - start} seconds")
alpha_range = np.array(alpha_range)
#%%
met_frani_jac = [mt.metric(sample_gr, threshs_alpha[i][0]).jaccard() for i in range(len(alpha_range))]
met_frani_dice = [mt.metric(sample_gr, threshs_alpha[i][0]).dice() for i in range(len(alpha_range))]
met_otsu_jac = [threshs_alpha[i][2].jaccard() for i in range(len(alpha_range))]
met_otsu_dice = [threshs_alpha[i][2].dice() for i in range(len(alpha_range))]
#%%
plt.figure(3)
plt.grid()
for i in range(alpha_range.size):
    if(i == 1):
        pltc.plot_pre_recall(threshs_alpha[i][0], sample_gr,  label='a='+str(np.round(alpha_range[i], 4)))
    elif(i == 4):
        pltc.plot_pre_recall(threshs_alpha[i][0], sample_gr, marker= 'o', label='a='+str(np.round(alpha_range[i], 4)))
    else:
        pltc.plot_pre_recall(threshs_alpha[i][0], sample_gr, marker= 'x',label='a='+str(np.round(alpha_range[i], 4)))
#%%
i = 10
fig, ax = plt.subplots(2, 3)
fig.suptitle('alpha param - Image ' + str(i))
ax[0, 0].imshow(threshs_alpha[0][0][i], cmap='gray')
ax[0, 0].set_title('a = 0.01')
ax[0, 0].axis('off')

ax[0, 1].imshow(threshs_alpha[1][0][i], cmap='gray')
ax[0, 1].set_title('a = 1')
ax[0, 1].axis('off')

ax[0, 2].imshow(threshs_alpha[2][0][i], cmap='gray')
ax[0, 2].set_title('a = 5')
ax[0, 2].axis('off')

ax[1, 0].imshow(threshs_alpha[3][0][i], cmap='gray')
ax[1, 0].set_title('a = 10')
ax[1, 0].axis('off')

ax[1, 1].imshow(threshs_alpha[4][0][i], cmap='gray')
ax[1, 1].set_title('a = 100')
ax[1, 1].axis('off')

ax[1, 2].imshow(sample_vol[i], cmap='gray')
ax[1, 2].set_title('raw volume')
ax[1, 2].axis('off')
#%%
frangi_filtered = frg.frangi_3D(sample_vol, 5, 0.1, 50.5, 3, 7, 1, 'white')
#%%
met_frani = mt.metric(sample_gr, frangi_filtered)
otsu_output = cp_th.compute_otsu(frangi_filtered, 'black')
met_otsu = mt.metric(sample_gr, otsu_output)
