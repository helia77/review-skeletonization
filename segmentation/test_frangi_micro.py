# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:53:52 2023

@author: helioum
"""

import numpy as np
import math
import time
import matplotlib.pyplot as plt
import frangi as frg
import plot_curves as pltc
import metric as mt
import cthresholding as cp_th
#%%
volume = np.load('micro_raw_600x700x1010.npy')
gr_truth = np.load('micro_grtruth_600x700x1010.npy')

sample_vol = volume[100:200, 100:200, 100:200]
sample_gr = gr_truth[100:200, 100:200, 100:200]
#%%
start = time.time()
scale = [2, 3, 4, 5]
beta = 2 * (1*1)
c = 2 * 50 * 50
alpha = 2 * (0.51 * 0.51)
start = time.time()

#alpha_terms = [frg.terms_alpha(sample_vol, s, beta, c, 'black') for s in scale]
#beta_terms  = [frg.terms_beta(sample_vol, s, alpha, c, 'black') for s in scale]
c_terms     = [frg.terms_c(sample_vol, s, alpha, beta, 'black') for s in scale]

print(f"Calculating terms took {time.time() - start} seconds")
#%%
c_range = [30, 35, 40, 45, 50]
start = time.time()
threshs_c = [frg.process_c(C, c_terms, sample_gr) for C in c_range]
print(f"\ntook {(time.time() - start)/60} mins")
c_range = np.array(c_range)
#%%
plt.figure(1)
plt.grid()
pltc.plot_pre_recall(threshs_c[0][0], sample_gr, marker= '>', label='0.1', color='#E49515', flag=True)
pltc.plot_pre_recall(threshs_c[1][0], sample_gr, label='1', color='#8503A9', flag=True)
pltc.plot_pre_recall(threshs_c[2][0], sample_gr, marker= '.', label='50', color='#0AA049', flag=True)
pltc.plot_pre_recall(threshs_c[3][0], sample_gr, marker= ',', label='100', color='#AF0731', flag=True)
pltc.plot_pre_recall(threshs_c[4][0], sample_gr, marker= 'o', label='1000', color='#1969AF', flag=False)


#%%
# make the dictionary of metrics
dic = {}
max_jac = 0
ijac = 0
max_dice = 0
idice = 0
for i in range(c_range.size):
    met = threshs_c[i][2]
    dic[str(c_range[i])] = met.jaccard(), met.dice()
    if (met.jaccard() >= max_jac):
        max_jac = met.jaccard()
        jac_ix = c_range[i]
    if (met.dice() >= max_dice):
        max_dice = met.dice()
        dice_ix = c_range[i]
normal_otsu, best = cp_th.compute_otsu(sample_vol, 'black')
met_otsu = mt.metric(sample_gr, normal_otsu)
dic['otsu'] = met_otsu.jaccard(), met_otsu.dice()        

#%%
# find the best beta value
alpha_range = [0.5, 0.51, 0.52, 0.53]
start = time.time()
threshs_alpha = [frg.process_alpha(A, alpha_terms, sample_gr) for A in alpha_range]
print(f"\ntook {time.time() - start} seconds")
alpha_range = np.array(alpha_range)

#%%
dic_alpha = {}
max_jac = 0
jac_ix = 0
max_dice = 0
dice_ix = 0
for i in range(alpha_range.size):
    met = threshs_alpha[i][2]
    dic_alpha[str(alpha_range[i])] = met.jaccard(), met.dice()
    if (met.jaccard() >= max_jac):
        max_jac = met.jaccard()
        jac_ix = alpha_range[i]
    if (met.dice() >= max_dice):
        max_dice = met.dice()
        dice_ix = alpha_range[i]
normal_otsu, best = cp_th.compute_otsu(sample_vol, 'black')
met_otsu = mt.metric(sample_gr, normal_otsu)
dic_alpha['otsu'] = met_otsu.jaccard(), met_otsu.dice()

#%%    
# find the best beta value
beta_range = [1, 1.1, 1.2]
start = time.time()
threshs_beta = [frg.process_beta(B, beta_terms, sample_gr) for B in beta_range]
print(f"\ntook {time.time() - start} seconds")
beta_range = np.array(beta_range)

#%%
dic_beta = {}
max_jac = 0
jac_ix = 0
max_dice = 0
dice_ix = 0
for i in range(beta_range.size):
    met = threshs_beta[i][2]
    dic_beta[str(beta_range[i])] = met.jaccard(), met.dice()
    if (met.jaccard() >= max_jac):
        max_jac = met.jaccard()
        jac_ix = beta_range[i]
    if (met.dice() >= max_dice):
        max_dice = met.dice()
        dice_ix = beta_range[i]
normal_otsu, best = cp_th.compute_otsu(sample_vol, 'black')
met_otsu = mt.metric(sample_gr, normal_otsu)
dic_beta['otsu'] = met_otsu.jaccard(), met_otsu.dice()    
#%%
i = 35
fig, ax = plt.subplots(2, 3)
fig.suptitle('c param - Image ' + str(i))
ax[0, 0].imshow(sample_gr[i], cmap='gray')
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

ax[1, 1].imshow(normal_otsu[i], cmap='gray')
ax[1, 1].set_title('c = 1000')
ax[1, 1].axis('off')

ax[1, 2].imshow(sample_vol[i], cmap='gray')
ax[1, 2].set_title('raw volume')
ax[1, 2].axis('off')