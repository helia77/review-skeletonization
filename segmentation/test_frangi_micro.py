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

sample_vol = volume[0:300, 100:400, 200:400]
sample_gr = gr_truth[0:300, 100:400, 200:400]
#%%
start = time.time()
scale = [2, 2.5, 3, 3.5, 4]
beta = 2 * (1*1)
c = 2 * 25 * 25
alpha = 2 * (0.48 * 0.48)
start = time.time()

#alpha_terms = [frg.terms_alpha(sample_vol, s, beta, c, 'black') for s in scale]
#beta_terms  = [frg.terms_beta(sample_vol, s, alpha, c, 'black') for s in scale]
c_terms      = [frg.terms_c(sample_vol, s, alpha, beta, 'black') for s in scale]
c_temrs_only = [frg.terms_c_only(sample_vol, s, 'black') for s in scale]
print(f"Calculating terms took {time.time() - start} seconds")
#%%
c_range = [1, 25, 50, 100, 1000]
start = time.time()
threshs_c      = [frg.process_c(C, c_terms, sample_gr) for C in c_range]
threshs_c_only = [frg.process_c(C, c_temrs_only, sample_gr) for C in c_range]
print(f"\ntook {(time.time() - start)/60} mins")
c_range = np.array(c_range)
#%%
plt.figure(1)
plt.grid()
pltc.plot_pre_recall(threshs_c[0][0], sample_gr, marker= '>', label='1', color='#E49515', flag=True)
pltc.plot_pre_recall(threshs_c[1][0], sample_gr, label='25', color='#8503A9', flag=True)
pltc.plot_pre_recall(threshs_c[2][0], sample_gr, marker= '.', label='50', color='#0AA049', flag=True)
pltc.plot_pre_recall(threshs_c[3][0], sample_gr, marker= ',', label='100', color='#AF0731', flag=True)
pltc.plot_pre_recall(threshs_c[4][0], sample_gr, marker= 'o', label='1000', color='#1969AF', flag=False)
#%%
plt.figure(2)
plt.grid()
pltc.plot_pre_recall(threshs_c[1][0], sample_gr, marker= '.', label='with alpha/beta', color='#E49515', flag=True)
pltc.plot_pre_recall(threshs_c_only[1][0], sample_gr, label='without', color='#8503A9', flag=True)
# pltc.plot_pre_recall(threshs_c[2][0], sample_gr, marker= '.', label='25', color='#0AA049', flag=True)
# pltc.plot_pre_recall(threshs_c[3][0], sample_gr, marker= ',', label='100', color='#AF0731', flag=True)
#pltc.plot_pre_recall(threshs_c[4][0], sample_gr, marker= 'o', label='1000', color='#1969AF', flag=False)

#7.16, 7.23, 7.27, 7.28, 8.2, 8.4
#%%
# make the dictionary of metrics
dic_c = {}
dic_conly = {}
for i in range(c_range.size):
    metc = threshs_c[i][2]
    metc_only = threshs_c_only[i][2]
    dic_c[str(c_range[i])] = metc.jaccard(), metc.dice()
    dic_conly[str(c_range[i])] = metc_only.jaccard(), metc_only.dice()
normal_otsu, best = cp_th.compute_otsu(sample_vol, 'black')
met_otsu = mt.metric(sample_gr, normal_otsu)
dic_c['otsu'] = met_otsu.jaccard(), met_otsu.dice()        
dic_conly['otsu'] = met_otsu.jaccard(), met_otsu.dice() 
#%%
# find the best beta value
alpha_range = [0.48, 0.49, 0.5]
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
beta_range = [0.8,0.9,1, 1.1, 1.2]
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
i = 100
fig, ax = plt.subplots(2, 6)
fig.suptitle('c param - Image ' + str(i))
ax[0, 0].imshow(threshs_c[0][0][i], cmap='gray')
ax[0, 0].set_title('c = 1')
ax[0, 0].axis('off')

ax[0, 1].imshow(threshs_c[1][1][i], cmap='gray')
ax[0, 1].set_title('c = 25')
ax[0, 1].axis('off')

ax[0, 2].imshow(threshs_c[2][0][i], cmap='gray')
ax[0, 2].set_title('c = 50')
ax[0, 2].axis('off')

ax[0, 3].imshow(threshs_c[3][0][i], cmap='gray')
ax[0, 3].set_title('c = 100')
ax[0, 3].axis('off')

ax[0, 4].imshow(threshs_c[4][0][i], cmap='gray')
ax[0, 4].set_title('c = 1000')
ax[0, 4].axis('off')

ax[1, 0].imshow(threshs_c_only[0][0][i], cmap='gray')
ax[1, 0].set_title('c = 1')
ax[1, 0].axis('off')

ax[1, 1].imshow(threshs_c_only[1][1][i], cmap='gray')
ax[1, 1].set_title('c = 25')
ax[1, 1].axis('off')

ax[1, 2].imshow(threshs_c_only[2][0][i], cmap='gray')
ax[1, 2].set_title('c = 50')
ax[1, 2].axis('off')

ax[1, 3].imshow(threshs_c_only[3][0][i], cmap='gray')
ax[1, 3].set_title('c = 100')
ax[1, 3].axis('off')

ax[1, 4].imshow(threshs_c_only[4][0][i], cmap='gray')
ax[1, 4].set_title('c = 1000')
ax[1, 4].axis('off')

ax[0, 5].imshow(sample_vol[i], cmap='gray')
ax[0, 5].set_title('volume')
ax[0, 5].axis('off')

ax[1, 5].imshow(sample_gr[i], cmap='gray')
ax[1, 5].set_title('truth')
ax[1, 5].axis('off')