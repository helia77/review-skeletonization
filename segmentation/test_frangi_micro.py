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

sample_vol = volume[400:600, 100:300, 200:400]
sample_gr = gr_truth[400:600, 100:300, 200:400]
#%%
start = time.time()
scale = [2, 2.5, 3, 3.5, 4]
beta = 2 * (1*1)
c = 2 * 25 * 25
alpha = 2 * (0.48 * 0.48)
start = time.time()

#alpha_terms = [frg.terms_alpha(sample_vol, s, beta, c, 'black') for s in scale]
#beta_terms  = [frg.terms_beta(sample_vol, s, alpha, c, 'black') for s in scale]
#c_terms      = [frg.terms_c(sample_vol, s, alpha, beta, 'black') for s in scale]
c_temrs_only = [frg.terms_c_only(sample_vol, s, 'black') for s in scale]
print(f"Calculating terms took {time.time() - start} seconds")
#%%
#c_range = [1, 25, 50, 100, 1000]
c_range = [25.75]#np.linspace(1, 100, 5)
start = time.time()
#threshs_c      = [frg.process_c(C, c_terms, sample_gr) for C in c_range]
threshs_c_only = [frg.process_c(C, c_temrs_only, sample_gr) for C in c_range]
print(f"\ntook {(time.time() - start)/60} mins")
c_range = np.array(c_range)
#%%
cmet_frani_jac = [mt.metric(sample_gr, threshs_c_only[i][0]).jaccard() for i in range(len(c_range))]
cmet_frani_dice = [mt.metric(sample_gr, threshs_c_only[i][0]).dice() for i in range(len(c_range))]
cmet_otsu_jac = [threshs_c_only[i][2].jaccard() for i in range(len(c_range))]
cmet_otsu_dice = [threshs_c_only[i][2].dice() for i in range(len(c_range))]

#%%
threshed = (threshs_c_only[0][0] >= 105.905)
met = mt.metric(sample_gr, threshed)

precision = met.precision()
recall = met.TPR()

plt.figure(1)
plt.grid()
pltc.plot_pre_recall(threshs_c_only[0][0], sample_gr, marker= '.', label='c=25.75', color='#E49515', flag=True)
# pltc.plot_pre_recall(threshs_c_only[1][0], sample_gr, label='25', color='#8503A9', flag=True)
# pltc.plot_pre_recall(threshs_c_only[2][0], sample_gr, marker= '.', label='50', color='#0AA049', flag=True)
# pltc.plot_pre_recall(threshs_c_only[3][0], sample_gr, marker= ',', label='100', color='#AF0731', flag=True)
# pltc.plot_pre_recall(threshs_c_only[4][0], sample_gr, marker= 'o', label='1000', color='#1969AF', flag=False)
plt.plot(recall, precision, color='r', marker='o', label='Otsu+Frangi')
#%%
import manage_data  as md
md.numpy_to_nrrd(threshs_c_only[0][0], 'frangi_micro.nrrd')
md.numpy_to_nrrd(threshs_c_only[0][1], 'frangi_otsu_micro.nrrd')
#%%
i = 100
fig, ax = plt.subplots(2, 4)
fig.suptitle('c param - Image ' + str(i))
ax[0, 0].imshow(threshs_c_only[1][0][i], cmap='gray')
ax[0, 0].set_title('c = 1')
ax[0, 0].axis('off')

ax[0, 1].imshow(threshs_c_only[2][0][i], cmap='gray')
ax[0, 1].set_title('c = 1')
ax[0, 1].axis('off')

ax[0, 2].imshow(threshs_c_only[3][0][i], cmap='gray')
ax[0, 2].set_title('c = 45')
ax[0, 2].axis('off')

ax[0, 3].imshow(sample_vol[i], cmap='gray')
ax[0, 3].set_title('vol')
ax[0, 3].axis('off')

ax[1, 0].imshow(threshs_c_only[1][1][i], cmap='gray')
ax[1, 0].set_title('c = 1')
ax[1, 0].axis('off')

ax[1, 1].imshow(threshs_c_only[2][1][i], cmap='gray')
ax[1, 1].set_title('c = 38')
ax[1, 1].axis('off')

ax[1, 2].imshow(threshs_c_only[3][1][i], cmap='gray')
ax[1, 2].set_title('c = 1000')
ax[1, 2].axis('off')

ax[1, 3].imshow(sample_gr[i], cmap='gray')
ax[1, 3].set_title('ground')
ax[1, 3].axis('off')
#%%
# make the dictionary of metrics
# dic_c = {}
# dic_conly = {}
# for i in range(c_range.size):
#     metc = threshs_c[i][2]
#     metc_only = threshs_c_only[i][2]
#     dic_c[str(c_range[i])] = metc.jaccard(), metc.dice()
#     dic_conly[str(c_range[i])] = metc_only.jaccard(), metc_only.dice()
normal_otsu, best = cp_th.compute_otsu(sample_vol, 'black')
met = mt.metric(sample_gr, normal_otsu)
# dic_c['otsu'] = met_otsu.jaccard(), met_otsu.dice()        
# dic_conly['otsu'] = met_otsu.jaccard(), met_otsu.dice() 
#%%
# find the best beta value
alpha_range = [0.01, 1, 5, 10, 20]
start = time.time()
threshs_alpha = [frg.process_alpha(A, alpha_terms, sample_gr) for A in alpha_range]
print(f"\ntook {time.time() - start} seconds")
alpha_range = np.array(alpha_range)

#%%
amet_frani_jac = [mt.metric(sample_gr, threshs_alpha[i][0]).jaccard() for i in range(len(alpha_range))]
amet_frani_dice = [mt.metric(sample_gr, threshs_alpha[i][0]).dice() for i in range(len(alpha_range))]
amet_otsu_jac = [threshs_alpha[i][2].jaccard() for i in range(len(alpha_range))]
amet_otsu_dice = [threshs_alpha[i][2].dice() for i in range(len(alpha_range))]

#%%
plt.figure(2)
plt.grid()
pltc.plot_pre_recall(threshs_alpha[0][0], sample_gr, marker= '>', label='0.01', color='#E49515', flag=True)
pltc.plot_pre_recall(threshs_alpha[1][0], sample_gr, label='1', color='#8503A9', flag=True)
pltc.plot_pre_recall(threshs_alpha[2][0], sample_gr, marker= '.', label='5', color='#0AA049', flag=True)
pltc.plot_pre_recall(threshs_alpha[3][0], sample_gr, marker= '|', label='10', color='#AF0731', flag=True)
pltc.plot_pre_recall(threshs_alpha[4][0], sample_gr, marker= 'o', label='20', color='#1969AF', flag=False)
plt.title('alpha change')
#%%
i = 100
fig, ax = plt.subplots(2, 4)
fig.suptitle('c param - Image ' + str(i))
ax[0, 0].imshow(threshs_alpha[1][0][i], cmap='gray')
ax[0, 0].set_title('a = 0.01')
ax[0, 0].axis('off')

ax[0, 1].imshow(threshs_alpha[2][0][i], cmap='gray')
ax[0, 1].set_title('a = 1')
ax[0, 1].axis('off')

ax[0, 2].imshow(threshs_alpha[3][0][i], cmap='gray')
ax[0, 2].set_title('a = 5')
ax[0, 2].axis('off')

ax[0, 3].imshow(sample_vol[i], cmap='gray')
ax[0, 3].set_title('vol')
ax[0, 3].axis('off')

ax[1, 0].imshow(threshs_alpha[1][1][i], cmap='gray')
ax[1, 0].set_title('a = 0.01')
ax[1, 0].axis('off')

ax[1, 1].imshow(threshs_alpha[2][1][i], cmap='gray')
ax[1, 1].set_title('a = 1')
ax[1, 1].axis('off')

ax[1, 2].imshow(threshs_alpha[3][1][i], cmap='gray')
ax[1, 2].set_title('a = 5')
ax[1, 2].axis('off')

ax[1, 3].imshow(sample_gr[i], cmap='gray')
ax[1, 3].set_title('ground')
ax[1, 3].axis('off')
#%%    
# find the best beta value
beta_range = [0.001, 0.01, 0.1, 1, 100]
start = time.time()
threshs_beta = [frg.process_beta(B, beta_terms, sample_gr) for B in beta_range]
print(f"\ntook {time.time() - start} seconds")
beta_range = np.array(beta_range)

#%%
bmet_frani_jac = [mt.metric(sample_gr, threshs_beta[i][0]).jaccard() for i in range(len(beta_range))]
bmet_frani_dice = [mt.metric(sample_gr, threshs_beta[i][0]).dice() for i in range(len(beta_range))]
bmet_otsu_jac = [threshs_beta[i][2].jaccard() for i in range(len(beta_range))]
bmet_otsu_dice = [threshs_beta[i][2].dice() for i in range(len(beta_range))]

#%%
plt.figure(3)
plt.grid()
pltc.plot_pre_recall(threshs_beta[0][0], sample_gr, marker= '>', label='0.001', color='#E49515', flag=True)
pltc.plot_pre_recall(threshs_beta[1][0], sample_gr, label='0.01', color='#8503A9', flag=True)
pltc.plot_pre_recall(threshs_beta[2][0], sample_gr, marker= '.', label='0.1', color='#0AA049', flag=True)
pltc.plot_pre_recall(threshs_beta[3][0], sample_gr, marker= ',', label='1', color='#AF0731', flag=True)
pltc.plot_pre_recall(threshs_beta[4][0], sample_gr, marker= 'o', label='100', color='#1969AF', flag=False)
plt.title('beta change')
#%%
i = 100
fig, ax = plt.subplots(2, 4)
fig.suptitle('c param - Image ' + str(i))
ax[0, 0].imshow(threshs_beta[1][0][i], cmap='gray')
ax[0, 0].set_title('a = 0.01')
ax[0, 0].axis('off')

ax[0, 1].imshow(threshs_beta[2][0][i], cmap='gray')
ax[0, 1].set_title('a = 1')
ax[0, 1].axis('off')

ax[0, 2].imshow(threshs_beta[3][0][i], cmap='gray')
ax[0, 2].set_title('a = 5')
ax[0, 2].axis('off')

ax[0, 3].imshow(sample_vol[i], cmap='gray')
ax[0, 3].set_title('vol')
ax[0, 3].axis('off')

ax[1, 0].imshow(threshs_beta[1][1][i], cmap='gray')
ax[1, 0].set_title('a = 0.01')
ax[1, 0].axis('off')

ax[1, 1].imshow(threshs_beta[2][1][i], cmap='gray')
ax[1, 1].set_title('a = 1')
ax[1, 1].axis('off')

ax[1, 2].imshow(threshs_beta[3][1][i], cmap='gray')
ax[1, 2].set_title('a = 5')
ax[1, 2].axis('off')

ax[1, 3].imshow(sample_gr[i], cmap='gray')
ax[1, 3].set_title('ground')
ax[1, 3].axis('off')

#%%
frangi_filtered = frg.frangi_3D(sample_vol, 5, 0.1, 25.75, 2, 4.5, 0.5, 'black')
#%%
met_frani = mt.metric(sample_gr, frangi_filtered)
otsu_output, _ = cp_th.compute_otsu_img(frangi_filtered, 'black')
met_otsu = mt.metric(sample_gr, otsu_output)

#%%
i = 10
fig, ax = plt.subplots(2, 2)
fig.suptitle('whole volume' + str(i))
ax[0, 0].imshow(frangi_filtered[i], cmap='gray')
ax[0, 0].set_title('a = 0.01')
ax[0, 0].axis('off')

ax[0, 1].imshow(otsu_output[i], cmap='gray')
ax[0, 1].set_title('a = 1')
ax[0, 1].axis('off')

ax[1, 0].imshow(sample_vol[i], cmap='gray')
ax[1, 0].set_title('vol')
ax[1, 0].axis('off')

ax[1, 1].imshow(sample_gr[i], cmap='gray')
ax[1, 1].set_title('ground')
ax[1, 1].axis('off')