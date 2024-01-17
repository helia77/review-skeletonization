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
import cthresholding as cth

#%%
# test for 3D volume - KESM
volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

sample_vol = volume[300:350, 50:100, 100:200]
sample_gr = gr_truth[300:350, 50:100, 100:200]
scale_range = np.arange(3, 5, 1)
#%%
# start = time.time()
# tau = 0.4
# bfrangi_output = frangi.beyond_frangi_filter(sample_vol, scale_range, tau, 'white')
# print('Beyond Frangi Took: ', time.time() - start, ' secs')
# #%%
# start = time.time()
# frangi_output = frangi.frangi_3D(sample_vol, 0.5, 0.5, 40, 3, 6, 1, 'white')
# print('Frangi Took: ', time.time() - start, ' secs')

# #%%
# botsu_img, _ = cth.compute_otsu_img(bfrangi_output, 'black')
# botsu_vol, _ = cth.compute_otsu(bfrangi_output, 'black')
# bmet_img = mt.metric(sample_gr, botsu_img)
# bmet_vol = mt.metric(sample_gr, botsu_vol)
# print('\nBFrangi+Otsu (img):\n\tdice: \t', bmet_img.dice())
# print('\tjaccard: ', bmet_img.jaccard())
# print('\nBFrangi+Otsu (vol):\n\tdice: \t', bmet_vol.dice())
# print('\tjaccard: ', bmet_vol.jaccard())

# otsu_img, _ = cth.compute_otsu_img(frangi_output, 'black')
# otsu_vol, _ = cth.compute_otsu(frangi_output, 'black')
# met2 = mt.metric(sample_gr, otsu_img)
# met3 = mt.metric(sample_gr, otsu_vol)
# print('\nFrangi+Otsu (img):\n\tdice: \t', met2.dice())
# print('\tjaccard: ', met2.jaccard())
# print('\nFrangi+Otsu (vol):\n\tdice: \t', met3.dice())
# print('\tjaccard: ', met3.jaccard())
# #%%
# i = 10
# fig, ax = plt.subplots(3, 3)
# ax[0, 0].imshow(sample_vol[i], cmap='gray')
# ax[0, 2].imshow(sample_gr[i], cmap='gray')
# ax[1, 0].imshow(frangi_output[i], cmap='gray')
# ax[1, 1].imshow(otsu_img[i], cmap='gray')
# ax[1, 2].imshow(otsu_vol[i], cmap='gray')
# ax[2, 0].imshow(bfrangi_output[i], cmap='gray')
# ax[2, 1].imshow(botsu_img[i], cmap='gray')
# ax[2, 2].imshow(botsu_vol[i], cmap='gray')

# ax[0, 0].axis('off')
# ax[0, 1].axis('off')
# ax[0, 2].axis('off')
# ax[1, 0].axis('off')
# ax[1, 1].axis('off')
# ax[1, 2].axis('off')
# ax[2, 0].axis('off')
# ax[2, 1].axis('off')
# ax[2, 2].axis('off')

# ax[0, 0].text(.5, -0.2, 'raw', fontsize=8, va='center', ha='center', transform=ax[0, 0].transAxes)
# ax[0, 1].text(1.7, 0, 'truth', fontsize=8, va='center', ha='center', transform=ax[0, 1].transAxes)
# ax[1, 0].text(.5, -.15, 'Frangi', fontsize=8, va='center', ha='center', transform=ax[1, 0].transAxes)
# ax[1, 1].text(.5, -.15, 'Frangi+Otsu (img)', fontsize=8, va='center', ha='center', transform=ax[1, 1].transAxes)
# ax[1, 2].text(.5, -.15, 'Frangi+Otsu (vol)', fontsize=8, va='center', ha='center', transform=ax[1, 2].transAxes)
# ax[2, 0].text(.5, -.15, 'Beyond Frangi', fontsize=8, va='center', ha='center', transform=ax[2, 0].transAxes)
# ax[2, 1].text(.5, -.15, 'BFrangi+Otsu (img)', fontsize=8, va='center', ha='center', transform=ax[2, 1].transAxes)
# ax[2, 1].text(.5, -.15, 'BFrangi+Otsu (vol)', fontsize=8, va='center', ha='center', transform=ax[2, 2].transAxes)

#%%
# plot AUC and tau parameter using Beyond Frangi function
taus = np.arange(1, 1.6, 0.1)
all_bfrangi = []
for i in range(taus.size):
    print('tau= ', taus[i])
    result = frangi.beyond_frangi_filter(sample_vol, scale_range, taus[i], 'white')
    thresh, _ = cth.compute_otsu_img(result, 'black')
    met_img = mt.metric(sample_gr, thresh)
    print(met_img.dice(), ' ', met_img.jaccard())
    all_bfrangi.append(result)

#%%
all_aucs = []
for i, response in enumerate(all_bfrangi):
    # create the thresholds
    if(np.unique(response).size > 1):
        th_range = np.delete(np.unique(response), 0)
    else:
        th_range = np.unique(response)
    precision   = np.zeros((th_range.size))
    recall      = np.zeros((th_range.size))
    #print(th_range)
    for j, t in enumerate(th_range):
        # global thresholding
        threshed = (response >= t)
        met = mt.metric(sample_gr, threshed)
        
        precision[j] = met.precision()
        recall[j] = met.TPR()
        
    # plt.figure(i)
    # plt.plot(recall, precision)
    # plt.title('PR for tau = ' + str(0.1 * (i+1)))
    indices = np.argsort(recall)
    sorted_recall = recall[indices]
    sorted_precision = precision[indices]
    
    auc = np.trapz(sorted_precision, sorted_recall)
    print('AUC is: ', auc)
    all_aucs.append(auc)
    #print('yes', end=' ')
#%%
plt.figure()
plt.plot(taus, np.array(all_aucs), marker='.', label='AUC-PR')
plt.title('AUC-PR for tau parameter')
plt.xlabel('tau')
plt.ylabel('AUC-PR')
plt.legend(loc='lower left')
plt.plot()