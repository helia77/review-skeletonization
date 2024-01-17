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

sample_vol = volume[300:350, 0:100, 100:200]
sample_gr = gr_truth[300:350, 0:100, 100:200]

#%%
start = time.time()
scale_range = np.arange(3, 6, 1)
tau = 0.4
output1 = frangi.beyond_frangi_filter(sample_vol, scale_range, tau, 'white')
print('Beyond Frangi Took: ', time.time() - start, ' secs')
#%%
start = time.time()
output2 = frangi.frangi_3D(sample_vol, 0.5, 0.5, 40, 3, 6, 1, 'white')
print('Frangi Took: ', time.time() - start, ' secs')

#%%
met1 = mt.metric(sample_gr, output1)
print('Beyond Frangi:\n\tdice: \t', met1.dice())
print('\tjaccard: ', met1.jaccard())

otsu_img, _ = cth.compute_otsu_img(output2, 'black')
otsu_vol, _ = cth.compute_otsu(output2, 'black')
met2 = mt.metric(sample_gr, otsu_img)
met3 = mt.metric(sample_gr, otsu_vol)
print('\nFrangi+Otsu (img):\n\tdice: \t', met2.dice())
print('\tjaccard: ', met2.jaccard())
print('\nFrangi+Otsu (vol):\n\tdice: \t', met3.dice())
print('\tjaccard: ', met3.jaccard())
#%%
i = 34
fig, ax = plt.subplots(3, 2)
ax[0, 0].imshow(sample_vol[i], cmap='gray')
ax[0, 1].imshow(sample_gr[i], cmap='gray')
ax[1, 0].imshow(output1[i], cmap='gray')
ax[1, 1].imshow(output2[i], cmap='gray')
ax[2, 0].imshow(otsu_img[i], cmap='gray')
ax[2, 1].imshow(otsu_vol[i], cmap='gray')

ax[0, 0].axis('off')
ax[0, 1].axis('off')
ax[1, 0].axis('off')
ax[1, 1].axis('off')
ax[2, 0].axis('off')
ax[2, 1].axis('off')

ax[0, 0].text(0.5, -.1, 'raw', fontsize=8, va='center', ha='center', transform=ax[0, 0].transAxes)
ax[0, 1].text(0.5, -.1, 'truth', fontsize=8, va='center', ha='center', transform=ax[0, 1].transAxes)
ax[1, 0].text(0.5, -.1, 'beyond Frangi', fontsize=8, va='center', ha='center', transform=ax[1, 0].transAxes)
ax[1, 1].text(0.5, -.1, 'Frangi', fontsize=8, va='center', ha='center', transform=ax[1, 1].transAxes)
ax[2, 0].text(.5, -.1, 'Frangi+Otsu (img)', fontsize=8, va='center', ha='center', transform=ax[2, 0].transAxes)
ax[2, 1].text(.5, -.1, 'Frangi+Otsu (vol)', fontsize=8, va='center', ha='center', transform=ax[2, 1].transAxes)