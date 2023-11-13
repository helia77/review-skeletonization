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
#%%
volume = np.load('micro_raw_600x700x1010.npy')
gr_truth = np.load('micro_grtruth_600x700x1010.npy')

sample_vol = volume[0:100, 200:300, 200:300]
sample_gr = gr_truth[0:100, 200:300, 200:300]
#%%
scale = [3, 4, 5, 6]
c_terms = [frg.terms_c(sample_vol, s, back='black') for s in scale]
print('terms done.')
#%%
c_range = [0.01, 1, 45, 100, 1000]
start = time.time()
#threshs_c = [vesselness(C) for C in c_range]
threshs_c = [frg.process_c(C, c_terms, sample_gr, back='black') for C in c_range]
print(f"\ntook {(time.time() - start)/60} seconds")
c_range = np.array(c_range)

#%%
plt.figure(1)
plt.grid()
for i in range(c_range.size):
    pltc.plot_pre_recall(threshs_c[i][0], sample_gr, marker= 'x', label='c='+str(c_range[i]))
        
#%%
i = 35
fig, ax = plt.subplots(2, 3)
fig.suptitle('c param - Image ' + str(i))
ax[0, 0].imshow(threshs_c[0][1][i], cmap='gray')
ax[0, 0].set_title('c = 0.01')
ax[0, 0].axis('off')

ax[0, 1].imshow(threshs_c[1][1][i], cmap='gray')
ax[0, 1].set_title('c = 1')
ax[0, 1].axis('off')

ax[0, 2].imshow(threshs_c[2][1][i], cmap='gray')
ax[0, 2].set_title('c = 45')
ax[0, 2].axis('off')

ax[1, 0].imshow(threshs_c[3][1][i], cmap='gray')
ax[1, 0].set_title('c = 100')
ax[1, 0].axis('off')

ax[1, 1].imshow(threshs_c[4][1][i], cmap='gray')
ax[1, 1].set_title('c = 1000')
ax[1, 1].axis('off')

ax[1, 2].imshow(sample_gr[i], cmap='gray')
ax[1, 2].set_title('raw volume')
ax[1, 2].axis('off')