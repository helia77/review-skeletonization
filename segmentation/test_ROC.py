# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:13:19 2023

@author: helioum
"""

from matplotlib import pyplot as plt
import numpy as np
import thresholding as th
import manage_data as md
import metric as mt
import time
import os

#%%
path = 'load_volume.npy'
volume = np.load(path)

# load the true volume
true_path = 'C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/data/Artem\'s data/micro_200x200x200.nrrd'
vol_true = md.nrrd_to_numpy(true_path)                           # convert nrrd file to numpy array

#%%
# plot the ROC curve
TPR = []
FPR = []
wtf = []
# iterate through all thresholds
start = time.time()
for i in range(40, np.min(volume), -1):
    thresholded = np.zeros(volume.shape)
    thresholded = (volume >= i).astype(int)
    met = mt.metric(vol_true, thresholded, 1)
    if(met.sensitivity() == 1):
        wtf.append(thresholded)
        print(str(i) + ' TP: ' + str(met.TP) + ' FN:' + str(met.FN)+ ' FP:' + str(met.FP) + ' TN:' + str(met.TN))
        print(str(met.fall_out()))
    TPR.append(met.sensitivity())
    FPR.append(met.fall_out())

print('\nROC curve calculation: Done\nExecution time: --- %s seconds ---' % (time.time() - start))

#%%
flat = volume.flatten()

plt.hist(flat, bins=50, range=(0, 255), color='blue', alpha=0.7)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.title('Intensity Histogram')
plt.show()