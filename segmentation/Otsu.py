# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:41:28 2023

@author: helioum

Applies Otsu's method.
In this code, image format as (z, y, x, RGB) has been used - if RGB values exist
"""
import sys
sys.path.append('..')
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
import manage_data as md

#%%
# returns a thresholded volume using Otsu's thresholding method
def compute_otsu(volume):
    # testing all thresholds from 0 to the maximum of the image
    threshold_range = range(np.max(volume)+1)
    criterias = []
    for th in threshold_range:
        """Otsu's method to compute criteria."""
        # create the thresholded volume
        thresholded_vol = np.zeros(volume.shape)
        thresholded_vol[volume >= th] = 1
    
        # compute weights
        nb_pixels = volume.size
        nb_pixels1 = np.count_nonzero(thresholded_vol)
        weight1 = nb_pixels1 / nb_pixels
        weight0 = 1 - weight1
    
        # if one of the classes is empty, eg all pixels are below or above the threshold, that threshold 
        # will not be considered in the search for the best threshold
        if weight1 == 0 or weight0 == 0:
            criterias.append(np.inf)
            continue
    
        # find all pixels belonging to each class
        val_pixels1 = volume[thresholded_vol == 1]
        val_pixels0 = volume[thresholded_vol == 0]
    
        # compute variance of these classes
        var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
        var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    
        criterias.append(weight0 * var0 + weight1 * var1)
        
    # best threshold is the one minimizing the Otsu criteria
    best_threshold = threshold_range[np.argmin(criterias)]
    thresh_volume = np.zeros(volume.shape, dtype=np.uint8)
    thresh_volume[volume > best_threshold] = 255
    
    return thresh_volume, best_threshold
    
#%%
path = 'C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/data/data'
crop_size = 200
# load data as a grayscale volume
volume = md.load_images(path, crop_size, True, crop_size, True)

#%%
start = time.time()
thresh_volume, ret = compute_otsu(volume)

cv2.imwrite('img15.jpg', thresh_volume[15])
cv2.imwrite('original.jpg', volume[15])

print('Execution time: --- %s seconds ---' % (time.time() - start))

#%%
path = 'C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/segmentation/data'
path2 = 'C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/segmentation/data2' 
md.save_slices(thresh_volume, path, 50)
md.save_slices(volume, path2, 50)
#%%
plt.hist(volume[4].ravel(),256)
plt.axvline(x=ret, color='r', linestyle='dashed', linewidth=2)
plt.show()

