# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:41:28 2023

@author: helioum

Applies Otsu's method.
In this code, image format as (z, y, x, RGB) has been used - if RGB values exist
"""
import numpy as np
import cv2
#%%
# Based on the code from Wikipedia
# returns a thresholded volume using Otsu's thresholding method
def compute_otsu(volume, pos_label):
    # testing all thresholds from 0 to the maximum of the image
    threshold_range = range(np.max(volume)+1)
    criteria = []
    for i, th in enumerate(threshold_range):
        """Otsu's method to compute criteria."""
        #if (i%10 == 0): print(i , end=' ') 
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
            criteria.append(np.inf)
            continue
    
        # find all pixels belonging to each class
        val_pixels1 = volume[thresholded_vol == 1]
        val_pixels0 = volume[thresholded_vol == 0]
    
        # compute variance of these classes
        var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
        var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    
        criteria.append(weight0 * var0 + weight1 * var1)
        
    # best threshold is the one minimizing the Otsu criteria
    best_threshold = threshold_range[np.argmin(criteria)]
    thresh_volume = np.zeros(volume.shape, dtype=np.uint8)
    thresh_volume[volume > best_threshold] = pos_label
    
    return thresh_volume, best_threshold
    

def adaptive_mean(img_list, w_size, const):
    thresh_img = []
    for img in img_list:
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, w_size, const)
        thresh_img.append(thresh)

    return np.stack(thresh_img, axis=0)

def adaptive_gaussian(img_list, w_size, const):
    thresh_img = []
    for img in img_list:
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, w_size, const)
        thresh_img.append(thresh)

    return np.stack(thresh_img, axis=0)