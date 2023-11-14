# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:41:28 2023

@author: helioum

Applies Otsu's method.
In this code, image format as (z, y, x, RGB) has been used - if RGB values exist
"""
import numpy as np
import cv2
import cupy as cp

#%%
# Based on the code from Wikipedia
# returns a thresholded volume using Otsu's thresholding method
def compute_otsu_hist(volume):
    # Compute histogram using CuPy
    # Set total number of bins in the histogram
    bins_num = 256
     
    # Get the image histogram
    hist, bin_edges = cp.histogram(volume, bins=bins_num)
     
    # Get normalized histogram if it is required
    hist = cp.divide(hist.ravel(), hist.max())
     
    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
     
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = cp.cumsum(hist)
    weight2 = cp.cumsum(hist[::-1])[::-1]
     
    # Get the class means mu0(t)
    mean1 = cp.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (cp.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
     
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
     
    # Maximize the inter_class_variance function val
    index_of_max_val = cp.argmax(inter_class_variance)
     
    best_thresh = bin_mids[:-1][index_of_max_val]
    return best_thresh

def process_volume(volume, best_thresh, backgr):
    thresholded_vol = cp.zeros(volume.shape)
    if backgr == 'black':
        thresholded_vol[volume >= best_thresh] = 1
    elif backgr == 'white':
        thresholded_vol[volume <= best_thresh] = 1
    else:
        print('Wrong background input. Choose \'white\' or \'black\'.')
        return 0
    #thresholded_vol[volume >= best_thresh] = 1
    return thresholded_vol
    
def compute_otsu(volume, background):
    volume = cp.asarray(volume)
    best_threshold = compute_otsu_hist(volume)
    thresholded_volume = process_volume(volume, best_threshold, backgr=background)
    return cp.asnumpy(thresholded_volume), cp.asnumpy(best_threshold)

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