# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:41:28 2023

@author: helioum

Applies Otsu's method.
In this code, image format as (z, y, x, RGB) has been used - if RGB values exist
"""
import numpy as np
import cv2

def compute_otsu_hist(volume):
    # Compute histogram using CuPy
    # Set total number of bins in the histogram
    bins_num = 256
     
    # Get the image histogram
    hist, bin_edges = np.histogram(volume, bins=bins_num)
     
    # Get normalized histogram if it is required
    hist = np.divide(hist.ravel(), hist.max())
     
    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
     
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
     
    # Get the class means:  mu0(t) and mu1(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
     
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
     
    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)
     
    best_thresh = bin_mids[:-1][index_of_max_val]
    return best_thresh

def process_volume(volume, best_thresh, backgr):
    thresholded_vol = np.zeros(volume.shape)
    if backgr == 'black':
        thresholded_vol[volume >= best_thresh] = 1
    elif backgr == 'white':
        thresholded_vol[volume <= best_thresh] = 1
    else:
        print('Wrong background input. Choose \'white\' or \'black\'.')
        return 0

    return thresholded_vol
    
def compute_otsu(volume, background):
    best_threshold = compute_otsu_hist(volume)
    thresholded_volume = process_volume(volume, best_threshold, backgr=background)
    return thresholded_volume, best_threshold

# Based on the code from Wikipedia
# returns a thresholded volume using Otsu's thresholding method
def otsu_img(volume, background='black'):
    # testing all thresholds from 0 to the maximum of the image
    threshold_range = range(np.min(volume), np.max(volume)+1, 1)
    criteria = []
    for i, th in enumerate(threshold_range):
        #(th, end=' ')
        """Otsu's method to compute criteria."""
        
        # create the thresholded volume
        thresholded_vol = np.zeros(volume.shape)
        if(background == 'black'):
            thresholded_vol[volume >= th] = 1
        elif(background == 'white'):
            thresholded_vol[volume <= th] = 1
        else:
            print('Wrong background input. Choose \'white\' or \'black\'.')
            return 0
    
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
    
    if(background == 'black'):
        thresh_volume[volume > best_threshold] = 1
    elif(background == 'white'):
        thresh_volume[volume < best_threshold] = 1
    
    return thresh_volume, best_threshold
    
def compute_otsu_img(volume, background):
    
    thresh_img = []
    mean = 0
    for i in range(volume.shape[0]):
        img = volume[i]
        thresh, test = otsu_img(img, background=background)
        mean += test
        thresh_img.append(thresh)
    thresh_images = np.stack(thresh_img, axis=0)
    mean /= volume.shape[0]
    return thresh_images, mean

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