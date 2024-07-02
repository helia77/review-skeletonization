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
    # Set total number of bins in the histogram
    bins_num = 256
    # Get the image histogram
    hist, bin_edges = np.histogram(volume, bins=bins_num)
    # Get normalized histogram if it is required
    hist = np.divide(hist.ravel(), hist.max())
    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # Get the class means:  mu0(t) and mu1(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]   
    inter_class_variance = weight1[:-1] * weight2[1:] * ((mean1[:-1] - mean2[1:]) ** 2)
    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance) 
    best_thresh = bin_mids[:-1][index_of_max_val]
    return best_thresh
    
def compute_otsu(volume, background='black'):
    if np.all(np.unique(volume) == 0):
        return np.uint8(volume), 0
    best_threshold = compute_otsu_hist(volume)
    if background == 'black':
        threshed_otsu3d = (volume >= best_threshold)
    elif background == 'white':
        threshed_otsu3d = (volume < best_threshold)
    else:
        print('Wrong background input. Choose \'white\' or \'black\'.')
    return threshed_otsu3d, best_threshold

# Based on the code from Wikipedia
# returns a thresholded volume using Otsu's thresholding method
def otsu_2D(volume, background='black'):
    
    # testing all thresholds from 0 to the maximum of the image
    thresh_imgs = []
    for j in range(volume.shape[0]):
        image = volume[j]
        if image.dtype == np.float64:
            image = np.uint8(image*255)
        threshold_range = np.unique(image)
        criteria = []
        for i, th in enumerate(threshold_range):
            #(th, end=' ')
            """Otsu's method to compute criteria."""
            
            # create the thresholded volume
            thresholded_vol = np.zeros_like(image)
            if(background == 'black'):
                thresholded_vol[image >= th] = 1
            elif(background == 'white'):
                thresholded_vol[image <= th] = 1
            else:
                print('Wrong background input. Choose \'white\' or \'black\'.')
                return 0
        
            # compute weights
            nb_pixels = image.size
            nb_pixels1 = np.count_nonzero(thresholded_vol)
            weight1 = nb_pixels1 / nb_pixels
            weight0 = 1 - weight1
        
            # if one of the classes is empty, eg all pixels are below or above the threshold, that threshold 
            # will not be considered in the search for the best threshold
            if weight1 == 0 or weight0 == 0:
                criteria.append(np.inf)
                continue
        
            # find all pixels belonging to each class
            val_pixels1 = image[thresholded_vol == 1]
            val_pixels0 = image[thresholded_vol == 0]
        
            # compute variance of these classes
            var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
            var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
        
            criteria.append(weight0 * var0 + weight1 * var1)
            
        # best threshold is the one minimizing the Otsu criteria
        best_threshold = threshold_range[np.argmin(criteria)]
        thresh_img = np.zeros(image.shape, np.uint8)
        
        if(background == 'black'):
            thresh_img[image >= best_threshold] = 1
        elif(background == 'white'):
            thresh_img[image <= best_threshold] = 1
        thresh_imgs.append(thresh_img)
    thresh_images = np.stack(thresh_imgs, axis=0)
    return thresh_images

def compute_otsu_img(volume):
    thresh_img = []
    for i in range(volume.shape[0]):
        img = volume[i]
        threshed = np.where(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], 0, 1)
        thresh_img.append(threshed)
        
    thresh_images = np.stack(thresh_img, axis=0)
    return thresh_images

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