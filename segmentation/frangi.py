# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:08:40 2023

@author: helioum
"""

import numpy as np
import math
import scipy.signal as sc
import time
import numpy.linalg as lin
from scipy.ndimage import filters
#%%
# filter the given image based on given scale and returns output image with vesselness values
def vesselness_2D(src, scale, beta, c):

    s2 = scale * scale
    
    # convolving image with Gaussian derivatives - including Dxx, Dxy, Dyy
    D = np.zeros(src.shape[0], src.shape[1], 3)
    filters.gaussian_filter(src, (scale, scale), (0, 2), D[:, :, 0])            # Dxx
    filters.gaussian_filter(src, (scale, scale), (1, 1), D[:, :, 1])            # Dxy
    filters.gaussian_filter(src, (scale, scale), (2, 0), D[:, :, 2])            # Dyy
    D *= s2                                                                     # for normalization

    # eigenvalue calculations from Dxx, Dxy, Dyy and compute the vesselnes function
    output = np.zeros((src.shape))
    
    for x in range(src.shape[0]):
        for y in range(src.shape[1]):
            # ----------- using conventional way (faster) ------------- #
            dxx = D[x, y, 0]
            dxy = D[x, y, 1]
            dyy = D[x, y, 2]
            
            # calculate eigenvalues
            tmp = dxx + dyy
            tmp2 = math.sqrt((dxx - dyy)**2 + 4*(dxy**2))
            
            lambda1 = 0.5 * (tmp + tmp2)
            lambda2 = 0.5 * (tmp - tmp2)
            
            # making sure they're sorted based on absolute values
            if (abs(lambda1) < abs(lambda2)):
                lambda2, lambda1 = lambda1, lambda2
            
            # ----------- using numpy linalg eigendecomposition (slower) ------------- #
            # hess_mat = np.zeros((2,2))
            # hess_mat[0, 0] = dyy
            # hess_mat[0, 1] = dxy
            # hess_mat[1, 0] = dxy
            # hess_mat[1, 1] = dxx
            
            # lam1, lam2 = np.linalg.eigvalsh(hess_mat)
            # if (abs(lam1) > abs(lam2)):
            #     lambda1, lambda2 = lam2, lam1
           
            if(lambda2==0):
                lambda2 = math.nextafter(0, 1)
            
            Rb = lambda1/lambda2
            S2 = (lambda1**2) + (lambda2**2)
            term1 = math.exp(-(Rb**2) / beta)
            term2 = math.exp(-S2 / c)
            
            output[x, y] = term1 * (1.0 - term2) if (lambda2 < 0) else 0
    return output

def frangi_2D(src, B, C, start, stop, step):
    # stores all the filtered images based on different scales
    all_filters = []
    
    beta = 2 * (B**2)
    c    = 2 * (C**2)
    scale_range = np.arange(start, stop, step)
    for scale in scale_range:
        filtered_img = vesselness_2D(src, scale, beta, c)
        all_filters.append(filtered_img)
    
    # pick the pixels with the highest vesselness value
    max_img = all_filters[0]
    output_img = np.zeros(src.shape)
    
    for x in range(src.shape[0]):
        for y in range(src.shape[1]):
            max_value = max_img[x, y]
            for img in all_filters:
                if (img[x, y] > max_value):
                    max_value = img[x, y]
            output_img[x, y] = max_value
    
    return output_img
        

def vesselness_3D(src, scale, alpha, beta, c):
    # eigenvalue calculations from Dxx, Dxy, Dyy and compute the vesselnes function
    s3 = scale * scale * scale
    
    D = np.zeros((src.shape[0], src.shape[1], src.shape[2], 3,3))
    
    filters.gaussian_filter(src, (scale, scale, scale), (0, 0, 2), D[:, :, :, 2,2])
    filters.gaussian_filter(src, (scale, scale, scale), (0, 1, 1), D[:, :, :, 1,2])
    filters.gaussian_filter(src, (scale, scale, scale), (0, 2, 0), D[:, :, :, 1,1])
    filters.gaussian_filter(src, (scale, scale, scale), (2, 0, 0), D[:, :, :, 0,0])
    filters.gaussian_filter(src, (scale, scale, scale), (1, 0, 1), D[:, :, :, 0,2])
    filters.gaussian_filter(src, (scale, scale, scale), (1, 1, 0), D[:, :, :, 0,1])
    D[:, :, :, 2,1] = D[:, :, :, 1,2]
    D[:, :, :, 1,0] = D[:, :, :, 0,1]
    D[:, :, :, 2,0] = D[:, :, :, 0,2]
    
    # normalization
    D *= s3
    
    output = np.zeros((src.shape))
    
    for x in range(src.shape[0]):
        for y in range(src.shape[1]):
            for z in range(src.shape[2]):
            
                lambda3, lambda2, lambda1 = lin.eigvalsh(D[z, y, x, :, :])
                if (abs(lambda1) > abs(lambda2)):
                    lambda2, lambda1 = lambda1, lambda2
                if (abs(lambda2) > abs(lambda3)):
                    lambda3, lambda2 = lambda2, lambda3
                    
                if (lambda3==0):
                    lambda3 = math.nextafter(0,1)
                if (lambda2 == 0):
                    lambda2 = math.nextafter(0,1)
                
                Rb2 = np.float128((lambda1**2)/(lambda2 * lambda3))             # Rb2 tends to get very large -> use of float128
                Ra = lambda2 / lambda3
                S2 = (lambda1**2) + (lambda2**2) + (lambda3**2)
                
                term1 = math.exp(-(Ra**2) / alpha)
                try: 
                    term2 = np.exp(-Rb2 / beta)
                except OverflowError:
                    term2 = float('inf')
                    print('yes')
                term3 = math.exp(-S2 / c)
                
                output[x, y, z] = (1.0 - term1) * (term2) * (1.0 - term3) if (lambda2 <= 0 and lambda3 <= 0) else 0
            
    return output

def frangi_3D(src, A, B, C, start, stop, step):
    all_filters = []
    
    beta  = 2 * (B**2)
    c     = 2 * (C**2)
    alpha = 2 * (A**2)
    scale_range = np.arange(start, stop, step)
    for scale in scale_range:
        print('Scale: ' + str(scale) + ' started ...')
        start = time.time()
        filtered_vol = vesselness_3D(src, scale, alpha, beta, c)
        all_filters.append(filtered_vol)
        print('Scale: ' + str(scale) + ' finished. \t Execution time: ' + str(time.time() - start))
    
    # pick the pixels with the highest vesselness value
    max_vol = all_filters[0]
    output_vol = np.zeros(src.shape)
    
    for x in range(src.shape[0]):
        for y in range(src.shape[1]):
            for z in range(src.shape[2]):
                max_value = max_vol[x, y, z]
                for vol in all_filters:
                    if (vol[x, y, z] > max_value):
                        max_value = vol[x, y, z]
                output_vol[x, y] = max_value
    
    return output_vol























