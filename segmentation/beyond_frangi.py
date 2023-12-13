# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:19:02 2023

@author: helioum
"""

import numpy as np
import math
import scipy.signal as sc
import time
import numpy.linalg as lin
import cupy as cp
from scipy.ndimage import filters
import cthresholding as cp_th
import metric as mt

#%%
def eigens(src, scale):
    # convolving image with Gaussian derivatives - including Dxx, Dxy, Dyy
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
    s3 = scale * scale * scale
    D *= s3

    lambdas = lin.eigvalsh(D)
    print('Eigen Done.')
    return lambdas

def vesselness_3D(src, scale, tau, background):
    
    # convolving image with Gaussian derivatives - including Dxx, Dxy, Dyy
    D = np.zeros((src.shape[0], src.shape[1], src.shape[2], 3,3))
    
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (0, 0, 2), D[:, :, :, 2,2])
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (0, 1, 1), D[:, :, :, 1,2])
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (0, 2, 0), D[:, :, :, 1,1])
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (2, 0, 0), D[:, :, :, 0,0])
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (1, 0, 1), D[:, :, :, 0,2])
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (1, 1, 0), D[:, :, :, 0,1])
    
    D[:, :, :, 2,1] = D[:, :, :, 1,2]
    D[:, :, :, 1,0] = D[:, :, :, 0,1]
    D[:, :, :, 2,0] = D[:, :, :, 0,2]
    print('Gaussian done.')
    # normalization
    s3 = scale * scale * scale
    D *= s3

    output = np.zeros((src.shape))
    print('eigendecoposition: ...', end=' ')
    start = time.time()
    lambdas = lin.eigvalsh(D)
    print(' Done.')
    print('Execution time: ', time.time() - start, ' seconds')
    
    
    for x in range(src.shape[2]):
        for y in range(src.shape[1]):
            for z in range(src.shape[0]):
                l1, l2, l3 = sorted(lambdas[z, y, x], key=abs)
                    
                if (l3 == 0):
                    l3 = math.nextafter(0,1)
                if (l2 == 0):
                    l2 = math.nextafter(0,1)
                
                Rb2 = np.float64((l1**2)/(l2 * l3))            # Rb2 tends to get very large -> use of float128
                Ra2 = (l2 / l3)**2
                S2 = (l1**2) + (l2**2) + (l3**2)
                
                term1 = math.exp(-Ra2 / alpha)
                term2 = np.exp(-Rb2 / beta)
                term3 = math.exp(-S2 / c)
                
                if (background == 'white'):
                    output[z, y, x] = (1.0 - term1) * (term2) * (1.0 - term3) if (l2 >= 0 and l3 >= 0) else 0
                elif (background == 'black'):
                    output[z, y, x] = (1.0 - term1) * (term2) * (1.0 - term3) if (l2 <= 0 and l3 <= 0) else 0
                else:
                    print('Invalid background - choose black or white')
                    return 0
            
    return output


def frangi_3D(src, A, B, C, start, stop, step, background='white'):
    all_filters = []
    
    beta  = 2 * (B**2)
    c     = 2 * (C**2)
    alpha = 2 * (A**2)
    scale_range = np.arange(start, stop, step)
    for scale in scale_range:
        print('\nScale: ' + str(scale) + ' started ...')
        start = time.time()
        filtered_vol = vesselness_3D(src, scale, alpha, beta, c, background)
        all_filters.append(filtered_vol)
        print('\nScale: ' + str(scale) + ' finished. \nExecution time: ' + str(time.time() - start))
    
    # pick the pixels with the highest vesselness value
    max_vol = all_filters[0]
    output_vol = np.zeros(src.shape)
    print('getting maximum pixels...')
    for x in range(src.shape[2]):
        for y in range(src.shape[1]):
            for z in range(src.shape[0]):
                max_value = max_vol[z, y, x]
                for vol in all_filters:
                    if (vol[z, y, x] > max_value):
                        max_value = vol[z, y, x]
                output_vol[z, y, x] = max_value
    
    return np.uint8(output_vol * 255)