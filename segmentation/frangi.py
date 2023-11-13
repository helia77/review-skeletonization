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
import cupy as cp
from scipy.ndimage import filters
import cthresholding as cp_th
import metric as mt
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
            l2 = 0.5 * (tmp - tmp2)
            
            # making sure they're sorted based on absolute values
            if (abs(lambda1) < abs(l2)):
                l2, lambda1 = lambda1, l2
            
            # ----------- using numpy linalg eigendecomposition (slower) ------------- #
            # hess_mat = np.zeros((2,2))
            # hess_mat[0, 0] = dyy
            # hess_mat[0, 1] = dxy
            # hess_mat[1, 0] = dxy
            # hess_mat[1, 1] = dxx
            
            # lam1, lam2 = np.linalg.eigvalsh(hess_mat)
            # if (abs(lam1) > abs(lam2)):
            #     lambda1, l2 = lam2, lam1
           
            if(l2==0):
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

def terms_alpha(src, scale, beta, c):
    lambdas = eigens(src, scale)
    output = np.zeros(lambdas.shape)
    for x in range(lambdas.shape[2]):
        for y in range(lambdas.shape[1]):
            for z in range(lambdas.shape[0]):
                l1, l2, l3 = sorted(lambdas[z, y, x], key=abs)
                if (l2 < 0 or l3 < 0):
                    continue
                else:
                    if (l3 == 0):
                        l3 = math.nextafter(0,1)
                    if (l2 == 0):
                        l2 = math.nextafter(0,1)
                    
                    Ra2 = (l2 / l3)**2
                    Rb2 = np.float64((l1**2)/(l2 * l3))            # Rb2 tends to get very large -> use float64
                    S2 = (l1**2) + (l2**2) + (l3**2)
                    
                    term2 = np.exp(-Rb2 / beta)
                    term3 = math.exp(-S2 / c)
                    
                    output[z, y, x, 0] = Ra2
                    output[z, y, x, 1] = term2
                    output[z, y, x, 2] = (1.0 - term3)
    print('terms calculation done.')
    return output


def terms_beta(src, scale, alpha, c):
    lambdas = eigens(src, scale)
    output = np.zeros(lambdas.shape)
    for x in range(lambdas.shape[2]):
        for y in range(lambdas.shape[1]):
            for z in range(lambdas.shape[0]):
                l1, l2, l3 = sorted(lambdas[z, y, x], key=abs)
                if (l2 < 0 or l3 < 0):
                    continue
                else:
                    if (l3 == 0):
                        l3 = math.nextafter(0,1)
                    if (l2 == 0):
                        l2 = math.nextafter(0,1)
                    
                    Ra2 = (l2 / l3)**2
                    Rb2 = (l1**2)/(l2 * l3)
                    S2 = (l1**2) + (l2**2) + (l3**2)
                    
                    term1 = math.exp(-(Ra2) / alpha)
                    term3 = math.exp(-S2 / c)
                    
                    output[z, y, x, 0] = (1.0 - term1)
                    output[z, y, x, 1] = Rb2
                    output[z, y, x, 2] = (1.0 - term3)
    print('terms calculation done.')
    return output

def terms_c(src, scale, alpha=1, beta=1, back='white'):
    lambdas = eigens(src, scale)
    output = np.zeros(lambdas.shape)
    for x in range(lambdas.shape[2]):
        for y in range(lambdas.shape[1]):
            for z in range(lambdas.shape[0]):
                l1, l2, l3 = sorted(lambdas[z, y, x], key=abs)
                if (back=='white' and (l2 < 0 or l3 < 0)):
                    continue
                elif(back=='black' and (l2 > 0 or l3 > 0)):
                    continue
                else:
                    if (l3 == 0):
                        l3 = math.nextafter(0,1)
                    if (l2 == 0):
                        l2 = math.nextafter(0,1)
                    
                    Ra2 = (l2 / l3)**2
                    Rb2 = np.float64((l1**2)/(l2 * l3))            # Rb2 tends to get very large -> use float64
                    S2 = (l1**2) + (l2**2) + (l3**2)
                    
                    term1 = math.exp(-(Ra2) / alpha)
                    term2 = np.exp(-Rb2 / beta)
                    
                    output[z, y, x, 0] = 1#(1.0 - term1)
                    output[z, y, x, 1] = 1#term2
                    output[z, y, x, 2] = S2
    print('terms calculation done.')
    return output


def vesselnese_alpha(values, alpha):
    output = np.zeros((values.shape[0], values.shape[1], values.shape[2]))    
    
    for x in range(values.shape[2]):
        for y in range(values.shape[1]):
            for z in range(values.shape[0]):
                Ra2   = values[z, y, x, 0]
                term1 = math.exp(-Ra2 / alpha)
                term2 = values[z, y, x, 1]
                term3 = values[z, y, x, 2]
                
                output[z, y, x] = (1.0 - term1) * (term2) * (term3)
            
    return np.uint8(output * 255)

def vesselnese_beta(values, beta):
    output = np.zeros((values.shape[0], values.shape[1], values.shape[2]))    
    
    for x in range(values.shape[2]):
        for y in range(values.shape[1]):
            for z in range(values.shape[0]):
                Rb2   = values[z, y, x, 1]
                term1 = values[z, y, x, 0]
                term2 = np.exp(np.float64(-Rb2 / beta))    # term2 tends to get very large -> use float64
                term3 = values[z, y, x, 2]
                
                output[z, y, x] = (term1) * (term2) * (term3)
            
    return np.uint8(output * 255)

def vesselnese_c(values, c):
    output = np.zeros((values.shape[0], values.shape[1], values.shape[2]))    
    
    for x in range(values.shape[2]):
        for y in range(values.shape[1]):
            for z in range(values.shape[0]):
                S2   = values[z, y, x, 2]
                term1 = values[z, y, x, 0]
                term2 = values[z, y, x, 1]
                term3 = math.exp(-S2 / c)
                
                output[z, y, x] = (term1) * (term2) * (1.0 - term3)
            
    return np.uint8(output * 255)

def highest_pixel(all_filters):
    max_vol = all_filters[0]
    output_vol = np.zeros(max_vol.shape, dtype=np.uint8)
    for x in range(max_vol.shape[2]):
        for y in range(max_vol.shape[1]):
            for z in range(max_vol.shape[0]):
                max_value = max_vol[z, y, x]
                for vol in all_filters:
                    if (vol[z, y, x] > max_value):
                        max_value = vol[z, y, x]
                output_vol[z, y, x] = max_value
    
    return output_vol

def process_alpha(A, ratios, sample_gr):
    alpha = 2 * (A**2)
    vesselness_1 = vesselnese_alpha(ratios[0], alpha)
    vesselness_2 = vesselnese_alpha(ratios[1], alpha)
    vesselness_3 = vesselnese_alpha(ratios[2], alpha)
    vesselness_4 = vesselnese_alpha(ratios[3], alpha)

    all_filters = [vesselness_1, vesselness_2, vesselness_3, vesselness_4]

    output = highest_pixel(all_filters)             # outputed volume for this alpha value

    # apply otsu's threshold
    thresh_volume, best_thresh = cp_th.compute_otsu(output,  background='white')
    
    # calculate metrics
    met_otsu = mt.metric(sample_gr, thresh_volume)
    print('.', end='')
    return np.uint8(output)#, thresh_volume, met_otsu

def process_beta(B, ratios, sample_gr):
    beta = 2 * (B**2)
    vesselness_1 = vesselnese_beta(ratios[0], beta)
    vesselness_2 = vesselnese_beta(ratios[1], beta)
    vesselness_3 = vesselnese_beta(ratios[2], beta)
    vesselness_4 = vesselnese_beta(ratios[3], beta)

    all_filters = [vesselness_1, vesselness_2, vesselness_3, vesselness_4]

    output = highest_pixel(all_filters)             # outputed volume for this alpha value

    # apply otsu's threshold
    thresh_volume, best_thresh = cp_th.compute_otsu(output,  background='white')
    
    # calculate metrics
    met_otsu = mt.metric(sample_gr, thresh_volume)
    print('.', end='')
    return np.uint8(output)#, thresh_volume, met_otsu

def process_c(C, terms, sample_gr, back='white'):
    c = 2 * (C**2)
    vesselness_1 = vesselnese_c(terms[0], c)
    vesselness_2 = vesselnese_c(terms[1], c)
    vesselness_3 = vesselnese_c(terms[2], c)
    vesselness_4 = vesselnese_c(terms[3], c)

    all_filters = [vesselness_1, vesselness_2, vesselness_3, vesselness_4]

    output = highest_pixel(all_filters)             # outputed volume for this alpha value

    # apply otsu's threshold
    thresh_volume, best_thresh = cp_th.compute_otsu(output,  background=back)
    
    # calculate metrics
    met_otsu = mt.metric(sample_gr, thresh_volume)
    print('.', end='')
    return output, thresh_volume, met_otsu

def vesselness_3D(src, scale, alpha, beta, c, background):
    s3 = scale * scale * scale
    
    # convolving image with Gaussian derivatives - including Dxx, Dxy, Dyy
    D = np.zeros((src.shape[0], src.shape[1], src.shape[2], 3,3))
    
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (0, 0, 2), D[:, :, :, 2,2])
    print('1st done: ', time.time() - start, ' seconds')
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (0, 1, 1), D[:, :, :, 1,2])
    print('2nd done: ', time.time() - start, ' seconds')
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (0, 2, 0), D[:, :, :, 1,1])
    print('3rd done: ', time.time() - start, ' seconds')
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (2, 0, 0), D[:, :, :, 0,0])
    print('4th done: ', time.time() - start, ' seconds')
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (1, 0, 1), D[:, :, :, 0,2])
    print('5th done: ', time.time() - start, ' seconds')
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (1, 1, 0), D[:, :, :, 0,1])
    print('6th done: ', time.time() - start, ' seconds')
    
    D[:, :, :, 2,1] = D[:, :, :, 1,2]
    D[:, :, :, 1,0] = D[:, :, :, 0,1]
    D[:, :, :, 2,0] = D[:, :, :, 0,2]
    print('Gaussian done.')
    # normalization
    D *= s3

    output = np.zeros((src.shape))
    print('\neigendecoposition: ...', end=' ')
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
                Ra = l2 / l3
                S2 = (l1**2) + (l2**2) + (l3**2)
                
                term1 = math.exp(-(Ra**2) / alpha)
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
    
    return output_vol























