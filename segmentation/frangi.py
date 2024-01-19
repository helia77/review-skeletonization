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
        

def max_norm(src, scale):
    # convolving image with Gaussian derivatives - including Dxx, Dxy, Dyy
    max_norm_all = 0.0
    src = src/255
    for s in scale:
        D = np.zeros((src.shape[0], src.shape[1], src.shape[2], 3,3))
        
        filters.gaussian_filter(src, (s, s, s), (0, 0, 2), D[:, :, :, 2,2])
        filters.gaussian_filter(src, (s, s, s), (0, 1, 1), D[:, :, :, 1,2])
        filters.gaussian_filter(src, (s, s, s), (0, 2, 0), D[:, :, :, 1,1])
        filters.gaussian_filter(src, (s, s, s), (2, 0, 0), D[:, :, :, 0,0])
        filters.gaussian_filter(src, (s, s, s), (1, 0, 1), D[:, :, :, 0,2])
        filters.gaussian_filter(src, (s, s, s), (1, 1, 0), D[:, :, :, 0,1])
        
        D[:, :, :, 2,1] = D[:, :, :, 1,2]
        D[:, :, :, 1,0] = D[:, :, :, 0,1]
        D[:, :, :, 2,0] = D[:, :, :, 0,2]
    
        # find norm
        norm = lin.norm(D*s*s*s)
        max_norm = np.max(norm)
        if max_norm > max_norm_all:
            max_norm_all = max_norm
    
    return max_norm_all

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
    #sprint('Eigen Done.')
    return lambdas

def terms_alpha(src, scale, beta, c, back='white'):
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
                    Rb2 = (l1**2)/(l2 * l3)
                    S2 = (l1**2) + (l2**2) + (l3**2)
                    
                    term2 = np.exp(-Rb2 / beta)
                    term3 = math.exp(-S2 / c)
                    
                    output[z, y, x, 0] = Ra2
                    output[z, y, x, 1] = term2
                    output[z, y, x, 2] = (1.0 - term3)
    print('terms calculation done.')
    return output
    
def terms_alpha_only(src, scale, back='white'):
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
                    
                    Ra2 = (l2 / l3)**2
                    
                    output[z, y, x, 0] = Ra2
                    output[z, y, x, 1] = 1
                    output[z, y, x, 2] = 1
    print('terms calculation done.')
    return output

def terms_beta(src, scale, alpha, c, back='white'):
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
                    Rb2 = (l1**2)/(l2 * l3)
                    S2  = (l1**2) + (l2**2) + (l3**2)
                    
                    term1 = math.exp(-(Ra2) / alpha)
                    term3 = math.exp(-S2 / c)
                    
                    output[z, y, x, 0] = (1.0 - term1)
                    output[z, y, x, 1] = Rb2
                    output[z, y, x, 2] = (1.0 - term3)
    print('terms calculation done.')
    return output

def terms_beta_only(src, scale, back='white'):
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
                    
                    Rb2 = (l1**2)/(l2 * l3)
                    
                    output[z, y, x, 0] = 1
                    output[z, y, x, 1] = Rb2
                    output[z, y, x, 2] = 1
    print('terms calculation done.')
    return output

def terms_c_only(src, scale, back='white'):
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
                    S2  = (l1**2) + (l2**2) + (l3**2)
                    
                    output[z, y, x, 0] = 1
                    output[z, y, x, 1] = 1
                    output[z, y, x, 2] = S2
    print('terms calculation done.')
    return output

def terms_c(src, scale, alpha, beta, back='white'):
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
                    Rb2 = (l1**2)/(l2 * l3)
                    S2  = (l1**2) + (l2**2) + (l3**2)
                    
                    term1 = math.exp(-(Ra2) / alpha)
                    term2 = np.exp(-Rb2 / beta)
                    
                    output[z, y, x, 0] = (1.0 - term1)
                    output[z, y, x, 1] = term2
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
    output_vol = np.zeros(max_vol.shape)
    for x in range(max_vol.shape[2]):
        for y in range(max_vol.shape[1]):
            for z in range(max_vol.shape[0]):
                max_value = max_vol[z, y, x]
                for vol in all_filters:
                    if (vol[z, y, x] > max_value):
                        max_value = vol[z, y, x]
                output_vol[z, y, x] = max_value
    
    return output_vol

def process_alpha(A, terms, sample_gr):
    alpha = 2 * (A**2)
    vesselness_1 = vesselnese_alpha(terms[0], alpha)
    vesselness_2 = vesselnese_alpha(terms[1], alpha)
    vesselness_3 = vesselnese_alpha(terms[2], alpha)
    vesselness_4 = vesselnese_alpha(terms[3], alpha)
    #vesselness_5 = vesselnese_alpha(terms[4], alpha)
    all_filters = [vesselness_1, vesselness_2, vesselness_3, vesselness_4]#, vesselness_5]

    output = highest_pixel(all_filters)             # outputed volume for this alpha value

    # apply otsu's threshold
    thresh_volume, best_thresh = cp_th.compute_otsu_img(output,  background='black')
    print("\nThresh: ", best_thresh)
    # calculate metrics
    met_otsu = mt.metric(sample_gr, thresh_volume)
    #print('.', end='')
    return np.uint8(output), thresh_volume, met_otsu

def process_beta(B, terms, sample_gr):
    beta = 2 * (B**2)
    vesselness_1 = vesselnese_beta(terms[0], beta)
    vesselness_2 = vesselnese_beta(terms[1], beta)
    vesselness_3 = vesselnese_beta(terms[2], beta)
    vesselness_4 = vesselnese_beta(terms[3], beta)
    #vesselness_5 = vesselnese_beta(terms[4], beta)
    all_filters = [vesselness_1, vesselness_2, vesselness_3, vesselness_4]#, vesselness_5]

    output = highest_pixel(all_filters)

    # apply otsu's threshold
    thresh_volume, best_thresh = cp_th.compute_otsu_img(output,  background='black')
    print("\nThresh: ", best_thresh)
    # calculate metrics
    met_otsu = mt.metric(sample_gr, thresh_volume)
    #print('.', end='')
    return np.uint8(output), thresh_volume, met_otsu

def process_c(C, terms, sample_gr):
    c = 2 * (C**2)
    vesselness_1 = vesselnese_c(terms[0], c)
    vesselness_2 = vesselnese_c(terms[1], c)
    vesselness_3 = vesselnese_c(terms[2], c)
    vesselness_4 = vesselnese_c(terms[3], c)
    #vesselness_5 = vesselnese_c(terms[4], c)
    all_filters = [vesselness_1, vesselness_2, vesselness_3, vesselness_4]#, vesselness_5]

    output = highest_pixel(all_filters)

    # apply otsu's threshold
    # after Frangi's filter, the background is black
    thresh_volume, best_thresh = cp_th.compute_otsu_img(output,  background='black')
    print("\nThresh: ", best_thresh)
    # calculate metrics
    met_otsu = mt.metric(sample_gr, thresh_volume)
    #print('.', end='')
    return output, thresh_volume, met_otsu

def vesselness_3D(src, scale, alpha, beta, c, background):
    s3 = scale * scale * scale
    
    # convolving image with Gaussian derivatives - including Dxx, Dxy, Dyy
    D = np.zeros((src.shape[0], src.shape[1], src.shape[2], 3,3))
    
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (0, 0, 2), D[:, :, :, 2,2])
    #print('1st done: ', time.time() - start, ' seconds')
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (0, 1, 1), D[:, :, :, 1,2])
    #print('2nd done: ', time.time() - start, ' seconds')
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (0, 2, 0), D[:, :, :, 1,1])
    #print('3rd done: ', time.time() - start, ' seconds')
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (2, 0, 0), D[:, :, :, 0,0])
    #print('4th done: ', time.time() - start, ' seconds')
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (1, 0, 1), D[:, :, :, 0,2])
    #print('5th done: ', time.time() - start, ' seconds')
    start = time.time()
    filters.gaussian_filter(src, (scale, scale, scale), (1, 1, 0), D[:, :, :, 0,1])
    #print('6th done: ', time.time() - start, ' seconds')
    
    D[:, :, :, 2,1] = D[:, :, :, 1,2]
    D[:, :, :, 1,0] = D[:, :, :, 0,1]
    D[:, :, :, 2,0] = D[:, :, :, 0,2]
    #print('Gaussian done.')
    # normalization
    D *= s3

    output = np.zeros((src.shape))
    #print('eigendecoposition: ...', end=' ')
    start = time.time()
    lambdas = lin.eigvalsh(D)
    #print(' Done.')
    #print('Execution time: ', time.time() - start, ' seconds')
    
    
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


def upgrade_vesselness(src, A, B, C, scale_range, background):
    all_filters = []
    
    beta  = 2 * (B**2)
    c     = 2 * (C**2)
    alpha = 2 * (A**2)
    for scale in scale_range:
        # convolving image with Gaussian derivatives - including Hxx, Hxy, Hyy
        Hxx = np.zeros((src.shape[0], src.shape[1], src.shape[2]), dtype=np.float64)
        Hyy = np.zeros((src.shape[0], src.shape[1], src.shape[2]), dtype=np.float64)
        Hzz = np.zeros((src.shape[0], src.shape[1], src.shape[2]), dtype=np.float64)
        Hxy = np.zeros((src.shape[0], src.shape[1], src.shape[2]), dtype=np.float64)
        Hxz = np.zeros((src.shape[0], src.shape[1], src.shape[2]), dtype=np.float64)
        Hzy = np.zeros((src.shape[0], src.shape[1], src.shape[2]), dtype=np.float64)
        
        filters.gaussian_filter(src, (scale, scale, scale), (0, 0, 2), Hxx)
        filters.gaussian_filter(src, (scale, scale, scale), (0, 1, 1), Hxy)
        filters.gaussian_filter(src, (scale, scale, scale), (0, 2, 0), Hyy)
        filters.gaussian_filter(src, (scale, scale, scale), (2, 0, 0), Hzz)
        filters.gaussian_filter(src, (scale, scale, scale), (1, 0, 1), Hxz)
        filters.gaussian_filter(src, (scale, scale, scale), (1, 1, 0), Hzy)
        
        # correct for scaling - normalization
        s3 = scale * scale * scale
        Hxx *= s3; Hyy *= s3; Hzz *= s3
        Hxy *= s3; Hxz *= s3; Hzy *= s3
        
        # reduce computation by computing vesselness only where needed
        B1 = -(Hxx + Hyy + Hzz)
        B2 = (Hxx * Hyy) + (Hxx * Hzz) + (Hyy * Hzz) - (Hxy * Hxy) - (Hxz * Hxz) - (Hzy * Hzy)
        B3 = (Hxx * Hzy * Hzy) + (Hxy * Hxy * Hzz) + (Hxz * Hyy * Hxz) - (Hxx * Hyy * Hzz) - (Hxy * Hzy * Hxz) - (Hxz * Hxy * Hzy)
        
        T = np.ones_like(B1, dtype=np.uint8)
            
        if background == 'black':
            T[B1 <= 0] = 0
            T[(B2 <= 0) & (B3 == 0)] = 0
            T[(B1 > 0) & (B3 > 0) & (B1*B2 < B3)] = 0
        else:
            T[B1 >= 0] = 0
            T[(B2 >= 0) & (B3 == 0)] = 0
            T[(B1 < 0) & (B2 < 0) & ((-B1)*(-B2) < (-B3))] = 0
        
        del B1, B2, B3
        Hxx *= T; Hyy *= T; Hzz *= T
        Hxy *= T; Hxz *= T; Hzy *= T
        
        H = np.zeros((src.shape[0], src.shape[1], src.shape[2], 3, 3))
        H[:, :, :, 2, 2] = Hxx;     H[:, :, :, 1, 1] = Hyy;     H[:, :, :, 0, 0] = Hzz;
        H[:, :, :, 1, 2] = Hxy;     H[:, :, :, 0, 2] = Hxz;     H[:, :, :, 0, 1] = Hzy;
        H[:, :, :, 2, 1] = Hxy;     H[:, :, :, 2, 0] = Hxz;     H[:, :, :, 1, 0] = Hzy;
        
        del Hxx, Hyy, Hzz, Hxy, Hxz, Hzy
        
        # eigendecomposition
        lambdas = lin.eigvalsh(H)
        
        idx = np.argwhere(T == 1)
        
        V0 = np.zeros_like(src, dtype=np.float64)
            
        for arg in idx:
            i, j, k = arg
            l1, l2, l3 = sorted(lambdas[i, j, k], key=abs)
            
            if background == 'white' and (l2 < 0 or l3 < 0):
                continue
            elif background == 'black' and (l2 > 0 or l3 > 0):
                continue
            
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
            V0[i, j, k] = (1.0 - term1) * (term2) * (1.0 - term3)
            # if (background == 'white'):
            #     V0[i, j, k] = (1.0 - term1) * (term2) * (1.0 - term3) if (l2 >= 0 and l3 >= 0) else 0
            # elif (background == 'black'):
            #     V0[i, j, k] = (1.0 - term1) * (term2) * (1.0 - term3) if (l2 <= 0 and l3 <= 0) else 0
            # else:
            #     print('Invalid background - choose black or white')
            #     return 0
            
        all_filters.append(V0)
    
    output = highest_pixel(all_filters)
    return np.uint8(output * 255)

def frangi_3D(src, A, B, C, scale_range, background='white'):
    all_filters = []
    
    beta  = 2 * (B**2)
    c     = 2 * (C**2)
    alpha = 2 * (A**2)
    for scale in scale_range:
        #print('\nScale: ' + str(scale) + ' started ...')
        #start = time.time()
        filtered_vol = vesselness_3D(src, scale, alpha, beta, c, background)
        all_filters.append(filtered_vol)
        #print('\nScale: ' + str(scale) + ' finished. \nExecution time: ' + str(time.time() - start))
    
    # pick the pixels with the highest vesselness value
    max_vol = all_filters[0]
    output_vol = np.zeros(src.shape)
    #print('getting maximum pixels...')
    for x in range(src.shape[2]):
        for y in range(src.shape[1]):
            for z in range(src.shape[0]):
                max_value = max_vol[z, y, x]
                for vol in all_filters:
                    if (vol[z, y, x] > max_value):
                        max_value = vol[z, y, x]
                output_vol[z, y, x] = max_value
    
    return np.uint8(output_vol * 255)


def beyond_frangi_filter(src, scale_range, tau, background):
    
    all_filters = []
    # for each scale
    for s in scale_range:
        # convolving image with Gaussian derivatives - including Hxx, Hxy, Hyy
        Hxx = np.zeros((src.shape[0], src.shape[1], src.shape[2]))
        Hyy = np.zeros((src.shape[0], src.shape[1], src.shape[2]))
        Hzz = np.zeros((src.shape[0], src.shape[1], src.shape[2]))
        Hxy = np.zeros((src.shape[0], src.shape[1], src.shape[2]))
        Hxz = np.zeros((src.shape[0], src.shape[1], src.shape[2]))
        Hzy = np.zeros((src.shape[0], src.shape[1], src.shape[2]))
        
        filters.gaussian_filter(src, (s, s, s), (0, 0, 2), Hxx)
        filters.gaussian_filter(src, (s, s, s), (0, 1, 1), Hxy)
        filters.gaussian_filter(src, (s, s, s), (0, 2, 0), Hyy)
        filters.gaussian_filter(src, (s, s, s), (2, 0, 0), Hzz)
        filters.gaussian_filter(src, (s, s, s), (1, 0, 1), Hxz)
        filters.gaussian_filter(src, (s, s, s), (1, 1, 0), Hzy)
    
        # correct for scaling - normalization
        s3 = s * s * s
        Hxx *= s3; Hyy *= s3; Hzz *= s3
        Hxy *= s3; Hxz *= s3; Hzy *= s3
        
        # reduce computation by computing vesselness only where needed
        B1 = - (Hxx + Hyy + Hzz)
        B2 = (Hxx * Hyy) + (Hxx * Hzz) + (Hyy * Hzz) - (Hxy * Hxy) - (Hxz * Hxz) - (Hzy * Hzy)
        B3 = (Hxx * Hzy * Hzy) + (Hxy * Hxy * Hzz) + (Hxz * Hyy * Hxz) - (Hxx * Hyy * Hzz) - (Hxy * Hzy * Hxz) - (Hxz * Hxy * Hzy)
        
        T = np.ones_like(B1, dtype=np.uint8)
        
        if background == 'black':
            T[B1 <= 0] = 0
            T[(B2 <= 0) & (B3 == 0)] = 0
            T[(B1 > 0) & (B2 > 0) & (B1*B2 < B3)] = 0
        else:
            T[B1 >= 0] = 0
            T[(B2 >= 0) & (B3 == 0)] = 0
            T[(B1 < 0) & (B2 < 0) & ((-B1)*(-B2) < (-B3))] = 0
        
        del B1, B2, B3
        Hxx *= T; Hyy *= T; Hzz *= T
        Hxy *= T; Hxz *= T; Hzy *= T
        
        H = np.zeros((src.shape[0], src.shape[1], src.shape[2], 3, 3))
        H[:, :, :, 2, 2] = Hxx;     H[:, :, :, 1, 1] = Hyy;     H[:, :, :, 0, 0] = Hzz;
        H[:, :, :, 1, 2] = Hxy;     H[:, :, :, 0, 2] = Hxz;     H[:, :, :, 0, 1] = Hzy;
        H[:, :, :, 2, 1] = Hxy;     H[:, :, :, 2, 0] = Hxz;     H[:, :, :, 1, 0] = Hzy;
        
        del Hxx, Hyy, Hzz, Hxy, Hxz, Hzy
        
        # eigendecomposition
        lambdas = lin.eigvalsh(H)
        
        idx = np.argwhere(T == 1)
        
        V0 = np.zeros_like(src, dtype=np.float64)
        for arg in idx:
            # sort the eigenvalues
            i, j, k = arg
            lambdas[i, j, k] = sorted(lambdas[i, j, k], key=abs)
        
        # find the maximum lambda3 across the volume with scale s
        max_l3 = np.max(lambdas[:, :, :, 2])  
        for arg in idx:
            i, j, k = arg
            _, l2, l3 = lambdas[i, j, k]        # no need for lambda1
            
            if background == 'black':
                l2 = -l2
                l3 = -l3
        
            # calculating lambda rho
            reg_term = tau * max_l3             # regularized term
            l_rho = l3
            if l3 > 0 and l3 < reg_term:
                l_rho = reg_term
            elif l3 <= 0:
                l_rho = 0
                
            # modified vesselness function
            V0[i, j, k] = (l2**2) * (l_rho - l2) * 27 / ((l2 + l_rho) ** 3)
            if l2 >= (l_rho/2) and l_rho > 0:
                V0[i, j, k] = 1
            elif l2 <= 0 or l_rho <= 0:
                V0[i, j, k] = 0
            

        all_filters.append(V0)
    
    # pick the highest vesselness values
    response = highest_pixel(all_filters)
    return np.uint8(response * 255)




















