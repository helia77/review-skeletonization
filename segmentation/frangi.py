# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:08:40 2023

@author: helioum
"""

import numpy as np
import math
import scipy.signal as sc
import time
import cv2
import numpy.linalg as lin
#%%

# create 2D Hessian kernels to later convolve with the image
def frangi_hessian(src, scale):
    half_kernel = math.ceil(3 * scale)
    n_kern_x = 2 * half_kernel + 1
    n_kern_y = n_kern_x
    
    # Hessian kernels
    kern_xx_f = np.zeros((n_kern_x, n_kern_y), dtype=float)  # kernel to create Dxx matrix
    kern_xy_f = np.zeros((n_kern_x, n_kern_y), dtype=float)  # kernel to create Dxy matrix
    kern_yy_f = np.zeros((n_kern_x, n_kern_y), dtype=float)  # kernel to create Dyy matrix
    
    s2 = scale * scale
    half_PI_s6 = 1.0 / (2.0 * math.pi * (s2**3))
    
    for x in range(-half_kernel, half_kernel + 1):
        x2 = x**2
        for y in range(-half_kernel, half_kernel + 1):
            y2 = y**2
            kern_xx_f[x + half_kernel][y + half_kernel] = half_PI_s6 * (x2  - s2) * math.exp(-(x2 + y2) / (2.0 * s2))
            kern_xy_f[x + half_kernel][y + half_kernel] = half_PI_s6 * (x * y) * math.exp(-(x2 + y2) / (2.0 * s2))
    
    kern_yy_f = np.transpose(kern_xx_f, (1, 0))
    
    # convolution of each kernel
    Dxx = sc.fftconvolve(src, kern_xx_f, mode='same') * scale * scale
    Dxy = sc.fftconvolve(src, kern_xy_f, mode='same') * scale * scale
    Dyy = sc.fftconvolve(src, kern_yy_f, mode='same') * scale * scale
    
    return Dxx, Dxy, Dyy

# Run 2D Hessian filter with parameter sigma on src, return the filtered image.
def vesselness_2D(Dxx, Dxy, Dyy, beta, c):
    # eigenvalue calculations from Dxx, Dxy, Dyy and compute the vesselnes function
    output = np.zeros((Dxx.shape))
    
    for x in range(Dxx.shape[0]):
        for y in range(Dxx.shape[1]):
            
            dxx = Dxx[x, y]
            dxy = Dxy[x, y]
            dyy = Dyy[x, y]
            
            # calculate eigenvalues
            tmp = dxx + dyy
            tmp2 = math.sqrt((dxx - dyy)**2 + 4*(dxy**2))
            
            mu1 = 0.5 * (tmp + tmp2)
            mu2 = 0.5 * (tmp - tmp2)
            
            lambda1 = mu1 if (abs(mu1) < abs(mu2)) else mu2
            lambda2 = mu2 if (abs(mu1) < abs(mu2)) else mu1

            if(lambda2==0):
                lambda2 = math.nextafter(0, 1)
            
            Rb = lambda1/lambda2
            S2 = (lambda1**2) + (lambda2**2)
            term1 = math.exp(-(Rb**2) / beta)
            term2 = math.exp(-S2 / c)
            
            output[x, y] = term1 * (1.0 - term2) if (lambda2 < 0) else 0
            
    return output

def frangi_2D(src, B, C, start, stop, step):
    all_filters = []
    
    beta = 2 * (B**2)
    c    = 2* (C**2)
    scale_range = np.arange(start, stop, step)
    for scale in scale_range:
        Dxx, Dxy, Dyy = frangi_hessian(src, scale)
        filtered_img = vesselness_2D(Dxx, Dxy, Dyy, beta, c)
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
        

def vesselness_3D(Dxx, Dxy, Dyy, beta, c):
    # eigenvalue calculations from Dxx, Dxy, Dyy and compute the vesselnes function
    output = np.zeros((Dxx.shape))
    
    for x in range(Dxx.shape[0]):
        for y in range(Dxx.shape[1]):
            for z in range(Dxx.shape[2]):
            
                dxx = Dxx[x, y, z]
                dxy = Dxy[x, y, z]
                dyy = Dyy[x, y, z]
                
                tmp = dxx - dyy
                tmp2 = math.sqrt(tmp * tmp + (4 * dxy * dxy))
                
                mu1 = 0.5 * (dxx + dyy + tmp2)
                mu2 = 0.5 * (dxx + dyy - tmp2)
                
                lambda1 = mu1 if (abs(mu1) < abs(mu2)) else mu2
                lambda2 = mu2 if (abs(mu1) < abs(mu2)) else mu1
                if(lambda2==0):
                    lambda2 = math.nextafter(0, 1)
                
                Rb = lambda1/lambda2
                S2 = (lambda1**2) + (lambda2**2)
                term1 = math.exp(-(Rb**2) / beta)
                term2 = math.exp(-S2 / c)
                
                output[x, y] = term1 * (1.0 - term2) if (lambda2 < 0) else 0
            
    return output

def frangi_3D(src, B, C, start, stop, step):

    all_filters = []
    
    beta = 2 * (B**2)
    c    = 2* (C**2)
    scale_range = np.arange(start, stop, step)
    for scale in scale_range:
        Dxx, Dxy, Dyy = frangi_hessian(src, scale)
        filtered_vol = vesselness_3D(Dxx, Dxy, Dyy, beta, c)
        all_filters.append(filtered_vol)
    
    # pick the pixels with the highest vesselness value
    max_img = all_filters[0]
    output_vol = np.zeros(src.shape)
    
    for x in range(src.shape[0]):
        for y in range(src.shape[1]):
            max_value = max_img[x, y]
            for img in all_filters:
                if (img[x, y] > max_value):
                    max_value = img[x, y]
            output_vol[x, y] = max_value
    
    return output_vol























