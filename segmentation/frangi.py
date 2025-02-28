# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:08:40 2023

@author: helioum
"""

import numpy as np
import math
import time
import numpy.linalg as lin
from scipy.ndimage import filters
import thresholding as th
import metric as mt
#%%
# filter the given image based on given scale and returns output image with vesselness values
def vesselness_2D(src, scale, beta, c, normalization):
    s2 = 1
    for _ in range(normalization):
        s2 *= scale
    
    # convolving image with Gaussian derivatives - including Dxx, Dxy, Dyy
    D = np.zeros((src.shape[0], src.shape[1], 3))
    filters.gaussian_filter(src, (scale, scale), (0, 2), D[:, :, 0])            # Dxx
    filters.gaussian_filter(src, (scale, scale), (1, 1), D[:, :, 1])            # Dxy
    filters.gaussian_filter(src, (scale, scale), (2, 0), D[:, :, 2])            # Dyy
    D *= s2                                                                     # for normalization
    
    # eigenvalue calculations from Dxx, Dxy, Dyy and compute the vesselnes function
    output = np.zeros((src.shape))
    #lambdas = lin.eigvalsh(D)
    
    for x in range(src.shape[0]):
        for y in range(src.shape[1]):
            
            #lambda1, lambda2 = sorted(lambdas[x,y], key=abs)
            
            # ----------- using conventional way (faster) ------------- #
            dxx = D[x, y, 0]
            dxy = D[x, y, 1]
            dyy = D[x, y, 2]
            
            # # calculate eigenvalues
            # tmp = dxx + dyy
            # tmp2 = math.sqrt((dxx - dyy)**2 + 4*(dxy**2))
            
            # lambda1 = 0.5 * (tmp + tmp2)
            # lambda2 = 0.5 * (tmp - tmp2)
            
            # # making sure they're sorted based on absolute values
            # if (abs(lambda2) < abs(lambda1)):
            #     lambda2, lambda1 = lambda1, lambda2
            
            # ----------- using numpy linalg eigendecomposition (slower) ------------- #
            hess_mat = np.zeros((2,2))
            hess_mat[0, 0] = dyy
            hess_mat[0, 1] = dxy
            hess_mat[1, 0] = dxy
            hess_mat[1, 1] = dxx
            
            lambda1, lambda2 = sorted(np.linalg.eigvalsh(hess_mat), key=abs)
            
           
            if(lambda2==0):
                lambda2 = math.nextafter(0, 1)
            
            Rb2 = np.float64((lambda1**2)/(lambda2**2))
            S2 = (lambda1**2) + (lambda2**2)
            term1 = math.exp((-Rb2) / beta)
            term2 = math.exp(-S2 / c)
            
            output[x, y] = term1 * (1.0 - term2) if (lambda2 < 0) else 0
    return output

def frangi_2D(src, B, C, start, stop, step, normalization):
    # stores all the filtered images based on different scales
    all_filters = []
    
    beta = 2 * (B**2)
    c    = 2 * (C**2)
    scale_range = np.arange(start, stop, step)
    for scale in scale_range:
        filtered_img = vesselness_2D(src, scale, beta, c, normalization)
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
    
    return np.uint8(output_img * 255)
        
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

def highest_pixel(all_filters):
    max_vol = all_filters[0]
    output_vol = np.zeros_like(max_vol)
    for x in range(max_vol.shape[2]):
        for y in range(max_vol.shape[1]):
            for z in range(max_vol.shape[0]):
                max_value = max_vol[z, y, x]
                for vol in all_filters:
                    if (vol[z, y, x] > max_value):
                        max_value = vol[z, y, x]
                output_vol[z, y, x] = max_value
    
    return output_vol

def vesselness_3D(src, scale, alpha, beta, c, background):
    s2 = scale * scale
    
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
    D *= s2

    output = np.zeros((src.shape))
    lambdas = lin.eigvalsh(D)

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
    
    alpha = 2 * (A**2) if A != 0 else math.nextafter(0, 1)
    beta  = 2 * (B**2) if B != 0 else math.nextafter(0, 1)
    c     = 2 * (C**2) if C != 0 else math.nextafter(0, 1)
    
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
        s2 = scale * scale
        Hxx *= s2; Hyy *= s2; Hzz *= s2
        Hxy *= s2; Hxz *= s2; Hzy *= s2
        
        # reduce computation by computing vesselness only where is needed
        # based on the paper of S.-F. Yang and C.-H. Cheng, “Fast computation of Hessian-based enhancement filters for medical images”
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
            
            Rb2 = (l1**2)/(l2 * l3)
            Ra2 = (l2**2) / (l3**2)
            S2 = (l1**2) + (l2**2) + (l3**2)
                
            term1 = math.exp(-Ra2 / alpha)
            term2 = math.exp(-Rb2 / beta)
            term3 = math.exp(-S2 / c)
            V0[i, j, k] = (1.0 - term1) * (term2) * (1.0 - term3)
            
        all_filters.append(V0)
    
    output = highest_pixel(all_filters)
    return output

def frangi_3D(src, A, B, C, scale_range, background='white'):
    all_filters = []
    
    beta  = 2 * (B**2)
    c     = 2 * (C**2)
    alpha = 2 * (A**2)
    for scale in scale_range:

        filtered_vol = vesselness_3D(src, scale, alpha, beta, c, background)
        all_filters.append(filtered_vol)
        
    # select pixels with the highest vesselness value
    max_vol = all_filters[0]
    output_vol = np.zeros(src.shape)
    
    for x in range(src.shape[2]):
        for y in range(src.shape[1]):
            for z in range(src.shape[0]):
                max_value = max_vol[z, y, x]
                for vol in all_filters:
                    if (vol[z, y, x] > max_value):
                        max_value = vol[z, y, x]
                output_vol[z, y, x] = max_value
    
    return np.uint8(output_vol * 255)


def beyond_frangi_filter(src, tau, scale_range, background):
    
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
        s2 = s * s
        Hxx *= s2; Hyy *= s2; Hzz *= s2
        Hxy *= s2; Hxz *= s2; Hzy *= s2
        
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
        
        Hxx *= T; Hyy *= T; Hzz *= T
        Hxy *= T; Hxz *= T; Hzy *= T
        
        H = np.zeros((src.shape[0], src.shape[1], src.shape[2], 3, 3))
        H[:, :, :, 2, 2] = Hxx;     H[:, :, :, 1, 1] = Hyy;     H[:, :, :, 0, 0] = Hzz;
        H[:, :, :, 1, 2] = Hxy;     H[:, :, :, 0, 2] = Hxz;     H[:, :, :, 0, 1] = Hzy;
        H[:, :, :, 2, 1] = Hxy;     H[:, :, :, 2, 0] = Hxz;     H[:, :, :, 1, 0] = Hzy;
        
        
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
    return response




















