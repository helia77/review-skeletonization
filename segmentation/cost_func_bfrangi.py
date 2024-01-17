# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:27:29 2024

cost function and optimization of Beyond Frangi method
NOTE: the cost function for Beyond Frangi is written for KESM data only (background='white')

@author: helioum
"""

import numpy as np
import numpy.linalg as lin
from scipy.ndimage import filters
import frangi
from scipy.optimize import minimize


def cost_bfrangi(tau):
    # test for 3D volume - KESM
    print('tau=', tau)
    volume = np.load('whole_volume_kesm.npy')
    gr_truth = np.load('ground_truth_kesm.npy')

    # the cropped sizes can be changed
    src = volume[300:350, 50:100, 150:200]
    sample_gr = gr_truth[300:350, 50:100, 150:200]
    
    all_filters = []

    scale_range = np.arange(3, 6, 1)
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
        
        V0 = np.zeros_like(src)
        for arg in idx:
            # sort the eigenvalues
            i, j, k = arg
            lambdas[i, j, k] = sorted(lambdas[i, j, k], key=abs)
        
        # find the maximum lambda3 across the volume with scale s
        max_l3 = np.max(lambdas[:, :, :, 2])  
        for arg in idx:
            i, j, k = arg
            _, l2, l3 = lambdas[i, j, k]        # no need for lambda1
        
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
    
    # pick the pixels with the highest vesselness value
    predicted = frangi.highest_pixel(all_filters)
    
    #predicted, best_thresh = cp_th.compute_otsu_img(response,  background='black')
    TP = np.logical_and(sample_gr, predicted).sum()
    FP = np.logical_and(np.logical_not(sample_gr), predicted).sum()
    FN = np.logical_and(sample_gr, np.logical_not(predicted)).sum()     
    
    dice = (2*TP) / float(2*TP + FP + FN)
    print('dice= ', dice, '\n')
    return 1 - dice



''' Beyond Frangi '''

# Set initial parameter and bounds for Beyond Frangi optimization
initial_tau = [0.5]
tau_bound = [(0.1, 1.0)]

# Run the optimization for Beyond Frangi
result_b = minimize(cost_bfrangi, initial_tau, method='Powell', bounds=tau_bound)
optimized_tau = result_b.x
print('best tau:\t', optimized_tau[0])