# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:09:28 2023

@author: helioum
"""

import numpy as np
import frangi as frg
import matplotlib.pyplot as plt
import time
import metric as mt
import cthresholding as cp_th
import plot_curves as pltc
import sklearn.metrics as sk
import pandas as pd
#%%
# test for 3D volume - KESM
volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

sample_vol = volume[300:400, 0:100, 100:200]
sample_gr = gr_truth[300:400, 0:100, 100:200]
#%%
''' preprocess the terms in vesselness function in two situation:
                    1st: calculate the frangi function using two constant parameter and store the third term as without their tuning parameter-
                    just the fraction (R_a, R_b, S), so it can later get processed using a range of the third parameter
                    2nd: calculate the frangi function by replacing the other two terms with one and use a range for the only parameter later
                    This version is later noted as (only) in the code                   '''

scale  = [3, 4, 5, 6]
alpha  = 2 * (0.5 * 0.5)
beta   = 2 * (0.5 * 0.5)

half_norm = frg.max_norm(sample_vol, scale) / 2
c = 2 * half_norm * half_norm
start = time.time()

alpha_terms      = [frg.terms_alpha(sample_vol, s, beta, c) for s in scale]
alpha_terms_only = [frg.terms_alpha_only(sample_vol, s) for s in scale]

beta_terms       = [frg.terms_beta(sample_vol, s, alpha, c) for s in scale]
beta_terms_only  = [frg.terms_beta_only(sample_vol, s) for s in scale]

c_terms          = [frg.terms_c(sample_vol, s, alpha, beta) for s in scale]
c_terms_only     = [frg.terms_c_only(sample_vol, s) for s in scale]
print(f"Calculating terms took {time.time() - start} seconds")
    
#%%
''' The next 3 cells calculate the "threshs" variables which each cell stand for the parameter's value. Then for each variable inside:
    cell[0]: frangi filtered applied volume, intensity ranges from 0 to 255
    cell[1]: frangi+Otsu applied volume (binarized), intensity range [0, 1]
    cell[2]: metrics of cell[1]''' # This part can totally get removed! Will remove later.

# calculate for range of c values
c = half_norm
c_range = [c-27.37, c, c+50.63, 150, 300]#np.linspace(1, 100, 5)
start = time.time()
threshs_c       = [frg.process_c(C, c_terms, sample_gr) for C in c_range]
threshs_c_only  = [frg.process_c(C, c_terms_only, sample_gr) for C in c_range]
print(f"\nc parameter took {time.time() - start} seconds")
c_range = np.array(c_range)

#%%
# calculate for range of beta values
b = 0.5
beta_range = [b/100, b/10, b, b*10, b*100]
start = time.time()
threshs_beta        = [frg.process_beta(B, beta_terms, sample_gr) for B in beta_range]
threshs_beta_only   = [frg.process_beta(B, beta_terms_only, sample_gr) for B in beta_range]
print(f"\nbeta parameter: took {time.time() - start} seconds")
beta_range = np.array(beta_range)

#%%
# calculate for range of alpha values
a = 0.5
alpha_range = [a/100, a/10, a, a*10, a*100]
start = time.time()
threshs_alpha       = [frg.process_alpha(A, alpha_terms, sample_gr) for A in alpha_range]
threshs_alpha_only  = [frg.process_alpha(A, alpha_terms_only, sample_gr) for A in alpha_range]
print(f"\nalpha parameter took {time.time() - start} seconds")
alpha_range = np.array(alpha_range)

#%%
'''     plots the precision-recall curve for all params 
        and one terms (only) range. Otsu's point added on the curve                   '''

# calculate the precision-recall point of Otsu's found by applying otsu on volume's slices (otsu_image)
_, best_thresh = cp_th.compute_otsu_img(sample_vol, 'white')
threshed = (sample_vol <= best_thresh)
met = mt.metric(sample_gr, threshed)
precision = met.precision()
recall = met.TPR()

# plot the precision-recall curve for each parameter (both condition:   1st: using all parameters, other two params constant at best value, 
#                                                                       2nd: replacing the other two terms with 1)

for param in [[threshs_c, threshs_c_only, c_range, 'c'], [threshs_beta, threshs_beta_only, beta_range,'$\\beta$'],
              [threshs_alpha, threshs_alpha_only, alpha_range, '$\\alpha$']]:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plt.suptitle('Precision-Recall', fontsize=16)
    plt.sca(axs[0])
    plt.grid()
    title = param[3] + ' + all params'
    param[2] = np.around(param[2], 3)
    pltc.plot_pre_recall(param[0][4][0], sample_gr, marker='>', label=f'{param[3]}='+str(param[2][4]), color='#0307bd', title=title)
    pltc.plot_pre_recall(param[0][3][0], sample_gr, marker='x', label=f'{param[3]}='+str(param[2][3]), color='#3c67d3', title=title)
    pltc.plot_pre_recall(param[0][2][0], sample_gr, marker=',', label=f'{param[3]}='+str(param[2][2]), color='#80b0ff', title=title)
    pltc.plot_pre_recall(param[0][1][0], sample_gr, label=f'{param[3]}='+str(param[2][1]), color='#e888cd', title=title)
    pltc.plot_pre_recall(param[0][0][0], sample_gr, label=f'{param[3]}='+str(param[2][0]), color='#de0000', title=title)
    plt.scatter(recall, precision, color='g', marker='o', label='Otsu')
    plt.legend(loc='lower left')
    
    plt.sca(axs[1]) 
    plt.grid()
    title = param[3] + ' only'
    pltc.plot_pre_recall(param[1][4][0], sample_gr, marker='>', label=f'{param[3]}='+str(param[2][4]), color='#0307bd', title=title)
    pltc.plot_pre_recall(param[1][3][0], sample_gr, marker='x', label=f'{param[3]}='+str(param[2][3]), color='#3c67d3', title=title)
    pltc.plot_pre_recall(param[1][2][0], sample_gr, marker=',', label=f'{param[3]}='+str(param[2][2]), color='#80b0ff', title=title)
    pltc.plot_pre_recall(param[1][1][0], sample_gr, label=f'{param[3]}='+str(param[2][1]), color='#e888cd', title=title)
    pltc.plot_pre_recall(param[1][0][0], sample_gr, label=f'{param[3]}='+str(param[2][0]), color='#de0000', title=title)
    plt.scatter(recall, precision, color='g', marker='o', label='Otsu')
    plt.legend(loc='lower left')
    
#%%
# calcualte the Area Under the Curve (AUC) of precision-recall curve for each parameter (refer to titles and their x-axis)
gr_truth_1d = np.ravel(sample_gr)
for param in [[threshs_c, threshs_c_only, c_range, 'c'], [threshs_beta, threshs_beta_only, beta_range,'$\\beta$'],
              [threshs_alpha, threshs_alpha_only, alpha_range, '$\\alpha$']]:
    auc = []
    auc_only = []
    for i in range(5):
        frangi_1d = np.ravel(param[0][i][0]/255)
        frangi_only_1d = np.ravel(param[1][i][0]/ 255)
        prec, rec, thresholds = sk.precision_recall_curve(gr_truth_1d, frangi_1d, pos_label=1)
        auc.append(sk.auc(rec, prec))
        prec, rec, thresholds = sk.precision_recall_curve(gr_truth_1d, frangi_only_1d, pos_label=1)
        auc_only.append(sk.auc(rec, prec))
    
    # plot the AUC curve for different values of c and c(only) using the written function from plot_curves.py
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plt.suptitle('AUC curve', fontsize=16)
    plt.sca(axs[0])
    frangis = [param[0][i][0] for i in range(len(param[0]))]
    plt.grid()
    pltc.plot_auc_pr(frangis, sample_gr, param[2], f'{param[3]} + all params', f'{param[3]} range')
    
    plt.suptitle('AUC curve', fontsize=16)
    plt.sca(axs[1])
    frangis_only = [param[1][i][0] for i in range(len(param[0]))]
    plt.grid()
    pltc.plot_auc_pr(frangis_only, sample_gr, param[2], f'{param[3]} only', f'{param[3]} range')

#%%
'''     create a dataframe of all metrics for each parameter (either alone or with other parameters)
        c_all/beta_all/alpha_all means frangi filter has been calculated for a range of the parameter, 
        while keeping the other two constant at their optimum value.
        c_only/beta_only/alpha_only means frangi filter has been calculated for a range of the parameter,
        replacing the other two exponential with 1                                                          '''
        
metric_dataframe = {'c_all':{}, 'c_only':{}, 'beta_all':{}, 'beta_only':{}, 'alpha_all':{}, 'alpha_only':{}}
for param in [[threshs_c, 'c_all', c_range], [threshs_c_only, 'c_only', c_range],
              [threshs_beta, 'beta_all', beta_range], [threshs_beta_only, 'beta_only', beta_range],
              [threshs_alpha, 'alpha_all', alpha_range], [threshs_alpha_only, 'alpha_only', alpha_range]]:
    # Values for 'frangi'
    frangi_jaccard_values = [mt.metric(sample_gr, param[0][i][0]).jaccard() for i in range(5)]
    frangi_dice_values = [mt.metric(sample_gr, param[0][i][0]).dice() for i in range(5)]
    
    # Values for 'frangi+otsu'
    frangi_otsu_jaccard_values = [param[0][i][2].jaccard() for i in range(5)]
    frangi_otsu_dice_values = [param[0][i][2].dice() for i in range(5)]
    # Create a DataFrame
    data = {
         param[1].split('_')[:-1][0]: param[2],
        'frangi_jaccard': frangi_jaccard_values,
        'frangi_dice': frangi_dice_values,
        'frangi_otsu_jaccard': frangi_otsu_jaccard_values,
        'frangi_otsu_dice': frangi_otsu_dice_values
    }
    
    metric_dataframe[param[1]] = pd.DataFrame(data)

#%%
no = 20
for param in [[threshs_c, threshs_c_only, 'c'], [threshs_beta, threshs_beta_only, '$\\beta$'], [threshs_alpha, threshs_alpha_only, '$\\alpha$']]:
    fig, ax = plt.subplots(5, 5)
    fig.suptitle('changes of alpha parameter - Image ' + str(no))
    for i in range(5):
        for j in range(5):
            if j == 0:
                ax[j, i].imshow(param[0][i][0][no], cmap='gray')
                ax[j, i].set_title(f'{param[2]}='+str(alpha_range[i]))
            elif j == 1:
                ax[j, i].imshow(param[0][i][1][no], cmap='gray')
            elif j == 2:
                ax[j, i].imshow(param[1][i][0][no], cmap='gray')
            elif j == 3:
                ax[j, i].imshow(param[1][i][1][no], cmap='gray')
            elif j == 4:
                if i == 1:
                    ax[j, i].imshow(sample_vol[no], cmap='gray')
                elif i == 3:
                    ax[j, i].imshow(sample_gr[no], cmap='gray')
                else:
                    ax[j, i].axis('off')
            ax[j, i].axis('off')
    
    ax[0, 0].text(-0.5, 0.5, 'all-frangi', fontsize=10, va='center', ha='center', rotation='vertical', transform=ax[0, 0].transAxes)
    ax[1, 0].text(-0.5, 0.5, 'all-otsu', fontsize=10, va='center', ha='center', rotation='vertical', transform=ax[1, 0].transAxes)
    ax[2, 0].text(-0.5, 0.5, f'{param[2]} - frangi', fontsize=10, va='center', ha='center', rotation='vertical', transform=ax[2, 0].transAxes)
    ax[3, 0].text(-0.5, 0.5, f'{param[2]} - otsu', fontsize=10, va='center', ha='center', rotation='vertical', transform=ax[3, 0].transAxes)
    ax[4, 1].text(-0.5, 0.5, 'raw data', fontsize=10, va='center', ha='center', rotation='vertical', transform=ax[4, 1].transAxes)
    ax[4, 3].text(-0.5, 0.5, 'ground truth', fontsize=10, va='center', ha='center', rotation='vertical', transform=ax[4, 3].transAxes)
