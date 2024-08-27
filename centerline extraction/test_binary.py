# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:12:56 2024

@author: helioum
"""

import os
import time
import thinning
import numpy as np
import scipy as sp
import manage_data as md
import matplotlib.pyplot as plt
import skeletonization as skelet

#%%
# perform am skeletonization method on all binary segmentation results, report FNR and FPR
file_path = '../segmentation/Data/Segmentations/LSM/'
binary_files = [np.load(os.path.join(file_path, i)) for i in os.listdir(file_path)]
binary_names = [i.split('.')[0][7:] for i in os.listdir(file_path)]
#gr_truth = np.load('../segmentation/ground_truth_kesm.npy')[0:200, 200:400, 300:500]
gr_truth = md.nrrd2npy('../segmentation/Data/LSM/lsm_brain_gr_truth.nrrd')
binary_files.append(gr_truth)
binary_names.append('ground')
#%%
#########     Change the name of the method below      ##################
skeletons = []
for volume, name in zip(binary_files, binary_names):
    print(f'Apply Lee\'s method on {name}:', end=' ')
    skeleton_result = thinning.skelet_lee(volume)
    skeletons.append(skeleton_result)
    #np.save('Output Data/Lee/'+name+'.npy')
    md.npy2obj(skeleton_result, 'Output Data/LSM/Lee/'+name+'.obj')
    print('done.')
    
#%%
# Evaluation
sigma = 0.01
threshold = 0.5

#gr_skeleton = md.nrrd2npy('centerline.nrrd') #md.NWT('centerline.obj')
gr_skeleton = md.nrrd2npy('../thinning/lsm_centerline.nrrd')
GR = np.argwhere(gr_skeleton).astype(np.float32)
#GR[:, 0] /= 1023.0
#GR[:, 1:3] /= 511.0
GR /= 399.0
GT_tree = sp.spatial.cKDTree(GR)

#%%
num = 1
for skeleton, name in zip(skeletons, binary_names):
    P_T = np.argwhere(skeleton).astype(np.float32)
    P_T /= 199.0
    T_tree = sp.spatial.cKDTree(P_T)
    
    [T_dist, _] = GT_tree.query(P_T)
    [GT_dist, _] = T_tree.query(GR)
    
    # convert distances to Gaussian metrics
    T_metric = np.exp(-0.5 * (T_dist ** 2 / sigma ** 2))
    T_metric[T_metric > threshold] = 1
    T_metric[T_metric <= threshold] = 0
    GT_metric = np.exp(-0.5 * (GT_dist ** 2 / sigma ** 2))
    GT_metric[GT_metric > threshold] = 1
    GT_metric[GT_metric <= threshold] = 0

    #calculate the FPR and FPR
    print('\nBinary Method - ', name)
    print("FNR = " + str(1 - np.mean(GT_metric)))
    print("FPR = " + str(1 - np.mean(T_metric)))
    
    plt.figure(num)
    plt.suptitle(name + ' Method')
    plt.subplot(1, 2, 1)
    #plt.scatter(P_GT[:, 0], P_GT[:, 1], s=sigma*shadow, c="grey")
    plt.scatter(P_T[:, 0], P_T[:, 1], c=1 - T_metric, cmap = "RdYlBu_r")
    plt.title("Test Case Network and Metric", fontsize=10)

    plt.subplot(1, 2, 2)
    plt.scatter(GR[:, 0], GR[:, 1], c=1 - GT_metric, cmap = "RdYlBu_r")
    plt.title("Ground Truth Network and Metric", fontsize=10)
    plt.show()
    num += 1
    
#%%
# plot ROC curve
TPR = []
FPR = []
thresholds = np.linspace(0.001, 1, 30)
plt.figure(6)
#for skeleton, name in zip(skeletons, binary_names):
skeleton = binary_files[2]
P_T = np.argwhere(skeleton).astype(np.float32)
P_T /= 199.0
T_tree = sp.spatial.cKDTree(P_T)

[T_dist, _] = GT_tree.query(P_T)
[GT_dist, _] = T_tree.query(GR)

# convert distances to Gaussian metrics
T_metric = np.exp(-0.5 * (T_dist ** 2 / sigma ** 2))
GT_metric = np.exp(-0.5 * (GT_dist ** 2 / sigma ** 2))

for thresh in thresholds:
    T_metric[T_metric > thresh] = 1
    T_metric[T_metric <= thresh] = 0
    GT_metric[GT_metric > thresh] = 1
    GT_metric[GT_metric <= thresh] = 0
    
    FPR.append(1 - np.mean(T_metric))
    TPR.append(np.mean(GT_metric))
    
plt.plot(FPR, TPR, label = name)
plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='lower right')
plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
plt.plot()