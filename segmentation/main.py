# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:54:00 2024

@author: helioum
"""

import frangi
import numpy as np
import metric as mt
import thresholding as th
import matplotlib.pyplot as plt
import manage_data as md
import plot_curves as pltc
import scipy.io as io
import os

SAVE = True
DATA = 'KESM'                     # change the dataset name
datasets = {"KESM": 'Data/KESM/sample_vol.npy', "LSM": 'Data/LSM/lsm_brain.npy', "Micro": 'Data/Micro-CT/sample_micro.npy'}

grounds = {"KESM": 'Data/KESM/sample_gr.npy', 
           "LSM": 'Data/LSM/lsm_brain_gr_truth.npy', 
           "Micro": 'Data/Micro-CT/sample_gr_micro.npy'}

scale_ranges = {"KESM": [0.9, 1.8, 2.7, 3.6], "LSM": [1.8, 3.0, 4.2, 5.4, 6.6], "Micro": [1.4, 2.0, 2.6, 3.2, 3.8, 4.4]}
back = 'white' if DATA=="KESM" else 'black'
#%%
output_path = 'Data/Segmentations/'+ DATA + '/'

volume = np.load(datasets[DATA])
if DATA == 'Micro':
    volume = (volume >> 8).astype(np.uint8)
scale_range = scale_ranges[DATA]
gr_truth = np.load(grounds[DATA])

#%%
# 2D Otsu's Thresholding
threshed_otsu2d = th.otsu_2D(volume, background=back)

#%%
# 3D Otsu's Thresholding
threshed_otsu3d, best_thresh_3d = th.compute_otsu(volume, background=back)

#%%
# Optimized 3-parameter (alpha, beta, c) Frangi
params = {"KESM": [0.003, 0.94, 18.69], "LSM": [0.001, 0.84, 23.74], "Micro": [0.2, 0.71, 6.57]}
alpha1, beta1, c1 = params[DATA]

opt_Frangi3p = frangi.upgrade_vesselness(volume, alpha1, beta1, c1, scale_range, back)
threshed3d_frangi3p, best3d_thresh_frangi = th.compute_otsu(opt_Frangi3p, background='black')
threshed2d_frangi3p = th.otsu_2D(opt_Frangi3p, background='black')

#%%
# Paper-suggested 3-parameter (alpha, beta, c) Frangi
alpha2 = 0.5
beta2 = 0.5
c2 = frangi.max_norm(volume, scale_range) / 2.0

reg_Frangi3p = frangi.upgrade_vesselness(volume, alpha2, beta2, c2, scale_range, back)
threshed3d_frangi3p_reg, best3d_thresh_frangi = th.compute_otsu(reg_Frangi3p, background='black')
threshed2d_frangi3p_reg = th.otsu_2D(reg_Frangi3p, background='black')

#%%
# Optimized 1-parameter (tau) Frangi - Beyond Frangi
taus = {"KESM": 0.77, "LSM": 1, "Micro": 0.737}
tau = taus[DATA]

opt_BFrangi = frangi.beyond_frangi_filter(volume, tau, scale_range, back)
threshed3d_bfrangi, best3d_thresh_bfrangi = th.compute_otsu(opt_BFrangi, background='black')
threshed2d_bfrangi = th.otsu_2D(opt_BFrangi, background='black')

#%%
cnns = {"KESM": 'Micro_new.npy', "LSM": 'lsm_sample.npy', "Micro": 'Micro.npy'} 

unet = np.load('CNN/processed_npy/' + cnns[DATA])#.astype(np.uint8)
unet_prediction = np.load('CNN/probability_npy/' + cnns[DATA])

if np.unique(unet)[-1] != 1:
    print('emtpy volume')
    unet = np.where(unet==255, 1, 0)

#%%
OOF_data = {"KESM": 'oof_kesm.mat', "LSM": 'oof_lsm.mat', "Micro": 'oof_micro.mat'}
OOF_thresh = {"KESM": -2.2, "LSM": 2.4, "Micro": 0.8}

OOF_raw = io.loadmat(os.path.join('./OOF', OOF_data[DATA]))['results']

if DATA == 'KESM':
    OOF_raw = OOF_raw.T
    OOF = OOF_raw < OOF_thresh[DATA]
else:
    OOF = OOF_raw > OOF_thresh[DATA]

#%%
outputs= [threshed_otsu2d, threshed_otsu3d, threshed2d_frangi3p_reg, threshed3d_frangi3p_reg, reg_Frangi3p, threshed2d_frangi3p, 
           threshed3d_frangi3p, opt_Frangi3p, threshed2d_bfrangi, threshed3d_bfrangi, opt_BFrangi, OOF.T, OOF_raw.T, 
           unet, unet_prediction]
#%%
save_names = ['binary_otsu2d', 'binary_otsu3d', 'binary_Frangi3p_reg_ostu2d', 'binary_Frangi3p_reg_ostu3d', 'predicted_Frangi3d_reg',
              'binary_Frangi3p_ostu2d', 'binary_Frangi3p_ostu3d', 'predicted_Frangi3d', 'binary_BFrangi_otsu2d', 
              'binary_BFrangi_otsu3d', 'predicted_BFrangi', 'binary_oof', 'predicted_oof', 'binary_unet', 'predicted_unet']
#%%
# save the result as npy
if SAVE:
    for vol, sname in zip(outputs, save_names):
        np.save(output_path+sname+'.npy', vol)
SAVE = False
#%%
names =  ['Otsu 2D', 'Otsu 3D', 'Reg. Frangi 2D', 'Reg. Frangi 3D', 'Reg. Frangi', 'Opt. Frangi 2D', 'Opt. Frangi 3D',
          'Opt. Frangi', 'B. Frangi 2D', 'B. Frangi 3D', 'BFrangi', 'Oriented Flux', 'OOF', 'Unet', 'Unet Pred']
# print the results
for vol, name in zip(outputs, names):
    if name in ['Reg. Frangi', 'Opt. Frangi', 'BFrangi', 'OOF', 'Unet Pred']:
        continue
    print('-'*10,'\t', name, '\t', '-'*10)
    met = mt.metric(gr_truth, vol)
    print('Dice:\t\t', round(met.dice(), 3))
    print('Jaccard:\t', round(met.jaccard(), 3))
    print('Precision:\t', round(met.precision(), 3))
    print('Recall:\t\t', round(met.TPR(), 3))

#%%
# plot the precision-recall curve of all results
volumes = [threshed_otsu2d, threshed_otsu3d, opt_Frangi3p, reg_Frangi3p, opt_BFrangi, OOF_raw]
labels = ['binary_otsu2d', 'binary_otsu3d', 'predicted_Frangi3d_reg','predicted_Frangi3d', 'predicted_BFrangi', 'predicted_oof']

plt.figure(1)
plt.title('Segmnetation Methods Performance')
for label in labels:
    vol = np.load('Data/Segmentations/'+ DATA + '/' + label + '.npy')
    pltc.plot_pre_recall(vol, gr_truth, label=label, color=np.random.rand(3,), end=True)
    
#%%
i = 10
fig, axes = plt.subplots(7, 2, figsize=(10, 10))

# Flatten the axes array for easy iteration
axes = axes.flatten()

for idx, (output, name) in enumerate(zip(outputs, save_names)):
    if name in ['Reg. Frangi', 'Opt. Frangi', 'BFrangi']:
        continue
    ax = axes[idx]
    im = ax.imshow(output[i], cmap='gray')
    ax.set_title(name)
    ax.axis('off')
    
axes[8].imshow(volume[i], 'gray')
axes[8].set_title('Volume')
axes[8].axis('off')

axes[9].imshow(gr_truth[i], 'gray')
axes[9].set_title('Ground')
axes[9].axis('off')

# Hide any remaining subplots if there are more subplots than outputs
for idx in range(len(outputs), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()