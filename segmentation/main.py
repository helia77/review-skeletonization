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
DATA = 'LSM'                     # change the dataset name
datasets = {"KESM": 'Data/KESM/sample_vol.npy', "LSM": 'Data/LSM/lsm_brain_8bit.npy', "Micro": 'Data/Micro-CT/sample_micro.npy'}

grounds = {"KESM": 'Data/KESM/sample_gr.npy', 
           "LSM": 'Data/LSM/lsm_brain_gr_truth.npy', 
           "Micro": 'Data/Micro-CT/sample_gr_micro.npy'}

scale_ranges = {"KESM": [0.9, 1.8, 2.7, 3.6], "LSM": [1.8, 3.0, 4.2, 5.4, 6.6], "Micro": [1.4, 2.0, 2.6, 3.2, 3.8, 4.4]}
back = 'white' if DATA=="KESM" else 'black'
#%%
output_path = 'Data/Segmentations/'+ DATA + '/'

volume = np.load(datasets[DATA])
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
params = {"KESM": [0.003, 0.94, 18.69], "LSM": [0.005, 0.85, 22.73], "Micro": [0.2, 0.71, 6.57]}
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
# Beyond Frangi - Optimized 1-parameter (tau) Frangi
taus = {"KESM": 0.77, "LSM": 1, "Micro": 0.737}
tau = taus[DATA]

opt_BFrangi = frangi.beyond_frangi_filter(volume, tau, scale_range, back)
threshed3d_bfrangi, best3d_thresh_bfrangi = th.compute_otsu(opt_BFrangi, background='black')
threshed2d_bfrangi = th.otsu_2D(opt_BFrangi, background='black')

#%%
cnns = {"KESM": 'KESM_new.npy', "LSM": 'LSM_new.npy', "Micro": 'Micro_new.npy'} 

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
    OOF = OOF_raw < OOF_thresh[DATA]
else:
    OOF = OOF_raw > OOF_thresh[DATA]

#%%
outputs= [threshed_otsu2d, threshed_otsu3d, threshed2d_frangi3p_reg, threshed3d_frangi3p_reg, reg_Frangi3p, threshed2d_frangi3p, 
           threshed3d_frangi3p, opt_Frangi3p, threshed2d_bfrangi, threshed3d_bfrangi, opt_BFrangi, OOF, OOF_raw,
           unet, unet_prediction]
#%%
save_names = ['binary_otsu2d', 'binary_otsu3d', 'binary_frangi_reg_otsu2d', 'binary_frangi_reg_otsu3d', 'predicted_frangi_reg',
              'binary_frangi_otsu2d', 'binary_frangi_otsu3d', 'predicted_frangi', 'binary_bfrangi_otsu2d', 
              'binary_bfrangi_otsu3d', 'predicted_bfrangi', 'binary_oof', 'predicted_oof', 'binary_unet', 'predicted_unet']
#%%
# save the result as npy
if SAVE:
    for vol, sname in zip(outputs, save_names):
        np.save(output_path+sname+'.npy', vol)
SAVE = False
#%%
# print out Precision and Recall results
names =  ['Otsu 2D', 'Otsu 3D', 'Reg. Frangi 2D', 'Reg. Frangi 3D', 'Reg. Frangi', 'Opt. Frangi 2D', 'Opt. Frangi 3D',
          'Opt. Frangi', 'B. Frangi 2D', 'B. Frangi 3D', 'BFrangi', 'OOF', 'OOF pred', 'Unet', 'Unet Pred']

# print the results
for vol, name in zip(outputs, names):
    if name in ['Reg. Frangi', 'Opt. Frangi', 'BFrangi', 'OOF pred', 'Unet Pred']:
        continue
    print('-'*10,'\t', name, '\t', '-'*10)
    met = mt.metric(gr_truth, vol)
    print('Dice:\t\t', round(met.dice(), 3))
    print('Jaccard:\t', round(met.jaccard(), 3))
    print('Precision:\t', round(met.precision(), 3))
    print('Recall:\t\t', round(met.TPR(), 3))

#%%
# plot the precision-recall curve of all results
names = ['binary_otsu2d', 'binary_otsu3d', 'predicted_frangi_reg','predicted_frangi', 
          'predicted_bfrangi', 'predicted_oof', 'predicted_unet']

labels = ['2D Otsu', '3D Otsu', 'Frangi', 'Opt. Frangi', 'B. Frangi', 'OOF', 'UNet']
#angles = [0, 0, 30, 20, 0, 0, 9]
#colors = ['#74c476', '#117733', '#661100', '#ddcc77', '#88ccee', '#cc6677', '#0069c0']
#colors = ['#004488', '#228833', '#BB5566', '#DDAA33', '#009988', '#882255', '#000000']
colors = ['#002f61', '#59d4d2', '#FF5733', '#3498DB', '#27AE60', '#8E44AD', '#F1C40F']
plt.figure(1, figsize=(10, 10))
#plt.title('Segmnetation Methods Performance')
for c, name, label in zip(colors, names, labels):
    print('Plot of', label, 'in progress...')
    vol = np.load('Data/Segmentations/'+ DATA + '/' + name + '.npy')
    if DATA == 'KESM' and name == 'predicted_oof':
        vol = -vol
    if np.min(vol) < 0:
        vol = vol + abs(np.min(vol))
    pltc.plot_pre_recall(vol, gr_truth, label=label, color=c, end=False)
    print('Done')
plt.grid(alpha=.3)

#%%
i = 10
fig, axes = plt.subplots(3, 6, figsize=(10, 10))

# Flatten the axes array for easy iteration
axes = axes.flatten()

for idx, (output, name) in enumerate(zip(outputs, save_names)):
    ax = axes[idx+2]
    im = ax.imshow(output[i], cmap='gray')
    ax.set_title(name)
    ax.axis('off')
    
axes[0].imshow(volume[i], 'gray')
axes[0].set_title('Volume')
axes[0].axis('off')

axes[1].imshow(gr_truth[i], 'gray')
axes[1].set_title('Ground')
axes[1].axis('off')

# Hide any remaining subplots if there are more subplots than outputs
fig.delaxes(axes[17])

plt.tight_layout()
plt.show()

#%%
plt.subplot(141).imshow(gr_truth[10], 'gray')
plt.subplot(142).imshow(unet[10], 'gray')
plt.subplot(143).imshow(unet_prediction[10], 'gray')
plt.subplot(144).imshow(volume[10], 'gray')
