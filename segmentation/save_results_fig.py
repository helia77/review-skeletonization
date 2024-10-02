# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:16:50 2024

@author: helioum
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import cv2 as cv


def outline(binary, img):
    # returns the location of the contour outline of the binary input image
    if np.unique(binary)[1] != 255:
        binary = np.uint8(binary) * 255
    # img should be binary [0, 255] of type uint8
    cont = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    outline = np.zeros(binary.shape, dtype=np.uint8)
    outline = cv.drawContours(outline, cont[0], -1, (255, 255, 255))
    outline = outline > 0
    
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    
    R[outline] = 115
    G[outline] = 77
    B[outline] = 38
    
    return np.dstack((R, G, B))

def false_color(img, ground):
    TP = np.logical_and(img, ground)
    FP = np.logical_and(img, np.logical_not(ground))
    final = np.dstack((FP*255, TP*150, np.zeros((img.shape), dtype=np.uint8)))
    background = np.logical_and(FP == 0, TP == 0)
    final[background] = 255
    return final

#%%
save_path = '../Figs/segmentation/'

kesm = np.load('Data/KESM/sample_vol.npy')
kesm_gr = np.load('Data/KESM/sample_gr.npy')
lsm = np.load('Data/LSM/lsm_brain_8bit.npy')
lsm_gr = np.load('Data/LSM/lsm_brain_gr_truth.npy')
micro =  np.load('Data/Micro/sample_micro.npy')
micro_gr = np.load('Data/Micro/sample_gr_micro.npy')

DATA = 'LSM'

slices = {'KESM': 123, 'LSM': 54, 'Micro': 37}
i = slices[DATA]

otsu2 = np.load('Data/Segmentations/' + DATA + '/binary_otsu2d.npy')[i]
otsu3 = np.load('Data/Segmentations/' + DATA + '/binary_otsu3d.npy')[i]
frangi_reg = np.load('Data/Segmentations/' + DATA + '/binary_frangi_reg_otsu3d.npy')[i]
frangi = np.load('Data/Segmentations/' + DATA + '/binary_frangi_otsu3d.npy')[i]
bfrangi = np.load('Data/Segmentations/' + DATA + '/binary_bfrangi_otsu3d.npy')[i]
unet = np.load('Data/Segmentations/' + DATA + '/binary_unet.npy')[i]
oof = np.load('Data/Segmentations/' + DATA + '/binary_oof.npy')[i]

raw = {"KESM": kesm, "LSM": lsm, "Micro": micro}
ground = {"KESM": kesm_gr, "LSM": lsm_gr, "Micro": micro_gr}


#%%

images = {'otsu2d':otsu2, 'otsu3d':otsu3, 'frangi_reg':frangi_reg, 'frangi':frangi, 'bfrangi':bfrangi, 'unet':unet, 'oof':oof}

for name, img in images.items():
    final = false_color(img, ground[DATA][i])
    # plt.imshow(final, 'gray')
    ski.io.imsave(save_path + DATA + '_' + name + '.png', final)

#%%
for name, vol in ground.items():
    j = slices[name]
    img = vol[j]
    blank = np.zeros((img.shape), dtype=np.uint8)
    final = np.dstack((blank, img*150, blank))
    final[img == 0] = 255
    # plt.imshow(final)
    ski.io.imsave(save_path + name + '_ground.png', final)
#%%
plt.figure(figsize=(19, 5))

plt.subplot(191).imshow(raw[DATA][i], 'gray')
plt.subplot(192).imshow(ground[DATA][i], 'gray')
plt.subplot(193).imshow(otsu2, 'gray')
plt.subplot(194).imshow(otsu3, 'gray')
plt.subplot(195).imshow(frangi_reg, 'gray')
plt.subplot(196).imshow(frangi, 'gray')
plt.subplot(197).imshow(bfrangi, 'gray')
plt.subplot(198).imshow(unet, 'gray')
plt.subplot(199).imshow(oof, 'gray')

