# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:40:28 2023

@author: helioum
"""

import numpy as np
import frangi as frg
import cv2
import matplotlib.pyplot as plt
import time
#%%
# test for 2D image
brain = cv2.imread('brain.bmp', cv2.IMREAD_GRAYSCALE )
#%%
start = time.time()
result = frg.frangi_2D(brain, 1, 20, 1, 3, 0.5)
print(time.time() - start)
#%%
plt.imshow(result, cmap='gray')
#%%
# test for 3D volume
micro_vol = np.load('micro_volume.npy')
#%%
start = time.time()
result = frg.frangi_3D(micro_vol, 1, 1, 20, 1, 2, 0.5)
print(time.time() - start)