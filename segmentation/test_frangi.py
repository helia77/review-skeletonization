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
brain = cv2.imread('brain.bmp', cv2.IMREAD_GRAYSCALE)

start = time.time()
result = frg.frangi_2D(brain, 1, 20, 1, 3, 0.5)
print(time.time() - start)
#%%

plt.imshow(result, cmap='gray')
