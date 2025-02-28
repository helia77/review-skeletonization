# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:37:08 2024
create the LSM volume using image folder

@author: helioum
"""

import numpy as np
import manage_data as md
import matplotlib.pyplot as plt
import cv2
from PIL import Image
#%%
folder_path = 'C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/data/LSM/Eye'

img_list = md.load_images(folder_path, num_img_range='all', grayscale=True)


#%%
modified_list = md.change_level(folder_path, num_img_range='all', shadows=0.0, highlights=0.1)
#%%
plt.imshow(modified_list[10], 'gray')
#plt.imshow(cv2.convertScaleAbs(img, 0.2, 10), 'gray')

#%%
volume = []
for img in modified_list:
    volume.append(img)
    
volume = np.stack(volume, axis=0)
#%%
# save the volume
np.save('lsm_raw_410x2801x2001.npy', volume)
#%%
# save and check the slices
folder_path = 'check'
md.save_slices(volume, folder_path, 'all')


