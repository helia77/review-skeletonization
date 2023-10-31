# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:39:02 2023

@author: helioum
"""

import manage_data as md
import numpy as np

#%%
folder_path = 'C:/Users/helioum/Desktop/region3_helia'
num_images = 772
volume_1000 = md.load_images(folder_path, num_images, True, grayscale=True)

#%%
# vol_ x _ y _ z : x horizontal, y vertical, z depth
vol_0_0_0 = volume_1000[0:200, 0:200, 0:200]
vol_1_0_0 = volume_1000[0:200, 0:200, 200:400]
vol_2_0_0 = volume_1000[0:200, 0:200, 400:600]
vol_3_0_0 = volume_1000[0:200, 0:200, 600:800]
vol_4_0_0 = volume_1000[0:200, 0:200, 800:1000]

md.numpy_to_nrrd(vol_0_0_0, 'data/vol_0_0_0.nrrd')
md.numpy_to_nrrd(vol_1_0_0, 'data/vol_1_0_0.nrrd')
md.numpy_to_nrrd(vol_2_0_0, 'data/vol_2_0_0.nrrd')
md.numpy_to_nrrd(vol_3_0_0, 'data/vol_3_0_0.nrrd')
md.numpy_to_nrrd(vol_4_0_0, 'data/vol_4_0_0.nrrd')

#%%
vol_0_1_0 = volume_1000[0:200, 200:400, 0:200]
vol_1_1_0 = volume_1000[0:200, 200:400, 200:400]
vol_2_1_0 = volume_1000[0:200, 200:400, 400:600]
vol_3_1_0 = volume_1000[0:200, 200:400, 600:800]
vol_4_1_0 = volume_1000[0:200, 200:400, 800:1000]

md.numpy_to_nrrd(vol_0_1_0, 'data/vol_0_1_0.nrrd')
md.numpy_to_nrrd(vol_1_1_0, 'data/vol_1_1_0.nrrd')
md.numpy_to_nrrd(vol_2_1_0, 'data/vol_2_1_0.nrrd')
md.numpy_to_nrrd(vol_3_1_0, 'data/vol_3_1_0.nrrd')
md.numpy_to_nrrd(vol_4_1_0, 'data/vol_4_1_0.nrrd')