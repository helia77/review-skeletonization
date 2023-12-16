# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:59:23 2023

@author: helioum
"""

import numpy as np
import scipy as sp
import manage_data as md
from skimage import measure
#%%
# test for 3D volume - KESM
volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

sample_vol = volume[237:344, 268:380, 480:600]

#%%
md.numpy_to_nrrd(sample_vol, 'small_kesm.nrrd')

#%%
sample_vol = np.squeeze(sample_vol)
T_blur = sp.ndimage.gaussian_filter(sample_vol, [1, 1, 1])

#marching cubes
verts, faces, _, _ = measure.marching_cubes(T_blur)

#save obj
def save_obj(filename, vertices, faces):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for face in faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

save_obj('test.obj', verts, faces)