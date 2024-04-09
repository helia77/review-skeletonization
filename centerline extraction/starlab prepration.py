# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:42:11 2024

@author: helioum
"""

import numpy as np
from skimage import measure
import pymeshlab as ml
import skeletonization as skelet
import manage_data as md
import pymeshfix

#%%
###### convert binary numpy 3D volume to .OFF format file
sample_gr = md.nrrd_to_numpy('ground.nrrd')
md.npy2obj(sample_gr, 'sample_gr.obj')
pymeshfix.clean_from_file('sample_gr.obj', 'sample_obj_fixed.obj')
kline, _ = skelet.skelet_kline(sample_gr, [125, 109, 199], dist_map_weight = 19.25,       # numpy array
        cluster_graph_weight=1500, min_branch_length = 10, min_branch_to_root = 15)

#%%
md.npy2obj(kline, 'kline.obj')
#%%
# downsample the original volume to smaller size and clean boundries
ehem = md.resample_volume(sample_gr, [2, 2, 2])
verts, faces, _, _ = measure.marching_cubes(ehem, level=0.0)
md.save_obj('small_gr.obj', verts, faces)
pymeshfix.clean_from_file('smaall_gr.obj', 'smaall_gr_fixed.obj')


#%%
gr_truth = np.load('C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/segmentation/ground_truth_kesm.npy')
sample_gr = gr_truth[0:200, 200:400, 300:500]

np.save('../segmentation/U-Net/CNN/unprocessed_data/sample_gr.npy', sample_gr)

volume = np.load('C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/segmentation/whole_volume_kesm.npy')
np.save('../segmentation/U-Net/CNN/unprocessed_data/volume.npy', volume[0:200, 200:400, 300:500])

#%%

md.npy2nrrd(np.load('../segmentation/U-Net/CNN/data/KESM/volume_ground_truth.npy'), '../segmentation/U-Net/CNN/data/KESM/gr.nrrd')



