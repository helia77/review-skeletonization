# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:04:39 2024

From paper:     Palagyi & Kuba, A Parallel 3D 12-Subiteration Thinning Algorithm,
                Graphical Models and Image Processing 61, 199-221 (1999)

@author: helioum - based on the code of ClearMap2
"""

import numpy as np
import functions as func
import manage_data as md
import time
import skimage.morphology as mph
#%%
# load binary input
gr_truth = np.load('C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/segmentation/ground_truth_kesm.npy')
sample_gr = gr_truth[0:200, 200:400, 300:500]

#%%
# ----------------- Lee et al ------------------------- #
# both methods works the same
#skelet = mph.skeletonize(sample_gr, method='lee')
skelet_3d = mph.skeletonize_3d(sample_gr)
md.numpy_to_nrrd(skelet_3d, 'C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/thinning/lee.nrrd')

#%%
# ----------------- Palagyi et al ------------------------- #
# the indices of black points in the input
#points = sample_gr.nonzero()
points_arg = np.argwhere(sample_gr)

binary = np.copy(sample_gr)
deletable = np.load('PK12.npy')
rotations = func.rotations12(func.cube_base())
#death = np.zeros(binary.shape, dtype = 'uint16')
#md.numpy_to_nrrd(sample_gr, 'C:/Users/helioum/Desktop/sample_gr.nrrd')
#%%
''' border point in the paper:
        a black point is a border point if its N_6 neighborhood has at least one white point'''
since = time.time()        
while True:
    # find which black points are border points (the index of the border varibale is the same as black points in point_arg)
    border = func.convolution_3d(binary, func.n6, points_arg) < 6
    border_points = points_arg[border]
    border_ids = np.nonzero(border)[0]
    keep = np.ones(len(border), dtype=bool)
    
    iterat = 0
    for i in range(12):         # 12 sub-iterations
    
        removables = func.convolution_3d(binary, rotations[i], border_points)
        rem_borders = deletable[removables]
        rem_points = border_points[rem_borders]
        
        binary[rem_points[:, 0], rem_points[:, 1], rem_points[:, 2]] = 0
        keep[border_ids[rem_borders]] = False
        
        iterat += len(rem_points)
        print(i, '\t-  Deleted points: %d' % (len(rem_points)))
    
    print('Total deletated points:\t', iterat, '\n')
    
    # update foreground
    points_arg = points_arg[keep]
    if iterat == 0:
        print('No more point removal.')
        break

print('Took ', (time.time() - since), ' seconds')
#%%
#md.binary_to_obj(binary, 'binary.obj')
md.numpy_to_nrrd(binary, 'C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/thinning/palagyi.nrrd')





       