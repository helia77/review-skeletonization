# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:11:45 2024

@author: helioum
"""

import numpy as np
import skimage.morphology as mph
import matplotlib.pyplot as plt
import manage_data as md

gr_truth = np.load('C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/segmentation/ground_truth_kesm.npy')

sample_gr = gr_truth[0:100, 300:400, 300:500]

#%%
# both methods works the same
#skelet = mph.skeletonize(sample_gr, method='lee')
skelet_3d = mph.skeletonize_3d(sample_gr)
md.numpy_to_nrrd(skelet_3d, 'C:/Users/helioum/Desktop/lee.nrrd')



