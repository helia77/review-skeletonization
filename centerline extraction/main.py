# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:02:59 2024

@author: helioum
"""

import kline as kl
import numpy as np
import time
import manage_data as md
#%%
# load binary input
gr_truth = np.load('C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/thinning/ground_truth_kesm.npy')
sample_gr = gr_truth[0:100, 200:300, 300:400]

#%%
start = time.time()
centerline, _ = kl.kline_vessel(sample_gr, [95, 76, 35], dist_map_weight = 15, cluster_graph_weight=15,
                                min_branch_length = 25, min_branch_to_root = 3)
print('Took {} minutes'.format((time.time() - start)/60))
#%%
md.numpy_to_nrrd(centerline, "sample_kline.nrrd")
#md.numpy_to_nrrd(sample_gr, "sample_gr.nrrd")
