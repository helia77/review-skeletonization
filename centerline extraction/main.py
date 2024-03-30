# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:02:59 2024

@author: helioum
"""
import time
import thinning
import numpy as np
import scipy as sp
import manage_data as md
import skeletonization as skelet

#%%
# load binary input
gr_truth = np.load('C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/thinning/ground_truth_kesm.npy')
sample_gr = gr_truth[0:100, 200:300, 300:400]

# apply all centerline extraction methods
start = time.time()
lee = thinning.skelet_lee(sample_gr)                                                # numpy array
print('Lee\'s method took:\t {}'.format((time.time() - start)/60.0))
 
start = time.time()
palagyi = thinning.skelet_palagyi(sample_gr)                                        # numpy array
print('Palagyi\'s method took:\t {}'.format((time.time() - start)/60.0))


exe_path = 'CDCVAM/build/bin/Release/centerLineGeodesicGraph.exe'
input_name = 'sample_gr.off'
output_name = 'sample_gr'
start = time.time()
skelet.skelet_kerautret(exe_path, input_name, output_name)
print('Kerautret\'s method took:\t {}'.format((time.time() - start)/60.0))
kerautret = md.NWT(output_name+'_centerline.obj')                                   # NWT file

start = time.time()
kline = skelet.skelet_kline(sample_gr, [95, 76, 35], dist_map_weight = 19.25,       # numpy array
                cluster_graph_weight=1518.45, min_branch_length = 5, min_branch_to_root = 10.11)
print('Kline\'s method took:\t {}'.format((time.time() - start)/60.0))

# Tagliasacchi's method (obtained using the software)
tagliasacchi = md.NWT('Output Data/skeleton.obj')

#%%
# Get the FPR and FNR rates using NetMets metric

skeleton_results = [lee, palagyi, kline]                    # numpy ones
gr_skeleton = md.nrrd_to_numpy('centerline.nrrd') #md.NWT('centerline.obj')
GR = np.argwhere(gr_skeleton).astype(np.float32)
GR[:, 0] /= 1023.0
GR[:, 1:3] /= 511.0
GT_tree = sp.spatial.cKDTree(GR)

for skeleton in skeleton_results:
    skeleton_pixels = np.argwhere(skeleton).astype(np.float32)

sigma = 10
subdiv = 4
#%%
'''
Results from optimizing a cost function based on NetMets metric (FPR + FNR)
dmp=19.25196951467709, cgw=1518.45, mbl=5.0, mbtr=10.1102224935092
Number of centerline voxels: 1847
Rate:  0.00018516065119977831
'''
# start = time.time()
# centerline, _ = kl.kline_vessel(sample_gr, [95, 76, 35], dist_map_weight = 19.25, cluster_graph_weight=1518.45,
#                                 min_branch_length = 5, min_branch_to_root = 10.11)
# print('Took {} minutes'.format((time.time() - start)/60))
# md.numpy_to_nrrd(centerline, "sample_kline.nrrd")


