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
import matplotlib.pyplot as plt
import skeletonization as skelet

#%%
# load binary input
gr_truth = np.load('C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/thinning/ground_truth_kesm.npy')
sample_gr = gr_truth[0:200, 200:400, 300:500]

# apply all centerline extraction methods
start = time.time()
lee = thinning.skelet_lee(sample_gr)                                                # numpy array
print('Lee\'s method took:\t {}'.format((time.time() - start)/60.0))
#%%
start = time.time()
palagyi = thinning.skelet_palagyi(sample_gr)                                        # numpy array
print('Palagyi\'s method took:\t {}'.format((time.time() - start)/60.0))

#%%
exe_path = 'CDCVAM/build/bin/Release/centerLineGeodesicGraph.exe'
input_name = 'sample_gr.off'
output_name = 'kerautret'
start = time.time()
#Dilate=1.4625,	DeltaG=3.8105,	radius=2.3025
skelet.skelet_kerautret(exe_path, input_name, output_name, dilateDist=1.4625, deltaG=3.8105, radius=2.3025, threshold=0.5)
print('Kerautret\'s method took:\t {}'.format((time.time() - start)/60.0))
kerautret = md.NWT(output_name+'.obj')                                              # NWT file
#%%
start = time.time()
# Dilate=1.4625,	DeltaG=3.8105,	radius=2.3025
kline, _ = skelet.skelet_kline(sample_gr, [95, 76, 35], dist_map_weight = 19.25,       # numpy array
                cluster_graph_weight=1518.45, min_branch_length = 6, min_branch_to_root = 10.11)
print('Kline\'s method took:\t {}'.format((time.time() - start)/60.0))
#%%
# Tagliasacchi's method (obtained using the software)
tagliasacchi = md.NWT('Output Data/skeleton_tag.obj')
antiga = md.NWT('Model_1.obj')

#%%
skeleton_results = [lee, 'Lee\'s', palagyi, 'Palagyi\'s', kline, 'Kline\'s',
                    kerautret, 'Kerautret\'s', 
                    antiga, 'Antiga\'s', tagliasacchi, 'Tagliasacchi\'s']
#%%
# Save them all as different formats for visualization
for i in range(0, len(skeleton_results), 2):
    skeleton = skeleton_results[i]
    name = skeleton_results[i + 1]
    
    # save obj and nrrd
    if isinstance(skeleton, np.ndarray):
        md.npy2obj(skeleton, name+'.obj')
        md.numpy_to_nrrd(skeleton, name+'.nrrd')
    elif isinstance(skeleton, md.NWT):
        skeleton.save_obj(name+'.obj')
    
#%%
# Get the FPR and FNR rates using NetMets metric
sigma = 0.01
threshold = 0.5
subdiv = 1

gr_skeleton = md.nrrd_to_numpy('centerline.nrrd') #md.NWT('centerline.obj')
GR = np.argwhere(gr_skeleton).astype(np.float32)
GR[:, 0] /= 1023.0
GR[:, 1:3] /= 511.0
GT_tree = sp.spatial.cKDTree(GR)
#%%
for i in range(0, len(skeleton_results), 2):
    skeleton = skeleton_results[i]
    print(type(skeleton))
    name = skeleton_results[i + 1]
    print('first part done.')
    if isinstance(skeleton, np.ndarray):
        P_T = np.argwhere(skeleton).astype(np.float32)
    else:
        P_T = np.array(skeleton.pointcloud(sigma/subdiv))
    print('2nd part done.')
    if name == 'Tagliasacchi\'s':
        P_T /= 99.0
    else:
        P_T /= 199.0
    print('3rd part done.')
    T_tree = sp.spatial.cKDTree(P_T)
    print('4th part done.')
    # Query each KD tree to get the corresponding geometric distances
    [T_dist, _] = GT_tree.query(P_T)
    [GT_dist, _] = T_tree.query(GR)
    
    # convert distances to Gaussian metrics
    T_metric = np.exp(-0.5 * (T_dist ** 2 / sigma ** 2))
    T_metric[T_metric > threshold] = 1
    T_metric[T_metric <= threshold] = 0
    GT_metric = np.exp(-0.5 * (GT_dist ** 2 / sigma ** 2))
    GT_metric[GT_metric > threshold] = 1
    GT_metric[GT_metric <= threshold] = 0

    #calculate the FPR and FPR
    print('\nMethod - ', name)
    print("FNR = " + str(1 - np.mean(GT_metric)))
    print("FPR = " + str(1 - np.mean(T_metric)))
    
    plt.figure(i/2)
    plt.suptitle(name + ' Method')
    plt.subplot(1, 2, 1)
    #plt.scatter(P_GT[:, 0], P_GT[:, 1], s=sigma*shadow, c="grey")
    plt.scatter(P_T[:, 0], P_T[:, 1], c=1 - T_metric, cmap = "RdYlBu_r")
    plt.title("Test Case Network and Metric")

    plt.subplot(1, 2, 2)
    plt.scatter(GR[:, 0], GR[:, 1], c=1 - GT_metric, cmap = "RdYlBu_r")
    plt.title("Ground Truth Network and Metric")
    plt.show()

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


