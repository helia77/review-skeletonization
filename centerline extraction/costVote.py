# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:25:08 2024

@author: helioum
"""
import os
import numpy as np
import time
import manage_data as md
import scipy as sp
from scipy.optimize import fmin_powell
import scipy.optimize as opt
import subprocess
import skeletonization as skelet


#%%
def pointcloud(self, spacing):
        ls = self.linesegments()
        pc = []
        for l in ls:
            pc = pc + l.pointcloud(spacing)
        return pc

######## convert SDP file to OBJ file ##########
def sdp2obj(output_name):
    # read vertex file
    with open(output_name+'Vertex.sdp', 'r') as inputfile:
        vlines = inputfile.readlines()
        
    # read edges file
    with open(output_name+'Edges.sdp', 'r') as inputfile:
        enum = inputfile.readlines()
        
    # write in the obj file
    with open(output_name+'_centerline.obj', 'w') as outputfile:
        outputfile.write('# vertices\n')
        for line in vlines:
            outputfile.write('v '+ line.strip() + '\n')
        
        outputfile.write('\n# edges\n')
        for edges in enum:
            num1, num2 = edges.strip().split(' ')
            num1 = int(num1)
            num2 = int(num2)
            outputfile.write('l '+ str(num1+1) + ' ' + str(num2+1) + '\n')

def kline_vote(gr_segment, GT_tree):
    start_ids = [[23, 76, 199], [8, 199, 162], [54, 199, 17], [8, 0, 133], [183, 0, 196],
    	[60, 94, 199]] #, [159, 32, 0], [53, 199, 64]]
    result = np.zeros_like(gr_segment)
    for i, start in enumerate(start_ids):
        kline, _ = skelet.skelet_kline(gr_segment, start, dist_map_weight=6, min_branch_to_root=10)
        result = np.logical_or(kline, result)
        name = str(i)
        print('saving', name, '...')
        #md.npy2nrrd(kline, 'kline_filename_'+name+'.nrrd')
        #md.npy2obj(kline, 'kline_'+name+'.obj')
    
    # md.npy2nrrd(result, 'kline_filename.nrrd')
    # P_T = np.argwhere(kline).astype(np.float32)/199.0
    
    # T_tree = sp.spatial.cKDTree(P_T)
    # # Query each KD tree to get the corresponding geometric distances
    # [T_dist, _] = GT_tree.query(P_T)
    # [GT_dist, _] = T_tree.query(GR)
    # sigma = 0.01
    # threshold = 0.5
    # # convert distances to Gaussian metrics
    # T_metric = np.exp(-0.5 * (T_dist ** 2 / sigma ** 2))
    # T_metric[T_metric > threshold] = 1
    # T_metric[T_metric <= threshold] = 0
    # GT_metric = np.exp(-0.5 * (GT_dist ** 2 / sigma ** 2))
    # GT_metric[GT_metric > threshold] = 1
    # GT_metric[GT_metric <= threshold] = 0
    # recall = np.mean(GT_metric)
    # precision = np.mean(T_metric)
    # fscore = (2*precision*recall)/(precision+recall)
    # print(f'Recall={recall:.3f}, Precision={precision:.3f}, \tFscore={fscore:.3f}')
    return result

#%%
def cost_vote(dilate, deltag, radius, th, GT_tree):
    # dilate distance of the confidence voxels, threshold in the confidence estimation (included), 
    # the param to consider interval of distances, radius used to compute the accumulation
    print('Dilate={},\tDeltaG={},\tRadius={},\tThresh={}'.format(round(dilate,4), round(deltag,4), round(radius, 4), round(th, 2)))
    # run the C++ code on the sample_gr binary .OFF file and obtain the centerline as SDP file
    exe_file = 'CDCVAM/build/bin/Release/centerLineGeodesicGraph.exe'
    output_name = "lsm_gr_test"
    if os.path.isfile(output_name+'Vertex.sdp'):
        os.remove(output_name+'Vertex.sdp')
        os.remove(output_name+'Edges.sdp')
    command = [
        exe_file,
        '-i', 'Data/LSM/lsm_brain_gr_100x100.off',
        "-o", output_name,
        "--dilateDist", str(dilate),
        "-g", str(deltag),
        "-R", str(radius),
        "-t", str(th)
    ]
    
    # Execute the command
    subprocess.run(command, check=True)
    
    # now we have SDP files saved in the directory
    sdp2obj(output_name)
    T = md.NWT(output_name+'_centerline.obj')
      
    # generate point clouds representing both networks
    P_T = np.array(T.pointcloud(0.1))/199.0
    print('Point clouds')
    # generate KD trees for each network
    T_tree = sp.spatial.cKDTree(P_T)
    print('T tree')
    # query each KD tree to get the corresponding geometric distances
    [T_dist, _] = GT_tree.query(P_T)
    [GT_dist, _] = T_tree.query(GR)
    print('Query')
    
    # convert distances to Gaussian metrics
    threshold = 0.5
    sigma = 0.01
    T_metric = np.exp(-0.5 * (T_dist ** 2 / sigma ** 2))
    T_metric[T_metric > threshold] = 1
    T_metric[T_metric <= threshold] = 0
    print('T Metrics')
    GT_metric = np.exp(-0.5 * (GT_dist ** 2 / sigma ** 2))
    GT_metric[GT_metric > threshold] = 1
    GT_metric[GT_metric <= threshold] = 0
    print('G Metrics')
    # we try to minimize the False Negative and Positive Rates combined based on the input parameters
    recall = np.mean(GT_metric)
    precision = np.mean(T_metric)
    fscore = (2*precision*recall)/(precision+recall)
    print(f'Recall={recall:.3f}, Precision={precision:.3f}, \tFscore={fscore:.3f}\n')
    return fscore

#%%
# for KESM data
# gr_skeleton = md.nrrd2npy('centerline.nrrd')
# GR = np.argwhere(gr_skeleton).astype(np.float32)
# GR[:, 0] /= 1023.0
# GR[:, 1:3] /= 511.0
# GT_tree = sp.spatial.cKDTree(GR)
# gr_segment = np.load('../segmentation/sample_gr.npy')

# for Micro data
# GR = np.load('Data/Micro/micro_centerlines_GR_float32.npy')
# GR /= 199.0
# GT_tree = sp.spatial.cKDTree(GR)
# gr_segment = np.load('../segmentation/Data/Micro-CT/sample_gr_micro.npy')

# for LSM data
gr_skeleton = md.nrrd2npy('Data/LSM/new/lsm_brain_centerline.nrrd')
GR = np.argwhere(gr_skeleton).astype(np.float32)
GR /= 199.0
GT_tree = sp.spatial.cKDTree(GR)
gr_segment = np.load('Data/LSM/lsm_brain_gr.npy')

#%%
# perform optimizing Kerautret's method
dilate = np.linspace(1, 1.5, 3)
delta = np.linspace(2.75, 3.5, 3)
radius = np.linspace(0.7, 1.5, 3)
thresh = [0.5]
best = 0
#%%
for dil in dilate:
    for dl in delta:
        for rad in radius:
            for th in thresh:
                fscore = cost_vote(dil, dl, rad, th, GT_tree)
                if fscore > best:
                    best = fscore
                    best_param = [dil, dl, rad, th]
                    
#%%
#ehem = cost_vote(1.1, 2.75, 0.8, 0.6, GT_tree)
#%%
result = kline_vote(gr_segment, GT_tree)

#%%
md.npy2obj(np.uint8(result), 'Data/Kline/kline_ground_lsm.obj')
P_T = np.argwhere(result).astype(np.float32)/199.0

T_tree = sp.spatial.cKDTree(P_T)
# Query each KD tree to get the corresponding geometric distances
[T_dist, _] = GT_tree.query(P_T)
[GT_dist, _] = T_tree.query(GR)
sigma = 0.01
threshold = 0.5
# convert distances to Gaussian metrics
T_metric = np.exp(-0.5 * (T_dist ** 2 / sigma ** 2))
T_metric[T_metric > threshold] = 1
T_metric[T_metric <= threshold] = 0
GT_metric = np.exp(-0.5 * (GT_dist ** 2 / sigma ** 2))
GT_metric[GT_metric > threshold] = 1
GT_metric[GT_metric <= threshold] = 0
recall = np.mean(GT_metric)
precision = np.mean(T_metric)
fscore = (2*precision*recall)/(precision+recall)
print(f'Recall={recall:.3f}, Precision={precision:.3f}, \tFscore={fscore:.3f}')

#%%
# check if all start IDs are 1 in all segmentations
start_ids = [[23, 76, 199], [8, 199, 162], [54, 199, 17], [8, 0, 133], [183, 0, 196], [60, 94, 199]]
list_name = ['binary_bfrangi_otsu3d.npy', 'binary_frangi_otsu3d.npy', 'binary_oof.npy', 'binary_otsu3d.npy', 'binary_otsu2d.npy', 'binary_unet.npy']

for name in list_name:
    vol = np.load('../segmentation/Data/Segmentations/LSM/' + name)
    print_name = (name.split('_')[1]).split('.')[0]
    # print(print_name)
    for start in start_ids:
        i, j, k = start
        if vol[i, j, k] == 0:
            print('Zero at', start, 'in ', print_name)
    

#%%
results = np.zeros((200, 200, 200), dtype=np.uint8)
files = [file for file in os.listdir('Data/Kline/npy files/')]
for file in files:
    centerline = np.load('Data/Kline/npy files/' + file)
    results = np.logical_or(centerline, results)
#np.save('Data/Kline/final_kline.npy', np.asarray(results))
#%%
results = np.load('Data/Kline/test/test_6_15.npy')
P_T = np.argwhere(results).astype(np.float32)/199.0

T_tree = sp.spatial.cKDTree(P_T)
# Query each KD tree to get the corresponding geometric distances
[T_dist, _] = GT_tree.query(P_T)
[GT_dist, _] = T_tree.query(GR)
sigma = 0.01
threshold = 0.5
# convert distances to Gaussian metrics
T_metric = np.exp(-0.5 * (T_dist ** 2 / sigma ** 2))
T_metric[T_metric > threshold] = 1
T_metric[T_metric <= threshold] = 0
GT_metric = np.exp(-0.5 * (GT_dist ** 2 / sigma ** 2))
GT_metric[GT_metric > threshold] = 1
GT_metric[GT_metric <= threshold] = 0
recall = np.mean(GT_metric)
precision = np.mean(T_metric)
fscore = (2*precision*recall)/(precision+recall)
print(f'Recall={recall:.4f}, Precision={precision:.4f}, \tFscore={fscore:.4f}')



