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

def pointcloud(self, spacing):
        ls = self.linesegments()
        pc = []
        for l in ls:
            pc = pc + l.pointcloud(spacing)
        return pc

######## convert SDP file to OBJ file for visualization ##########
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

def cost_vote(params):
    # dilate distance of the confidence voxels, threshold in the confidence estimation (included), 
    # the param to consider interval of distances, radius used to compute the accumulation
    dilate, deltaG, radius = params
    print('Dilate={},\tDeltaG={},\tradius={}'.format(round(dilate,4), round(deltaG,4), round(radius,4)))
    start = time.time()
    # run the C++ code on the sample_gr binary .OFF file and obtain the centerline as SDP file
    exe_file = 'CDCVAM/build/bin/Release/centerLineGeodesicGraph.exe'
    output_name = "sample_gr"
    if os.path.isfile(output_name+'Vertex.sdp'):
        os.remove(output_name+'Vertex.sdp')
        os.remove(output_name+'Edges.sdp')
    command = [
        exe_file,
        '-i', 'small_gr_fixed_for_test.off',
        "-o", output_name,
        "--dilateDist", str(dilate),
        "-g", str(deltaG),
        "-R", str(radius),
        "-t", "0.5"
    ]
    
    # Execute the command
    subprocess.run(command, check=True)
    
    # now we have SDP files saved in the directory
    sdp2obj(output_name)
    gr_skeleton = md.nrrd_to_numpy('centerline.nrrd')
    T = md.NWT(output_name+'_centerline.obj')
    GR = np.argwhere(gr_skeleton).astype(np.float32)
    
    sigma = 0.01
    subdiv = 2
    # generate point clouds representing both networks
    P_T = np.array(T.pointcloud(sigma/subdiv))
    P_T /= 99.0
    GR[:, 0] /= 1023.0
    GR[:, 1:3] /= 511.0
    
    # generate KD trees for each network
    T_tree = sp.spatial.cKDTree(P_T)
    GT_tree = sp.spatial.cKDTree(GR)
    
    # query each KD tree to get the corresponding geometric distances
    [T_dist, _] = GT_tree.query(P_T)
    [GT_dist, _] = T_tree.query(GR)
    
    # convert distances to Gaussian metrics
    threshold = 0.5
    T_metric = np.exp(-0.5 * (T_dist ** 2 / sigma ** 2))
    T_metric[T_metric > threshold] = 1
    T_metric[T_metric <= threshold] = 0
    
    GT_metric = np.exp(-0.5 * (GT_dist ** 2 / sigma ** 2))
    GT_metric[GT_metric > threshold] = 1
    GT_metric[GT_metric <= threshold] = 0
    
    # we try to minimize the False Negative and Positive Rates combined based on the input parameters
    FNR = 1 - np.mean(GT_metric)
    FPR = 1 - np.mean(T_metric)
    print('FNR={}, FPR={}, \tTOTAL={}\t time:{} mins'.format(FNR, FPR, FNR+FPR, (time.time()-start)/60.0))
    return FNR+FPR
    
    
initial_params = (2, 3, 5)
start = time.time()
#result_b = fmin_powell(cost_vote, initial_params)
result = opt.minimize(cost_vote, initial_params, method="Powell", bounds=((0.5, 10), (0.5, 10), (0.5, 5)))
print('Took {} minutes.'.format((time.time() - start)/60.0))

#%%
print('dialte:\t', result[0])
print('deltaG:\t', result[1])
print('radius:\t', result[2])
print('thresh:\t', result[2])
#%%
output_name = "sample_gr"
dilate, radius = [2, 10]
exe_file = 'CDCVAM/build/bin/Release/centerLineGeodesicGraph.exe'
command = [
    exe_file,
    '-i', 'sample_gr.off',
    "-o", output_name,
    "--dilateDist", '2',
    "-g", '3',
    "-R", '10',
    "-t", '0.5'
]

# Execute the command
start = time.time()
subprocess.run(command, check=True)
print('Took {} minutes.'.format((time.time() - start)/60.0))