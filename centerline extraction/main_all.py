# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:00:14 2024

@author: helioum
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:02:59 2024

@author: helioum
"""
import os
import sys
import time
import numpy as np
import scipy as sp
sys.path.append("../")
import thinning as thin
import manage_data as md
import matplotlib.pyplot as plt
import skeletonization as skelet

#%%
SAVE = True
DATA = 'LSM'                     # change the dataset name

# segmented datasets directory
segments = {"KESM": '../segmentation/Data/Segmentations/KESM/', 
            "LSM": '../segmentation/Data/Segmentations/LSM/', 
            "Micro": '../segmentation/Data/Segmentations/Micro/'}

# groung truth of the centerline
ground_centers = {"KESM": 'centerline.nrrd', 
           "LSM": 'Data/LSM/new/lsm_brain_centerline.nrrd', #'lsm_centerline_ground.nrrd', 
           "Micro": 'Data/Micro/micro_centerlines_GR_float32.npy'}

#%%
# load all segmented data
list_name = [name for name in os.listdir(segments[DATA]) if name.split('_')[0] == 'binary' and 
             ((name.split('_')[-1]).split('.'))[0] != 'otsu2d' and len(name.split('_')) < 4]
list_name.append('binary_otsu2d.npy')

#%%
# apply all centerline extraction methods
lee_all = {}
start = time.time()
for name in list_name:
    print_name = (name.split('_')[1]).split('.')[0]
    print(print_name, 'loading...', end='\t')
    volume = np.load(os.path.join(segments[DATA], name))
    lee = thin.skelet_lee(volume)    # numpy array
    lee_all[print_name] = lee
    print('Done')
print('Lee\'s method took:\t {}'.format((time.time() - start)/60.0))

#%%
palagyi_all = {}
start = time.time()
for name in list_name:
    print_name = (name.split('_')[1]).split('.')[0]
    print(print_name, 'loading...', end='\t')
    volume = np.load(os.path.join(segments[DATA], name))
    palagyi = thin.skelet_palagyi(volume)   # numpy array
    palagyi_all[print_name] = palagyi
    print('Done')
print('Palagyi\'s method took:\t {}'.format((time.time() - start)/60.0))

 #%%
# --- uncomment to create obj from npy to work with Kerautret's method (then convert to .off file online)
for name in list_name:
    print(name.split('_')[1], ' converting to obj...')
    volume = np.load(os.path.join(segments[DATA], name))
    md.npy2obj(volume, 'Data/'+DATA+'/obj file (large)/'+name.split('.')[0]+'.obj')
    
#%%
# Kerautret's method
off_path = 'Data/'+ DATA +'/off file/'
exe_path = 'CDCVAM/build/bin/Release/centerLineGeodesicGraph.exe'
output_name = 'kerautret_' + DATA
params = {"KESM": [1.0, 3.3, 1.1, 0.5], "LSM": [1, 3.1, 0.7, 0.5], "Micro": [1.1, 2.75, 0.8, 0.6]}
dil, delta, rad, th = params[DATA]
kerautret_all = {}
start = time.time()
for name in list_name:
    print_name = (name.split('_')[1]).split('.')[0]
    input_name = off_path + '/binary_' + print_name +'.off'
    print(print_name, 'loading...')
    skelet.skelet_kerautret(exe_path, input_name, output_name, dilateDist=dil, deltaG=delta, radius=rad, threshold=th)
    kerautret = md.NWT(output_name+'.obj')              # NWT file
    kerautret_all[print_name] = kerautret
print('Kerautret\'s method took:\t {}'.format((time.time() - start)/60.0))

#%%
kline_path = {"KESM": 'Data/Kline/KESM/npy files/', "LSM": 'Data/Kline/LSM/npy files/', "Micro": 'Data/Kline/Micro/npy files/'}
kline_all = {}
for file in [f for f in os.listdir(kline_path[DATA]) if f.startswith('kline')]:
    method = (file.split('_')[-1]).split('.')[0]
    print(method)
    kline = np.load(kline_path[DATA] + file)
    kline_all[method] = kline

#%%
# Tagliasacchi's method (obtained using the implemented software)
taglia_all = {}
taglia_path = {"KESM": 'Data/Tagliasacchi/KESM Output/', "LSM": 'Data/Tagliasacchi/LSM Output/',
               "Micro": 'Data/Tagliasacchi/Micro Output/'}
files = [centers for centers in os.listdir(taglia_path[DATA]) if centers.endswith('unet.obj')]
for file in files:
    method = (file.split('_')[-1]).split('.')[0]
    print(method)
    taglia = md.NWT(taglia_path[DATA] + file)
    taglia_all[method] = taglia

# *** ------------ NOTE -------------- ***
#   Make sure you sub-sample the point cloud (P_T) by 201.0
#   Because the volumes were padded with zero and are in shape 202 cube
#%%
# Antiga's method (obtained using Slicer3D VMTK library)
antiga_all = {}
antiga_path = {"KESM": 'Data/Antiga/obj files/KESM/', "LSM": 'Data/Antiga/obj files/LSM/', "Micro": 'Data/Antiga/obj files/Micro/'}

for file in [f for f in os.listdir(antiga_path[DATA]) if f.startswith('antiga')]:
    method = (file.split('_')[-1]).split('.')[0]
    print(method)
    antiga = md.NWT(antiga_path[DATA] + file)
    antiga_all[method] = antiga

#%%
# Get the FPR and FNR rates using NetMets metric
sigma = 0.01
threshold = 0.5
subdiv = 1

if ground_centers[DATA].endswith('nrrd'):
    gr_skeleton = md.nrrd2npy(ground_centers[DATA])
    GR = np.argwhere(gr_skeleton).astype(np.float32)
elif ground_centers[DATA].endswith('obj'):
    gr_skeleton = md.NWT(ground_centers[DATA])
    GR = np.array(gr_skeleton.pointcloud(sigma/subdiv))
elif DATA == 'Micro' and ground_centers[DATA].endswith('npy'):
    print('.')
    GR = np.load(ground_centers[DATA])
else:                                                               # numpy file
    gr_skeleton = np.load(ground_centers[DATA])
    GR = np.argwhere(gr_skeleton).astype(np.float32)
 
#%%
if DATA == 'KESM':    
    GR[:, 0] /= 1023.0
    GR[:, 1:3] /= 511.0
else:
    GR /= 199.0
GT_tree = sp.spatial.cKDTree(GR)

#%%
for name, skeleton in kerautret_all.items():
    if isinstance(skeleton, np.ndarray):
        P_T = np.argwhere(skeleton).astype(np.float32)
    else:
        print('obj file loaded')
        P_T = np.array(skeleton.pointcloud(sigma/subdiv))
    
    print('Point cloud done.')
    P_T /= 199.0
    T_tree = sp.spatial.cKDTree(P_T)
    
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
    print('\nMethod:', name.split('.')[0])
    print(f"Precision = {np.mean(T_metric):.3f}")
    print(f"Recall = {np.mean(GT_metric):.3f}")

#%%
plt.figure()
plt.suptitle(' Method')
plt.subplot(1, 2, 1)
#plt.scatter(P_GT[:, 0], P_GT[:, 1], s=sigma*shadow, c="grey")
plt.scatter(P_T[0:10000, 0], P_T[0:10000, 1], c=1 - T_metric[0:10000], cmap = "RdYlBu_r")
plt.title("Test Case Network and Metric")

plt.subplot(1, 2, 2)
plt.scatter(GR[0:10000, 0], GR[0:10000, 1], c=1 - GT_metric[0:10000], cmap = "RdYlBu_r")
plt.title("Ground Truth Network and Metric")
plt.show()

#%%


method = 'Palagyi'
save_path = 'Data/'+ method + '/' + DATA + '_'
for name, output in palagyi_all.items():
    if isinstance(output, np.ndarray):
        np.save(save_path + name + '.npy', output)
    else:
        output.save_obj(save_path + name + '.obj')
