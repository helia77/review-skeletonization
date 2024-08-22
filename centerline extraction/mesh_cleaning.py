# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:44:50 2024

@author: helioum
"""

import manage_data as md
import numpy as np
import pymeshlab as pl
import os, shutil
import skeletonization as skelet
import scipy as sp
import pyvista as pv
import pymeshfix as pf

# split npy volumes, pad, and save as obj files. check if it works
vol_path = 'Data/LSM/npy file/'
list_name = []

for file in os.listdir(vol_path):
    # vol = np.load(os.path.join(vol_path, file))
    print_name = (file.split('.')[0]).split('_')[1]
    list_name.append(print_name)
    
#     for i in range(0, 2):
#         for j in range(0, 2):
#             for k in range(0, 2):
#                 sample = np.uint8(vol[100*i:100*(i+1), 100*j:100*(j+1), 100*k:100*(k+1)])
#                 sample = np.pad(sample, (1,), 'constant', constant_values=(0,))
#                 #np.save('Data/Tagliasacchi/LSM fixed meshs/' + print_name + '/' + print_name + f'_fixed_{i}_{j}_{k}.npy', sample)
#                 md.npy2obj(sample, 'Data/Tagliasacchi/LSM fixed meshs/' + print_name + '/' + print_name + f'_{i}_{j}_{k}.obj')
                
                
                
#%%
obj_path = 'Data/Tagliasacchi/LSM fixed meshs/'
for name in list_name:
    path = obj_path + '/' + name
        
    for vol in [f for f in os.listdir(path) if f.endswith('obj')]:
        
        if os.path.exists(path + '/fixed') and os.path.isdir(path + '/fixed'):
            print('fixed deleted')
            shutil.rmtree(path + '/fixed')
        os.makedirs(path + '/fixed')
        
        mesh = pv.read(os.path.join(path, vol))
        sub_div = mesh.subdivide(1)
        all_bods = sub_div.split_bodies()
        sample_name = vol.split('.')[0]
        # clean each segment and save in folder
        for i in range(all_bods.n_blocks):
            print(i)
            body = all_bods[i]
            ehem = body.extract_surface()
            print('surface extracted.')
            mesh = pf.PyTMesh()
            mesh.load_array(ehem.points, ehem.faces.reshape(-1, 4)[:, 1:])
            print('array loaded.')
            mesh.clean()
            print('mesh cleaned.')
            mesh.save_file(path + '/fixed/' + sample_name  + '_sample_'+ str(i) + '.obj')
        new_files = os.listdir(path + '/fixed')
        meshes = pv.read(path + '/fixed/' + new_files[0])
        for i in range(1, len(os.listdir(path + '/fixed'))):
            mesh = pv.read(path + '/fixed/' + sample_name  + '_sample_'+ str(i) + '.obj')
            meshes.merge(mesh, inplace=True)
        final = pf.PyTMesh()
        final.load_array(meshes.points, meshes.faces.reshape(-1, 4)[:, 1:])
        final.save_file('Data/Tagliasacchi/LSM fixed meshs/' + name + '/' + sample_name + '_fixed.obj')

#%%
name = 'unet'
vol = 'unet_1_1_0.obj'
path = obj_path + '/' + name
if os.path.exists(path + '/fixed') and os.path.isdir(path + '/fixed'):
    print('fixed deleted')
    shutil.rmtree(path + '/fixed')
os.makedirs(path + '/fixed')

mesh = pv.read(os.path.join(path, vol))
sub_div = mesh.subdivide(1)
all_bods = sub_div.split_bodies()
sample_name = vol.split('.')[0]
# clean each segment and save in folder
for i in range(all_bods.n_blocks):
    print(i)
    body = all_bods[i]
    ehem = body.extract_surface()
    print('surface extracted.')
    mesh = pf.PyTMesh()
    mesh.load_array(ehem.points, ehem.faces.reshape(-1, 4)[:, 1:])
    print('array loaded.')
    mesh.clean()
    print('mesh cleaned.')
    mesh.save_file(path + '/fixed/' + sample_name  + '_sample_'+ str(i) + '.obj')
new_files = os.listdir(path + '/fixed')
meshes = pv.read(path + '/fixed/' + new_files[0])
for i in range(1, len(os.listdir(path + '/fixed'))):
    mesh = pv.read(path + '/fixed/' + sample_name  + '_sample_'+ str(i) + '.obj')
    meshes.merge(mesh, inplace=True)
final = pf.PyTMesh()
final.load_array(meshes.points, meshes.faces.reshape(-1, 4)[:, 1:])
final.save_file('Data/Tagliasacchi/LSM fixed meshs/' + name + '/' + sample_name + '_fixed.obj')
#%%

path = 'Data/Tagliasacchi/LSM fixed meshs/unet/'

for file in [f for f in os.listdir(path) if f.endswith('.cg')]:
    md.cg2obj(path + file)
    