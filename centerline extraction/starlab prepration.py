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
import pymeshfix as pf
import os
import scipy as sp
import pyvista as pv

SEG = 'bfrangi'
DATA = 'LSM'
#%%
# apply padding then marching cubes
path = '../segmentation/Data/Segmentations/' + DATA + '/binary_BFrangi_otsu3d.npy'
file = np.uint8(np.load(path))
pad = np.pad(file, (1,), 'constant', constant_values=(0,))
md.npy2obj(pad, './Data/' + DATA + '/obj file/binary_' + SEG + '_padded.obj')

#%%
# read the fixed file (in MeshLab) and divide it into individual segments
path = 'Data/' + DATA + '/obj file/binary_' + SEG + '_fixed.obj'
mesh = pv.read(path)
sub_div = mesh.subdivide(1)
all_bods = sub_div.split_bodies()

#%%
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
    mesh.save_file('Data/' + DATA + '/obj file/' + SEG + '/sample_'+ str(i) + '.obj')
    
#%%
# merge all the cleaned obj files
path = 'Data/' + DATA + '/obj file/' + SEG
file_num = len(os.listdir(path))
meshes = pv.read(path + '/sample_0.obj')
for i in range(1, 18):
    print(i)
    mesh = pv.read(path + '/sample_' + str(i) + '.obj')
    meshes.merge(mesh, inplace=True)

#%%
final = pf.PyTMesh()
final.load_array(meshes.points, meshes.faces.reshape(-1, 4)[:, 1:])
final.save_file('Data/' + DATA + '/obj file/' + SEG + '/all_' + SEG + '.obj')

#%%
# convert cg skeletons to obj for merge
md.cg2obj('Data/Tagliasacchi/Micro Output/tagliasacchi_' + SEG + '2.cg')

#%%
# fix the surface mesh files for Starlab software (Tagliasacchi method)
for i in [inp for inp in os.listdir('Data/LSM/obj file/')]:
    name = (i.split('.')[0]).split('_')[1]
    print(name)
    pf.clean_from_file('Data/LSM/obj file/'+ i, 'Data/Tagliasacchi/LSM fixed meshs/binary_' + name + '_fixed.obj')

#%%
# convert cg file (output skeleton type of Starlab software) to obj
path = 'Data/Tagliasacchi/KESM Output/'
input_center = [inp for inp in os.listdir(path)]
for center in input_center:
    md.cg2obj(path + center)


