# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:56:02 2024

@author: helioum
"""

import numpy as np
import functions as func

n26 =  np.array([[[1,1,1],[1,1,1],[1,1,1]],
                 [[1,1,1],[1,0,1],[1,1,1]], 
                 [[1,1,1],[1,1,1],[1,1,1]]], dtype=np.uint8);

test = np.array([[[1,2,3],[4, 5, 6],[7, 8, 9]],
                 [[10,11,12],[13,14,15],[16,17,18]], 
                 [[19,20,21],[22,23,24],[26,27,28]]], dtype=np.uint8);

#%%

def centerline_to_obj(binary):
    
    vertices = np.argwhere(binary)
    all_lines = []
    
    # the 26-neighborhood has either 1, 2, or 3 connected points
    point_neighbors = func.convolution_3d(binary, func.n26, vertices)
    branch_points = point_neighbors == 3
    # perform tracing from the first vertix
    line = []
    for i, v in enumerate(vertices):
        
        neighbor = point_neighbors[i]
        x0, y0, z0 = v
        x1, y1, z1 = vertices[i+1]
        
        # check if in the n-26 neighborhood
        if abs(x1-x0) <= 1 and abs(y1- y0) <= 1 and abs(z1-z0) <= 1:
            if len(line) == 0:
                line.append([x0, y0, z0])
                line.append([x1, y1, z1])
            else:
                line.append([x1, y1, z1])
        
        
    
def numpy_obj(binary):
    
    vertices = np.argwhere(binary)
    edges = [i+1 for i in range(len(vertices)-1)]
    
    # save obj file
    with open('binary.obj', 'w') as f:
        for v in vertices:
            #edges = list(range(1, len(verts[i]) + 1))
            #f.write("# new line\n")
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                
        f.write("l ") 
        for e in edges:
            f.write(f"{e} ")
        f.write("\n")