# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:57:43 2024

required function fo Palagyi 12 subiteration skeletonization algorithm

@author: helioum
"""
import numpy as np
import torch
import torch.nn as nn

# the 6-Neighborhood excluding center
n6 =  np.array([[[0,0,0],[0,1,0],[0,0,0]],
                [[0,1,0],[1,0,1],[0,1,0]], 
                [[0,0,0],[0,1,0],[0,0,0]]]);


def convolution_3d(input_data, kernel, points):
    
    num_points = points.shape[0]
    output = np.zeros(num_points, dtype = kernel.dtype)
    
    dk, dj, di = input_data.shape
    
    for n in range(num_points):
        z, y, x = points[n]
        
        if z+1<dk and z>0 and y+1<dj and y>0 and x+1<di and x>0:
            for k in range(3):
                zk = z + k - 1
                for j in range(3):
                    yj = y + j - 1
                    for i in range(3):
                        xi = x + i - 1
                        
                        output[n] += input_data[zk, yj, xi] * kernel[k, j, i]
    return output

    
def cube_base():
    # default center is 0
    # Returns an array with base 2 numbers on the cube for convolution and LUT matching
    cube = np.zeros((3,3,3), dtype = int);
    k = 0;
    for z in range(3):
      for y in range(3):
        for x in range(3):
          if x == 1 and y ==1 and z == 1:
            cube[x,y,z] = 0;
          else:
            cube[x, y, z] = 2**k;
            k+=1;
    return cube
            
###############################################################################

# Rotate a cube around an axis in 90 degrees steps
def rotate(input_data, axis = 2, steps = 0):
  
  cube = input_data.copy();  
  
  steps = steps % 4;
  if steps == 0:
    return cube;
  
  elif axis == 0:
    if steps == 1:
      return cube[:, ::-1, :].swapaxes(1, 2)
    elif steps == 2:  # rotate 180 degrees around x
      return cube[:, ::-1, ::-1]
    elif steps == 3:  # rotate 270 degrees around x
      return cube.swapaxes(1, 2)[:, ::-1, :]
      
  elif axis == 1:
    if steps == 1:
      return cube[:, :, ::-1].swapaxes(2, 0)
    elif steps == 2:  # rotate 180 degrees around x
      return cube[::-1, :, ::-1]
    elif steps == 3:  # rotate 270 degrees around x
      return cube.swapaxes(2, 0)[:, :, ::-1]
      
  if axis == 2: # z axis rotation
    if steps == 1:
      return cube[::-1, :, :].swapaxes(0, 1)
    elif steps == 2:  # rotate 180 degrees around z
      return cube[::-1, ::-1, :]
    elif steps == 3:  # rotate 270 degrees around z
      return cube.swapaxes(0, 1)[::-1, :, :]


# Generate rotations in 12 diagonal directions
def rotations12(cube):
  
  rotUS = cube.copy();
  rotUW = rotate(cube, axis = 2, steps = 1);  
  rotUN = rotate(cube, axis = 2, steps = 2); 
  rotUE = rotate(cube, axis = 2, steps = 3);  

  rotDS = rotate(cube,  axis = 1, steps = 2);
  rotDW = rotate(rotDS, axis = 2, steps = 1); 
  rotDN = rotate(rotDS, axis = 2, steps = 2); 
  rotDE = rotate(rotDS, axis = 2, steps = 3);

  rotSW = rotate(cube, axis = 1, steps = 1);   
  rotSE = rotate(cube, axis = 1, steps = 3); 

  rotNW = rotate(rotUN, axis = 1, steps = 1);
  rotNE = rotate(rotUN, axis = 1, steps = 3);
  
  return [rotUS, rotNE, rotDW,  rotSE, rotUW, rotDN,  rotSW, rotUN, rotDE,  rotNW, rotUE, rotDS];