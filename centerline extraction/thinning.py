# -*- coding: utf-8 -*-
"""
Algorithms for thinning a binary image down to a 1-D centerline
"""
import numpy as np
import skimage.morphology as mph

''' -------------------------------------- Lee's Method -------------------------------------- '''
def skelet_lee(binary_volume):
    '''
    For full description, refer to https://github.com/scikit-image/scikit-image/blob/main/skimage/morphology/_skeletonize.py
    '''
    # Alternative function : mph.skeletonize(sample_gr, method='lee')
    return mph.skeletonize_3d(binary_volume)




''' -------------------------------------- Palagyi's Method -------------------------------------- '''

def skelet_palagyi(binary_volume):
    """ This function creates a centerline from the input binary volume
    Based on the code of ClearMap2 (@ChristophKirst)

    Inputs:
        binary_volume : ndarray
            A 3D binary volume, where
            0: background
            1: object to be skeletonized/centerline extracted
    Requires: 
        "PK12.npy" file, contatinig the 14-templates, downloaded from the ClearMap2 repository
    Returns:
        binary : ndarray
        A binary numpy array containin a 1-D skeleton of the input image

    References
    ----------
    .. [Palagy1999] Palagyi & Kuba, A Parallel 3D 12-Subiteration Thinning Algorithm,
        Graphical Models and Image Processing 61, 199-221 (1999).

    """
    # Extract indices vessel points in the input
    points_arg = np.argwhere(binary_volume)
    binary = np.copy(binary_volume)
    deletable = np.load('PK12.npy')
    rotations = rotations12(cube_base())

    # Border point definition: a black point is a border point if its N_6 neighborhood has at least one white point
       
    while True:
        # Find which black points are border points 
        # the index of the border varibale is the same as black points in point_arg
        border = convolution_3d(binary, n6(), points_arg) < 6
        border_points = points_arg[border]
        border_ids = np.nonzero(border)[0]
        keep = np.ones(len(border), dtype=bool)
        
        iterat = 0
        for i in range(12):         # 12 sub-iterations
        
            removables = convolution_3d(binary, rotations[i], border_points)
            rem_borders = deletable[removables]
            rem_points = border_points[rem_borders]
            
            binary[rem_points[:, 0], rem_points[:, 1], rem_points[:, 2]] = 0
            keep[border_ids[rem_borders]] = False
            
            iterat += len(rem_points)
            #print(i, '\t-  Deleted points: %d' % (len(rem_points)))
        
        #print('Total deletated points:\t', iterat, '\n')
        
        # update foreground
        points_arg = points_arg[keep]
        if iterat == 0:
            #print('No more point removal.')
            break

    return binary


# the 6-Neighborhood excluding center
def n6():
    n6 = np.array([[[0,0,0],[0,1,0],[0,0,0]],
                [[0,1,0],[1,0,1],[0,1,0]], 
                [[0,0,0],[0,1,0],[0,0,0]]]);
    return n6

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

def convolution_3d(input_data, kernel, points):
    
    num_points = points.shape[0]
    output = np.zeros(num_points, dtype = kernel.dtype)
    
    dk, dj, di = input_data.shape
    
    for n in range(num_points):
        z, y, x = points[n]
        
        #if z+1<dk and z>0 and y+1<dj and y>0 and x+1<di and x>0:
        for k in range(3):
            zk = z + k - 1
            if zk < dk and zk >= 0:
                for j in range(3):
                    yj = y + j - 1
                    if yj < dj and yj >= 0:
                        for i in range(3):
                            xi = x + i - 1
                            if xi < di and xi >= 0:        
                                output[n] += input_data[zk, yj, xi] * kernel[k, j, i]
    return output