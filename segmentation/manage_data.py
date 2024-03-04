# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:42:47 2023

@author: helioum

access, load, save, stack data
"""

import os
import shutil
import numpy as np
import cv2
import nrrd
from PIL import Image
#%%

# given the files path, it loads them all into one numpy array
# num_images: choose how many images to load (the number of images on the z axis)
# stack: if True, it returns all the images as multi-dimension np array - else, returns a list of images
# crop_size: if not zero, it crops all the images into one same size, in both x and y axes
# grayscale: if True, it loads all the images as grayscale. Else, loads as 3-channel RGB
def load_images(folder_path, num_img_range, stack=False, grayscale=False, crop_size=[0,0], crop_location = [0,0]):
    
    # create a list of all image names
    images_list = [f for f in os.listdir(folder_path) if f.endswith('.bmp') or f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.png')]
    if num_img_range == 'all':
        img_range = [0, len(images_list)]
    elif isinstance(num_img_range, int):
        img_range = [0, num_img_range]
    else:
        img_range = num_img_range
        
    images = []
    for i in range(img_range[0], img_range[1], 1):
        # load the image with desired type (grayscale or RGB)
        img = cv2.imread(os.path.join(folder_path, images_list[i]), (cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR))
    
        if crop_size != [0, 0]:
            x, y = crop_location
            img = img[x:x+crop_size[0], y:y+crop_size[1]]
            
        images.append(img)

    if stack:
        return np.stack(images, axis=0)
    else:
        return images

#%%
# given the volume, saves all the slices as images in a created folder
def save_slices(volume, folder_path, number):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
        
    if number == 'all':
        number = len(volume)
    for i in range(number):
        cv2.imwrite(os.path.join(folder_path, 'img' + str(i) + '.jpg'), volume[i])
        if i > number:
            break

#%%
# loads nrrd file (segmentations from Slicer 3D), and converts it to numpy file
def nrrd_to_numpy(nrrd_path):
    file = nrrd.read(nrrd_path)[0]
    return file.astype(np.uint8)

def numpy_to_nrrd(arr, filename):
    # convertin numpy array to nrrd file
    nrrd.write(filename, arr)
    
#%%
# convert numpy to obj file
def numpy_to_obj(vertices, edges, f, offset):
    f.write("# new line\n")
    for v in vertices:
        if(len(v) == 3):
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        if(len(v) == 2):
            f.write(f"v {v[0]} {v[1]} {'0'}\n")
        
    f.write("l ") 
    for e in edges:
        f.write(f"{e+offset} ")
    f.write("\n")
    
#%%
def change_level(folder_path, num_img_range, shadows=0.0, highlights=1.0, stack=False):
    images_list = [f for f in os.listdir(folder_path) if f.endswith('.bmp') or f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.png')]
    
    if num_img_range == 'all':
        img_range = [0, len(images_list)]
    elif isinstance(num_img_range, int):
        img_range = [0, num_img_range]
    else:
        img_range = num_img_range
        
    images = []
    for i in range(img_range[0], img_range[1], 1):
        # load the image with desired type (grayscale or RGB)
        img = Image.open(os.path.join(folder_path, images_list[i]))
        # make sure the image is grayscale
        img_pil = img.convert("L")

        # Apply the point operation to adjust levels
        img_modified = img_pil.point(lambda x: x * (highlights - shadows) + shadows)
        images.append(img_modified)
    
    if stack:
        return np.stack(images, axis=0)
    else:
        return images
    
#%%
def resample_volume(src, interpolator=sitk.sitkLinear, new_spacing = [1, 1, 1]):
    volume = sitk.GetImageFromArray(src)
    volume = sitk.Cast(volume, sitk.sitkFloat32)
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    sitk_volume = sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())
    
    resampled_volume = sitk.GetArrayFromImage(sitk_volume).astype(dtype=np.uint8)
    return resampled_volume