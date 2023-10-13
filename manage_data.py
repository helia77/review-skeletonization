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

#%%

# given the files path, it loads them all into one numpy array
# num_images: choose how many images to load (the number of images on the z axis)
# stack: if True, it returns all the images as multi-dimension np array - else, returns a list of images
# crop_size: if not zero, it crops all the images into one same size, in both x and y axes
# grayscale: if True, it loads all the images as grayscale. Else, loads as 3-channel RGB
def load_images(file_path, num_images, stack=False, crop_size=0, grayscale=False):
    
    # create a list of all image names
    images_list = [f for f in os.listdir(file_path) if f.endswith('.bmp') or f.endswith('.jpg')]
    
    images = []
    for i in range(num_images):
        # load the image with desired type (grayscale or RGB)
        img = cv2.imread(os.path.join(file_path, images_list[i]), (cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR))
        
        if crop_size:
            img = img[0:crop_size, 0:crop_size]
            
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
        
    for i in range(number):
        cv2.imwrite(os.path.join(folder_path, 'img' + str(i) + '.jpg'), volume[i])
        if i > number:
            break
