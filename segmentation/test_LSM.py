# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:29:11 2023

@author: helioum
"""

from matplotlib import pyplot as plt
import numpy as np
import thresholding as th
import manage_data as md
import metric as mt
import time
import os
import cv2

#%%
# load the images as grayscale, crop, and stack them up into one volume
path = 'C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/data/LSM'
crop_size = 200
stack = True
grayscale = True

volume = md.load_images(path, crop_size, stack, crop_size, grayscale, [600, 800])
img_list = md.load_images(path, crop_size, not stack, crop_size, grayscale, [600, 800])
