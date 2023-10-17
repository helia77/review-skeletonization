# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 12:37:35 2023

@author: helioum
"""
import sys
sys.path.append('..')
from matplotlib import pyplot as plt
import numpy as np
import otsu
import manage_data as md
import metric as mt
import time
#import cv2

#%%
# load the images as grayscale, crop, and stack them up into one volume
path = 'C:/Users/helif/Documents/GitHub/review-paper-skeletonization/data/data'
crop_size = 200
stack = True
grayscale = True

volume = md.load_images(path, crop_size, stack, crop_size, grayscale)
img_list = md.load_images(path, crop_size, not stack, crop_size, grayscale)
#%%
# compute Otsu's thresholded volume
start = time.time()
thresh_volume, best_thresh = otsu.compute_otsu(volume)

print('\nOtsu\'s threshold: ' + str(best_thresh) + '\n Execution time: --- %s seconds ---' % (time.time() - start))

#%%
# save the slices of each volume for evaluation
num_slices = 50
path = 'C:/Users/helif/Documents/GitHub/review-paper-skeletonization/segmentation/data'
path2 = 'C:/Users/helif/Documents/GitHub/review-paper-skeletonization/segmentation/data2'

md.save_slices(thresh_volume, path, num_slices)
md.save_slices(volume, path2, num_slices)

#%%
# load the true volume (segmented by Slicer3D)
path = 'C:/Users/helif/Documents/GitHub/review-paper-skeletonization/data/Vessels_1-Segment_2-label.nrrd'
vol_true = md.nrrd_to_numpy(path)                               # convert nrrd file to numpy array
vol_true = vol_true[:, 0:200, 0:200].astype(np.uint8)           # crop so that it corresponds to the original volume
vol_true = np.where(vol_true == 255, 0, 255)                    # swap 0's with 1's

#%%
# test for otsu's method for each image slice
thresh_img = []
mean = 0
for i, img in enumerate(img_list):
    thresh, test = otsu.compute_otsu(img)
    mean += test
    if i == 0:
        first_thresh = test
    thresh_img.append(thresh)

thresh_images = np.stack(thresh_img, axis=0)
mean /= len(img_list)
#%%
# testing for adaptive mean thresholding
adaptive_mean = otsu.adaptive_mean(img_list, 21, 5)
adaptive_gaussian = otsu.adaptive_gaussian(img_list, 21, 5)
#%%
# calculate Jaccard index
# thresh = np.zeros(volume.shape, dtype=np.uint8)
# thresh[volume > 72] = 255
jaccard_vol = mt.jaccard_idx(vol_true, thresh_volume)
jaccard_img = mt.jaccard_idx(vol_true, thresh_images)
jaccard_mean = mt.jaccard_idx(vol_true, adaptive_mean)
jaccard_gaussian = mt.jaccard_idx(vol_true, adaptive_gaussian)

dice_vol = mt.dice_coeff(vol_true, thresh_volume)
dice_img = mt.dice_coeff(vol_true, thresh_images)
dice_mean = mt.dice_coeff(vol_true, adaptive_mean)
dice_gaussian = mt.dice_coeff(vol_true, adaptive_gaussian)
#%%
# plot the images
fig, ax = plt.subplots(1,3)
ax[0].imshow(volume[0, :, :], cmap='gray')
ax[0].set_title('Original volume')
ax[0].axis('off')
ax[1].imshow(thresh_volume[0, :, :], cmap='gray')
ax[1].set_title('Otsu\'s volume')
#ax[1].text(10, 10, jaccard_vol, color='black', fontsize=12, fontweight='bold')
ax[1].axis('off')
ax[2].imshow(thresh_images[0, :, :], cmap='gray')
ax[2].set_title('Otsu\'s images')
ax[2].axis('off')

plt.tight_layout()
plt.show()

#%%
# show histogram of a slice
plt.hist(volume.ravel(), 256)
plt.axvline(x=best_thresh, color='r', linestyle='dashed', linewidth=2)
plt.axvline(x=mean, color='g', linestyle='dashed', linewidth=2)
plt.title('Original Data Histogram')
plt.show()