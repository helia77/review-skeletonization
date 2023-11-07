# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 12:37:35 2023

@author: helioum
"""
from matplotlib import pyplot as plt
import numpy as np
import thresholding as th
import manage_data as md
import metric as mt
import time
import os
import cthresholding as cp_th
#%%
# load the images as grayscale, crop, and stack them up into one volume
# path = 'C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/data/KESM'
# crop_size = 200
# stack = True
# grayscale = True

# volume = md.load_images(path, crop_size, stack, crop_size, grayscale)
# img_list = md.load_images(path, crop_size, not stack, crop_size, grayscale)

# # load the true volume (segmented by Slicer3D)
# path = 'C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/data/Segmentation.nrrd'
# vol_true = md.nrrd_to_numpy(path)                               # convert nrrd file to numpy array
# vol_true = vol_true[:, 0:200, 0:200]                            # crop so that it corresponds to the original volume
# vol_true = np.where(vol_true == 255, 0, 255)                    # swap 0's with 1's because background is white

#%%
volume = np.load('whole_volume_kesm.npy').astype(np.uint8)
gr_truth = np.load('ground_truth_kesm.npy').astype(np.uint8)

folder_path = 'C:/Users/helioum/Desktop/region3_helia'
img_list = md.load_images(folder_path, [0,400], stack=False, crop_size=[400, 1000], grayscale=True, crop_location = [0,0])

#%%
# check to make sure all volumes are from the same section
fig, ax = plt.subplots(3,1)

ax[0].imshow(volume[0, :, :], cmap='gray')
ax[0].set_title('Original volume')
ax[0].axis('off')

ax[1].imshow(img_list[0], cmap='gray')
ax[1].set_title('Image List')
ax[1].axis('off')

ax[2].imshow(gr_truth[0, :, :], cmap='gray')
ax[2].set_title('Ground Truth')
ax[2].axis('off')
#%%
# compute Otsu's thresholded volume
start = time.time()
thresh_volume, best_thresh = cp_th.compute_otsu(volume,  background='white')

exe_time = time.time() - start
if (exe_time > 60):
    print('\nOtsu\'s threshold: ' + str(best_thresh) + '\nExecution time: --- %s minutes ---' % (exe_time / 60))
else:
    print('\nOtsu\'s threshold: ' + str(best_thresh) + '\nExecution time: --- %s seconds ---' % (exe_time))

#%%
# test otsu's method for each image slice
thresh_img = []
mean = 0
start = time.time()
for i, img in enumerate(img_list):
    print(i, end=' ')
    thresh, test = th.compute_otsu(img, background='white')
    mean += test
    if i == 0:
        first_thresh = test
    thresh_img.append(thresh)

thresh_images = np.stack(thresh_img, axis=0)
exe_time = time.time() - start
if (exe_time > 60):
    print('\nOtsu\'s threshold: Done\nExecution time: --- %s minutes ---' % (exe_time / 60))
else:
    print('\nOtsu\'s threshold: Done\nExecution time: --- %s seconds ---' % (exe_time))
mean /= len(img_list)
#%%
# testing for adaptive mean thresholding
start = time.time()
adaptive_mean = th.adaptive_mean(img_list, 5, 5)
print(time.time() - start)
start = time.time()
adaptive_gaussian = th.adaptive_gaussian(img_list, 5, 5)
print(time.time() - start)
#%%
# calculate Jaccard index
volume_met   = mt.metric(gr_truth, thresh_volume)
img_met      = mt.metric(gr_truth, thresh_images)
mean_met     = mt.metric(gr_truth, adaptive_mean)
gaussian_met = mt.metric(gr_truth, adaptive_gaussian)
#%%
# plot the images
#for i in range(volume.shape[0]):
fig, ax = plt.subplots(2, 2)
fig.suptitle('Image 250')
ax[0, 0].imshow(volume[250, :, :], cmap='gray')
ax[0, 0].set_title('Original volume')
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])

ax[0, 1].imshow(gr_truth[250, :, :], cmap='gray')
ax[0, 1].set_title('Ground Truth')
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])

ax[1, 0].imshow(thresh_volume[250, :, :], cmap='gray')
ax[1, 0].set_title('Otsu\'s volume')
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])

ax[1, 1].imshow(thresh_images[250, :, :], cmap='gray')
ax[1, 1].set_title('Otsu\'s images')
ax[1, 1].set_xticks([])
ax[1, 1].set_yticks([])
plt.tight_layout()
#%%
# show histogram of a slice
plt.hist(volume.ravel(), 256)
plt.axvline(x=best_thresh, color='r', linestyle='dashed', linewidth=2, label='Volume')
plt.axvline(x=mean, color='g', linestyle='dashed', linewidth=2, label='Image')
plt.title('Original Data Histogram')
plt.legend()
plt.show()
