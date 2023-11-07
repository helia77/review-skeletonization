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
import cthresholding as cp_th

#%%
volume = np.load('micro_raw_600x700x1010.npy')
#%%
path = 'C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/data/Artem\'s data/indata'
img_list = md.load_images(path, [280, 880], False, [700, 1010], True, [280, 120])

#%% 
gr_truth = np.load('micro_grtruth_600x700x1010.npy')
#%%
# load the images as grayscale, crop, and stack them up into one volume
# path = 'C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/data/Artem\'s data/indata'
# volume = md.load_images(path, 200, True, 200, True, [600, 600])
# img_list = md.load_images(path, 200, False, 200, True, [600, 600])
# # load the true volume
# true_path = 'C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/data/Artem\'s data/micro_200x200x200.nrrd'
# vol_true = md.nrrd_to_numpy(true_path)                           # convert nrrd file to numpy array

#%%
# compute Otsu's thresholded volume
start = time.time()
thresh_volume, best_thresh = cp_th.compute_otsu(volume, background='black')
print('\nOtsu\'s (volume) threshold: ' + str(best_thresh) + '\nExecution time: --- %s seconds ---' % (time.time() - start))

#%%
# test otsu's method for each image slice
thresh_img = []
mean = 0
start = time.time()
for i, img in enumerate(img_list):
    print(i, end=' ')
    test, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
    mean += test
    if i == 0:
        first_thresh = test
    thresh_img.append(thresh)

thresh_images = np.stack(thresh_img, axis=0)
mean /= len(img_list)
print('\nOtsu\'s (image) threshold: Done \nExecution time: --- %s seconds ---' % (time.time() - start))
#%%
# testing for adaptive mean thresholding
start = time.time()
adaptive_mean = th.adaptive_mean(img_list, 5, 5)
print('\nAdaptive threshold: Done\nExecution time: --- %s seconds ---' % (time.time() - start))
start = time.time()
adaptive_gaussian = th.adaptive_gaussian(img_list, 5, 5)
print('\nAdaptive threshold: Done\nExecution time: --- %s seconds ---' % (time.time() - start))
#%%
# calculate metrics
volume_met   = mt.metric(gr_truth, thresh_volume)
img_met      = mt.metric(gr_truth, thresh_images)
mean_met     = mt.metric(gr_truth, adaptive_mean)
gaussian_met = mt.metric(gr_truth, adaptive_gaussian)

#%%
# plot the images
fig, ax = plt.subplots(2,2)

i = 100
fig.suptitle('Image %s' % i)
ax[0,0].imshow(volume[i], cmap='gray')
ax[0,0].set_title('Original volume')
ax[0,0].axis('off')

ax[0,1].imshow(gr_truth[i, :, :], cmap='gray')
ax[0,1].set_title('Ground Truth')
ax[0,1].axis('off')

ax[1,0].imshow(thresh_volume[i, :, :], cmap='gray')
ax[1,0].set_title('Otsu\'s vol')
ax[1,0].axis('off')

ax[1,1].imshow(thresh_images[i, :, :], cmap='gray')
ax[1,1].set_title('Otsu\'s Img')
ax[1,1].axis('off')
plt.tight_layout()

 #%%
# show histogram of a slice
plt.hist(volume.ravel(), 256)
plt.axvline(x=best_thresh, color='r', linestyle='dashed', linewidth=2, label='Volume')
plt.axvline(x=mean, color='g', linestyle='dashed', linewidth=2, label='Image')
plt.title('Image Histogram')
plt.xlabel('Intensity')
plt.ylabel('Number of Pixels')
ax = plt.gca()
plt.legend()
plt.show()



