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
import sklearn.metrics as sk

#%%
# load the images as grayscale, crop, and stack them up into one volume
path = 'C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/data/Artem\'s data/indata'
crop_size = 200
stack = True
grayscale = True

volume = md.load_images(path, crop_size, stack, crop_size, grayscale, [600, 600])
img_list = md.load_images(path, crop_size, not stack, crop_size, grayscale, [600, 600])

# save nrrd file to load in Slicer3D
#filename = 'raw_micro_200x200x200.nrrd'
#md.numpy_to_nrrd(volume, filename)

# load the true volume
true_path = 'C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/data/Artem\'s data/micro_200x200x200.nrrd'
vol_true = md.nrrd_to_numpy(true_path)                           # convert nrrd file to numpy array
vol_true *= 255 
#%%
# compute Otsu's thresholded volume
start = time.time()
thresh_volume, best_thresh = th.compute_otsu(volume)

print('\nOtsu\'s threshold: ' + str(best_thresh) + '\nExecution time: --- %s seconds ---' % (time.time() - start))

#%%
# test otsu's method for each image slice
thresh_img = []
mean = 0
for i, img in enumerate(img_list):
    thresh, test = th.compute_otsu(img)
    mean += test
    if i == 0:
        first_thresh = test
    thresh_img.append(thresh)

thresh_images = np.stack(thresh_img, axis=0)
mean /= len(img_list)

#%%
# testing for adaptive mean thresholding
adaptive_mean = th.adaptive_mean(img_list, 21, 5)
adaptive_gaussian = th.adaptive_gaussian(img_list, 21, 5)
#%%
# calculate metrics
volume_met   = mt.metric(vol_true, thresh_volume)
img_met      = mt.metric(vol_true, thresh_images)
mean_met     = mt.metric(vol_true, adaptive_mean)
gaussian_met = mt.metric(vol_true, adaptive_gaussian)

#%%
folder_path = 'C:/Users/helioum/Documents/GitHub/review-paper-skeletonization/segmentation/figures_micro/'
os.makedirs(folder_path)
# plot the images
for i in range(volume.shape[0]):
    fig, ax = plt.subplots(1,3)
    
    ax[0].imshow(img_list[i], cmap='gray')
    ax[0].set_title('Original volume')
    ax[0].axis('off')
    
    ax[1].imshow(vol_true[i, :, :], cmap='gray')
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')
    
    ax[2].imshow(thresh_images[i, :, :], cmap='gray')
    ax[2].set_title('Otsu\'s images')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(folder_path + 'plot' + str(i) + '.png')
    plt.close(fig)

#%%
# show histogram of a slice
plt.hist(volume.ravel(), 256)
plt.axvline(x=best_thresh, color='r', linestyle='dashed', linewidth=2)
plt.axvline(x=mean, color='g', linestyle='dashed', linewidth=2)
plt.title('Original Data Histogram')
plt.show()

#%%

# plot the ROC curve
TPR = []
FPR = []

# iterate through all thresholds
for i in range(240, 20, -1):
    thresholded = np.zeros(volume.shape)
    
    thresholded[volume >= i] = 255
    met = mt.metric(vol_true, thresholded, 255)

    TPR.append(met.sensitivity())
    FPR.append(met.fall_out())

#%%
plt.plot(FPR, TPR, label='ROC', marker='o', color='green')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


