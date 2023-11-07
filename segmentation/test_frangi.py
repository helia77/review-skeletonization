# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:40:28 2023

@author: helioum
"""

import numpy as np
import frangi as frg
import cv2
import matplotlib.pyplot as plt
import time
import metric as mt
import manage_data as md
import cthresholding as cp_th
import thresholding as th
#%%
# test for 3D volume - KESM
volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

folder_path = 'C:/Users/helioum/Desktop/region3_helia'
img_list = md.load_images(folder_path, [0,400], stack=False, crop_size=[400, 1000], grayscale=True, crop_location = [0,0])
#%%
sample_vol = volume[200:300, 0:100, 0:100]
sample_gr = gr_truth[200:300, 0:100, 0:100]

#%%
start = time.time()
result = frg.frangi_3D(volume, 0.01, 1, 45, 3, 7, 1, 'white')
print('Frangi filter took: ', time.time() - start)

 #%%

# check the metrics
new_result = np.uint8(result * 255)
met_frangi = mt.metric(gr_truth, new_result)
np.save('frangi_result_kesm_a.01b1c45-[3-7-1].npy', result)
np.save('valid_frangi_kesm_a.01b1c45-[3-7-1].npy', new_result)
md.numpy_to_nrrd(result, 'frangi_kesm_a.01b1c45-[3-7-1].nrrd')
#%%
# check to make sure all volumes are from the same section
fig, ax = plt.subplots(3,1)
i = 10
fig.suptitle('Image' + str(i))
ax[0].imshow(volume[i], cmap='gray')
ax[0].set_title('Original volume')
ax[0].axis('off')

ax[1].imshow(gr_truth[i], cmap='gray')
ax[1].set_title('Ground Truth')
ax[1].axis('off')

ax[2].imshow(new_result[i], cmap='gray')
ax[2].set_title('Frangi results')
ax[2].axis('off')


#%%
start = time.time()
thresh_volume, best_thresh = cp_th.compute_otsu(new_result,  background='white')

exe_time = time.time() - start
if (exe_time > 60):
    print('\nOtsu\'s threshold: ' + str(best_thresh) + '\nExecution time: --- %s minutes ---' % (exe_time / 60))
else:
    print('\nOtsu\'s threshold: ' + str(best_thresh) + '\nExecution time: --- %s seconds ---' % (exe_time))

#%%
thresh_img = []
mean = 0
start = time.time()
for i in range(new_result.shape[0]):
    img = new_result[i]
    test, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
    mean += test
    thresh_img.append(thresh)

thresh_images = np.stack(thresh_img, axis=0)
exe_time = time.time() - start

mean /= len(img_list)
if (exe_time > 60):
    print('\nOtsu\'s img threshold: ' + str(mean) + '\nExecution time: --- %s minutes ---' % (exe_time / 60))
else:
    print('\nOtsu\'s img threshold: '+ str(mean) + '\nExecution time: --- %s seconds ---' % (exe_time))


#%%
met_otsu_vol = mt.metric(gr_truth, thresh_volume)
met_otsu_img = mt.metric(gr_truth, thresh_images)
#%%
fig, ax = plt.subplots(2,1)
i = 10
fig.suptitle('Image ' + str(i))
# ax[0].imshow(volume[i], cmap='gray')
# ax[0].set_title('Original volume')
# ax[0].axis('off')

# ax[1].imshow(gr_truth[i], cmap='gray')
# ax[1].set_title('Ground Truth')
# ax[1].axis('off')

# ax[2].imshow(result[i], cmap='gray')
# ax[2].set_title('Frangi results')
# ax[2].axis('off')

ax[0].imshow(thresh_volume[i], cmap='gray')
ax[0].set_title('Volume Otsu')
ax[0].axis('off')

ax[1].imshow(thresh_images[i], cmap='gray')
ax[1].set_title('Image Otsu')
ax[1].axis('off')


#%%
pre_result = np.load('frangi results/valid_frangi_results_a.1b.5c50-[9-10-1].npy')
TPR_       = np.zeros((np.unique(pre_result).size))
FPR_       = np.zeros((np.unique(pre_result).size))
precision_ = np.zeros((np.unique(pre_result).size))
#%%
start = time.time()
for t in np.unique(pre_result):
    print(t, end=' ')
    # global thresholding for KESM data
    threshed_frangi = (pre_result >= t)
    met_frangi = mt.metric(gr_truth, threshed_frangi)

    TPR_[t] = met_frangi.TPR
    FPR_[t] = met_frangi.FPR
    precision_[t] = met_frangi.precision()

print('\nExecution time: ', (time.time() - start)/60)

#%%
vol_size = np.unique(volume).size
TPR_2       = np.zeros((vol_size))
FPR_2       = np.zeros((vol_size))
precision_2 = np.zeros((vol_size))
#%%
for i, t in enumerate(np.unique(volume)):
    print(i, end=' ')
    # global thresholding for KESM data
    threshed_kesm = (volume <= t)
    met_kesm = mt.metric(gr_truth, threshed_kesm)

    TPR_2[i] = met_kesm.TPR
    FPR_2[i] = met_kesm.FPR
    precision_2[i] = met_kesm.precision()
    
#%%
plt.figure(1)
plt.plot(FPR, TPR, marker='.', label='Parameter 1')
plt.plot(FPR_, TPR_, marker='.', label='Parameter 2')
plt.plot(FPR_2, TPR_2, marker='.', label='Global')
plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend(loc='lower right')
plt.grid()

plt.show()

#%%
plt.figure(2)
plt.plot(TPR, precision, marker='.', label='Parameter 1')
plt.plot(TPR_, precision_, marker='.', label='Parameter 2')
plt.plot(TPR_2, precision_2, marker='.', label='Global')
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.legend(loc='lower left')
plt.grid()

plt.show()