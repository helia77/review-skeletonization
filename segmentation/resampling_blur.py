# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:39:34 2024

@author: helioum
"""
import numpy as np
import scipy.ndimage as nd
import SimpleITK as sitk
import nrrd 

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

volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

sample_vol = volume[0:200, 200:400, 300:500]
sample_gr = gr_truth[0:200, 200:400, 300:500]

#%%
# resample raw and ground truth volumes (z-axis x2)
resampled_volume = resample_volume(sample_vol, new_spacing=[0.5, 1, 1])
resampled_volume = resampled_volume[:, :, 100:300]
resampled_gr = resample_volume(sample_gr, new_spacing=[0.5, 1, 1])
resampled_gr = resampled_gr[:, :, 100:300]

#%%
# blur the volumes
volume_blurred = nd.median_filter(resampled_volume, size=3)
ground_blurred = nd.median_filter(resampled_gr, size=3)

# save nrrd files
nrrd.write('raw_volume(resampled)_gaussian.nrrd', volume_blurred)
nrrd.write('gr_truth(resampled)_gaussian.nrrd', ground_blurred)

