# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:34:49 2023

@author: helioum
"""

import numpy as np

#%%

def TP(vol_true, vol_pred):
    return np.logical_and(vol_true, vol_pred).sum()
def TN(vol_true, vol_pred):
    return np.logical_and(vol_pred == 0, vol_true == 0).sum()
def FP(vol_true, vol_pred):
    return np.logical_and(vol_pred == 255, vol_true == 0).sum()
def FN(vol_true, vol_pred):
    return np.logical_and(vol_pred == 0, vol_true == 255).sum()

    
def jaccard_idx(vol_true, vol_pred):
    intersection = np.logical_and(vol_true, vol_pred)
    union = np.logical_or(vol_true, vol_pred)
    
    #j_index = intersection.sum() / float(union.sum())
    j_index = (TP(vol_true, vol_pred)) / (TP(vol_true, vol_pred) + FP(vol_true, vol_pred) + FN(vol_true, vol_pred))
    return j_index * 100

def dice_coeff(vol_true, vol_pred):
    intersection = np.logical_and(vol_true, vol_pred)
    summation = vol_true.size + vol_pred.size
    
    #dice = 2 * (intersection.sum()) / summation
    dice = (2*TP(vol_true, vol_pred)) / (2*TP(vol_true, vol_pred) + FP(vol_true, vol_pred) + FN(vol_true, vol_pred))
    return dice * 100