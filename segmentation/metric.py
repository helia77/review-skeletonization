# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:34:49 2023

@author: helioum
"""

import numpy as np

#%%

def jaccard_idx(vol_true, vol_pred):
    intersection = np.logical_and(vol_true, vol_pred)
    union = np.logical_or(vol_true, vol_pred)
    
    j_index = intersection.sum() / float(union.sum())
    return j_index