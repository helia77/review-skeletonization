# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:34:49 2023

@author: helioum
"""

import numpy as np

#%%
class metric:
    def __init__(self, true, predicted):
        self.vol_true = true
        self.vol_pred = predicted
        
    def TP(self):
        return np.logical_and(self.vol_true, self.vol_pred).sum()
    def TN(self):
        return np.logical_and(self.vol_pred == 0, self.vol_true == 0).sum()
    def FP(self):
        return np.logical_and(self.vol_pred == 255, self.vol_true == 0).sum()
    def FN(self):
        return np.logical_and(self.vol_pred == 0, self.vol_true == 255).sum()
    
        
    def jaccard_idx(self):
        j_index = (self.TP()) / float(self.TP() + self.FP() + self.FN())
        return j_index * 100
    
    def dice_coeff(self):
        dice = (2*self.TP()) / float(2*self.TP() + self.FP() + self.FN())
        return dice * 100
    
    # aka True Positive Rate
    def sensitivity(self):
        recall = self.TP() / float(self.TP() + self.FN())
        return recall * 100
    
    # aka True Negative Rate
    def specificity(self):
        ratio = self.TN() / float(self.TN() + self.FP())
        return ratio * 100