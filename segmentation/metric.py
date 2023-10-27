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
        self.positives = np.count_nonzero(true)
        self.negatives = np.count_nonzero(np.logical_not(true))
        
        self.TP = np.logical_and(self.vol_true, self.vol_pred).sum()
        self.TN = np.logical_and(np.logical_not(self.vol_pred), np.logical_not(self.vol_true)).sum()
        self.FP = np.logical_and(np.logical_not(self.vol_true), self.vol_pred).sum()
        self.FN = np.logical_and(self.vol_true, np.logical_not(self.vol_pred)).sum()    
        
        self.TPR = self.TP / self.positives
        self.FPR = self.FP / self.negatives
        
    def jaccard_idx(self):
        j_index = self.TP / float(self.TP + self.FP + self.FN)
        return j_index
    
    def dice_coeff(self):
        dice = (2*self.TP) / float(2*self.TP + self.FP + self.FN)
        return dice
    
    # aka True Positive Rate (recall)
    def recall(self):
        sensitivity = self.TP / (self.TP + self.FN)
        return sensitivity
    
    # aka True Negative Rate
    def specificity(self):
        ratio = self.TN / (self.TN + self.FP)
        return ratio
    
    # aka False Positive Rate
    def fall_out(self):
        return self.FP / (self.FP + self.TN)
    
    def precision(self):
        return self.TP / (self.TP + self.FP)
    