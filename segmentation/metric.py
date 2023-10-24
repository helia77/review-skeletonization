# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:34:49 2023

@author: helioum
"""

import numpy as np

#%%
class metric:
    def __init__(self, true, predicted, pos_label):
        self.vol_true = true
        self.vol_pred = predicted
        self.pos_label = pos_label
        self.positives = (true==pos_label).sum()
        self.TP = np.logical_and(self.vol_true,                     self.vol_pred).sum()
        self.TN = np.logical_and(self.vol_pred == 0,                self.vol_true == 0).sum()
        self.FP = np.logical_and(self.vol_pred == self.pos_label,   self.vol_true == 0).sum()
        self.FN = np.logical_and(self.vol_pred == 0,                self.vol_true == self.pos_label).sum()    
    
    def jaccard_idx(self):
        j_index = self.TP / float(self.TP + self.FP + self.FN)
        return j_index
    
    def dice_coeff(self):
        dice = (2*self.TP) / float(2*self.TP + self.FP + self.FN)
        return dice
    
    # aka True Positive Rate (recall)
    def sensitivity(self):
        recall = self.TP / float(self.TP + self.FN)
        return recall
    
    # aka True Negative Rate
    def specificity(self):
        ratio = self.TN / float(self.TN + self.FP)
        return ratio
    
    # aka False Positive Rate
    def fall_out(self):
        return self.FP / float(self.FP + self.TN)