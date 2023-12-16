# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:34:49 2023

@author: helioum
"""

import numpy as np
import math
#%%
class metric:
    def __init__(self, true, predicted):
        # self.vol_true = true
        # self.vol_pred = predicted
        self.positives = np.count_nonzero(true)
        self.negatives = np.count_nonzero(np.logical_not(true))
        
        self.TP = np.logical_and(true, predicted).sum()
        self.TN = np.logical_and(np.logical_not(predicted), np.logical_not(true)).sum()
        self.FP = np.logical_and(np.logical_not(true), predicted).sum()
        self.FN = np.logical_and(true, np.logical_not(predicted)).sum()    
        
        #self.TPR = self.TP / self.positives
        #self.FPR = self.FP / self.negatives
        
        #self.jaccard = self.TP / float(self.TP + self.FP + self.FN)
        #self.dice = (2*self.TP) / float(2*self.TP + self.FP + self.FN)
        
    # aka recall, sensitivity
    def TPR(self):
        sensitivity = self.TP / self.positives
        return sensitivity
    
    def FPR(self):
        return self.FP / self.negatives
    
    # aka True Negative Rate
    def specificity(self):
        ratio = self.TN / (self.TN + self.FP)
        return ratio
    
    # aka False Positive Rate
    def fall_out(self):
        return self.FP / (self.FP + self.TN)
    
    def precision(self):
        if(self.TP + self.FP == 0):
            return 0
        return self.TP / (self.TP + self.FP)
    
    def jaccard(self):
        return self.TP / float(self.TP + self.FP + self.FN)
    
    # aka F1 score (harmonic mean of precision and recall)
    def dice(self):
        return (2*self.TP) / float(2*self.TP + self.FP + self.FN)