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
        self.vol_true = true
        self.vol_pred = predicted
        self.positives = np.count_nonzero(true)
        self.negatives = np.count_nonzero(np.logical_not(true))
        
        self.TP = np.logical_and(true, predicted).sum()
        self.TN = np.logical_and(np.logical_not(predicted), np.logical_not(true)).sum()
        self.FP = np.logical_and(np.logical_not(true), predicted).sum()
        self.FN = np.logical_and(true, np.logical_not(predicted)).sum()    
        
    # aka recall, sensitivity
    def TPR(self):
        sensitivity = self.TP / (self.TP + self.FN)
        return sensitivity
    
    def FPR(self):
        return self.FP / self.negatives
    
    def accuracy(self):
        return (self.TP + self.TN) / (self.positives + self.negatives)
    
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
    
    def f_score(self):
        return self.dice(self)
    
    def jaccard(self):
        if self.positives == 0:
            return 1
        return self.TP / float(self.TP + self.FP + self.FN)
    
    # aka Fscore (harmonic mean of precision and recall)
    def dice(self):
        return (2*self.TP) / float(2*self.TP + self.FP + self.FN)
    
    def return_auc(self, background):
        if(np.unique(self.vol_pred).size > 1):
            th_range = np.delete(np.unique(self.vol_pred), 0)
        else:
            th_range = np.unique(self.vol_pred)
        precision   = np.zeros((th_range.size))
        recall      = np.zeros((th_range.size))
        for i, t in enumerate(th_range):
            # global thresholding
            if background == 'black':
                threshed = (self.vol_pred >= t)
            elif background == 'white':
                threshed = (self.vol_pred <= t)
            else:
                print('Wrong background input. Choose \'white\' or \'black\'.')
            met = metric(self.vol_true, threshed)
            precision[i] = met.precision()
            recall[i] = met.TPR()
        
        indices = np.argsort(recall)
        sorted_recall = recall[indices]
        sorted_precision = precision[indices]
        
        auc = np.trapz(sorted_precision, sorted_recall)
        return auc