# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:22:51 2023

This code plots ROC curves and Precision-Recall, based on the inputs: predicted data and the ground truth
@author: helioum
"""

import numpy as np
import matplotlib.pyplot as plt
import metric as mt
#%%

def plot_pre_recall(predicted, truth,  marker='.', label=''):
    if(np.unique(predicted).size > 1):
        th_range = np.delete(np.unique(predicted), 0)
    else:
        th_range = np.unique(predicted)
    precision   = np.zeros((th_range.size))
    recall      = np.zeros((th_range.size))
    for i, t in enumerate(th_range):
        # global thresholding
        threshed = (predicted >= t)
        met = mt.metric(truth, threshed)

        precision[i] = met.precision()
        recall[i] = met.TPR()
        
        if(recall[i] == 1):
            print(i)

    plt.plot(recall, precision, marker=marker, label = label)
    plt.title('Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.plot()


def plot_roc(predicted, truth, marker='.', label=''):
    th_range = np.unique(predicted)
    TPR      = np.zeros((th_range.size))
    FPR      = np.zeros((th_range.size))
    for i, t in enumerate(th_range):
        # global thresholding
        threshed = (predicted > t)
        met = mt.metric(truth, threshed)

        TPR[i] = met.TPR()
        FPR[i] = met.FPR()
    
    plt.plot(FPR, TPR, marker=marker, label = label)
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='lower right')
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.plot()


