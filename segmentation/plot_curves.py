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

def plot_pre_recall(predicted, truth,  marker='', label='', color='b', title='', background='black', end=False, scatter=False):
    if(np.unique(predicted).size > 1):
        th_range = np.delete(np.unique(predicted), 0)
    else:
        th_range = np.unique(predicted)
    #print(th_range)
    precision   = np.zeros((th_range.size))
    recall      = np.zeros((th_range.size))
    
    for i, t in enumerate(th_range):
        # global thresholding
        if background=='black':
            threshed = (predicted >= t)
        elif background=='white':
            threshed = (predicted <= t)
        met = mt.metric(truth, threshed)

        precision[i] = met.precision()
        recall[i] = met.TPR()
        
        if(recall[i] == 1):
            print('recall is 1 at threhsold:', i)

    if(scatter):
        plt.scatter(recall, precision, marker=marker, color=color, label=label)
    else:
        plt.plot(recall, precision, marker=marker, color=color, label=label)
        if end:
            idxmin = np.argwhere(precision == min(precision))
            plt.scatter(recall[idxmin], precision[idxmin], marker='o', c=color, s=35)
        
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if label is None:
        plt.legend(loc='lower left')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.plot()

def plot_auc_pr(predicteds, truth, var_range, title='', xlabel=''):
    aucs = []
    for i, var in enumerate(var_range):
        print(np.round(var, 3), end=':\t\t')
        predicted = predicteds[i]
        # create the thresholds
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
        indices = np.argsort(recall)
        sorted_recall = recall[indices]
        sorted_precision = precision[indices]
        
        auc = np.trapz(sorted_precision, sorted_recall)
        print('AUC: ', np.round(auc))
        aucs.append(auc)
        
    print('AUC calculations done.\n')
    #plt.cla()
    plt.plot(var_range, np.array(aucs), marker='.', label = 'AUC-PR')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('AUC-PR')
    plt.legend(loc='lower right')
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


