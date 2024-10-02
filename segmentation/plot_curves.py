# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:22:51 2023

This code plots ROC curves and Precision-Recall, based on the inputs: predicted data and the ground truth
@author: helioum
"""
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle, CapStyle
from matplotlib.transforms import Affine2D
import metric as mt
import numpy as np

csfont = {'fontname':'Times New Roman'}
#%%

def plot_pre_recall(predicted, truth,  marker='', label='', color='b', angle=0, background='black', end=False):
    
    id_del = 0
    #t = Affine2D().rotate_deg(angle)
    mitte = np.mean(predicted)
    
    print(mitte)
    if np.unique(predicted).size == 2:                  # in case of binary file
        th_range = np.unique(predicted)
    elif abs(mitte) < 0.0005:
        th_range = np.unique(np.round(predicted, 5))
    elif abs(mitte) < 0.005:
        th_range = np.unique(np.round(predicted, 4))    
    elif abs(mitte) < 0.05:
        th_range = np.unique(np.round(predicted, 3))
    elif abs(mitte) < 0.5:
        th_range = np.unique(np.round(predicted, 3))
    elif abs(mitte) < 1.0:
        th_range = np.unique(np.round(predicted, 2))
    else:
        th_range = np.unique(np.round(predicted, 1))
    
    #mrk = MarkerStyle('X', transform=t, capstyle=CapStyle('round'))
    th_range = np.delete(th_range, 0)
    print(th_range.size)

    precision   = np.zeros((th_range.size))
    recall      = np.zeros((th_range.size))
    
    for i, t in enumerate(th_range):
        # global thresholding
        if background=='black':
            if t < 0:
                threshed = (-predicted <= t)
            else:
                threshed = (predicted >= t)
        elif background=='white':
            threshed = (predicted <= t)
        met = mt.metric(truth, threshed)

        precision[i] = met.precision()
        recall[i] = met.TPR()
        
        if(recall[i] == 0 and precision[i] == 0):
            id_del = i
            print('zeros at threshold:', t)
    if th_range[id_del] == th_range[-1] and th_range.size > 2:
        recall = np.delete(recall, -1)
        precision = np.delete(precision, -1)
    if(th_range.size == 1 and th_range[0] == 1):
        plt.scatter(recall, precision, marker='X', s=280, color=color, label=label, linewidths=1)
    else:
        plt.plot(recall, precision, marker=marker, color=color, label=label, linewidth=3)
        #if end:
        #    plt.scatter(recall[0], precision[0], marker=mrk, c=color, s=275, linewidths=1)

    plt.xlabel('Recall', fontsize=25, **csfont)
    plt.ylabel('Precision', fontsize=25, **csfont)
    if label is not None:
        plt.legend(loc='lower left', prop={"family":'Times New Roman', "size":"18", "weight":"bold"})
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


