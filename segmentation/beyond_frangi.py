# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:19:02 2023

@author: helioum
"""
import numpy as np
import matplotlib.pyplot as plt
import frangi
import metric as mt
import time
import thresholding as th
import plot_curves as pltc
#%%
def plot_pre_recall(predicted, truth, tau, label='', color='b', lwidth=1.0, offset=0.0, end=False, start=False, scatter=False):
    if(np.unique(predicted).size > 1):
        th_range = np.delete(np.unique(predicted), 0)
    else:
        th_range = np.unique(predicted)
        
    #print(th_range)
    precision   = np.zeros((th_range.size))
    recall      = np.zeros((th_range.size))
    
    for i, t in enumerate(th_range):
        # global thresholding
        threshed = (predicted >= t)
        met = mt.metric(truth, threshed)
        
        precision[i] = met.precision()
        recall[i] = met.TPR()
        
        if(recall[i] == 1):
            print('recall is 1 at threhsold:', i)

    if(scatter):
        plt.scatter(recall, precision, color=color, label=label)
    else:
        plt.plot(recall+offset, precision+offset, color=color, label=label, linewidth=lwidth)
        if end:
            idxmax = np.argwhere(precision == min(precision))
            plt.scatter(recall[idxmax]+offset, precision[idxmax]+offset, marker='o', c=color, s=35)
        if start:
            idxmin = np.argwhere(recall == min(recall))
            plt.scatter(recall[idxmin]+offset, precision[idxmin]+offset, marker='o', c=color, s=35)
        #plt.annotate('$\\tau$=' + str(tau), xy=(recall[idxmin] - 0.1, precision[idxmin] - 0.01))
    plt.title('Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.legend(loc='lower left', fontsize='small')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.plot()
#%%
# test for 3D volume - KESM
volume = np.load('whole_volume_kesm.npy')
gr_truth = np.load('ground_truth_kesm.npy')

sample_vol = volume[0:100, 300:400, 300:500]
sample_gr = gr_truth[0:100, 300:400, 300:500]
scale_range = [0.9, 1.8, 2.7, 3.6]

#%%
# plot AUC and tau parameter using Beyond Frangi function
#taus = [0.05, 0.1, 0.25, 0.5, 10]
taus = [0.1, 0.3, 0.7, 1, 1.7]
all_bfrangi = []
all_final = []
start = time.time()
for tau in taus:
    print('tau=', tau)
    result = frangi.beyond_frangi_filter(sample_vol, tau, scale_range, 'white')
    thresh, _ = th.compute_otsu_img(result, 'black')
    met_img = mt.metric(sample_gr, thresh)
    print(met_img.dice(), ' ', met_img.jaccard())
    all_bfrangi.append(result)
    all_final.append(thresh)
print('Took ', time.time() - start, ' secs')

#%%
#n = 10
n = 40
fig, ax = plt.subplots(6, 2)
for i in range(5):
    ax[i, 0].imshow(all_bfrangi[i][n], cmap='gray')
    ax[i, 1].imshow(all_final[i][n], cmap='gray')

ax[5, 0].imshow(sample_vol[n], cmap='gray')
ax[5, 1].imshow(sample_gr[n], cmap='gray')

for i in range(6):
    for j in range(2):
        ax[i, j].axis('off')
        
ax[0, 0].text(.5, 1.5, 'Filtered'             , va='center', ha='center', transform=ax[0, 0].transAxes)
ax[0, 0].text(-.5, .5, '$\\tau$='+str(taus[0]), va='center', ha='center', transform=ax[0, 0].transAxes)
ax[0, 1].text(.5, 1.5, '+Otsu'                , va='center', ha='center', transform=ax[0, 1].transAxes)
ax[1, 0].text(-.5, .5, '$\\tau$='+str(taus[1]), va='center', ha='center', transform=ax[1, 0].transAxes)
ax[2, 0].text(-.5, .5, '$\\tau$='+str(taus[2]), va='center', ha='center', transform=ax[2, 0].transAxes)
ax[3, 0].text(-.5, .5, '$\\tau$='+str(taus[3]), va='center', ha='center', transform=ax[3, 0].transAxes)
ax[4, 0].text(-.5, .5, '$\\tau$='+str(taus[4]), va='center', ha='center', transform=ax[4, 0].transAxes)

ax[5, 0].text(.5, -2.5, 'raw'                  , va='center', ha='center', transform=ax[3, 0].transAxes)
ax[5, 1].text(.5, -2.5, 'ground'               , va='center', ha='center', transform=ax[3, 1].transAxes)

#%%
# plot the precision-recall curves for the results of each tau
plt.figure(2)
plt.grid()
plot_pre_recall(all_bfrangi[0], sample_gr, taus[0], color='b', label='$\\tau$='+str(taus[0]))
plot_pre_recall(all_bfrangi[1], sample_gr, taus[1], color='g', label='$\\tau$='+str(taus[1]), start=True)
plot_pre_recall(all_bfrangi[2], sample_gr, taus[2], color='r', label='$\\tau$='+str(taus[2]), start=True)
plot_pre_recall(all_bfrangi[3], sample_gr, taus[3], color='k', label='$\\tau$='+str(taus[3]))
plot_pre_recall(all_bfrangi[4], sample_gr, taus[4], color='k', label='$\\tau$='+str(taus[4]), end=True)
#pltc.plot_pre_recall(sample_vol, sample_gr, label='global', color='c', background='white')

# calculate the precision-recall point of Otsu's image
_, best_thresh = th.compute_otsu_img(sample_vol, background='white')
threshed_vol = (sample_vol <= best_thresh)
mets = mt.metric(sample_gr, threshed_vol)
precision = mets.precision()
recall = mets.TPR()
plt.scatter(recall, precision, marker='x', c='red', s=35, label='Otsu Img.')

# calculate the precision-recall point of Otsu's volume
best_thresh_vol = th.compute_otsu_hist(sample_vol)
threshed_vol = (sample_vol <= best_thresh)
mets = mt.metric(sample_gr, threshed_vol)
precision = mets.precision()
recall = mets.TPR()
plt.scatter(recall, precision, marker='o', c='blue', s=25, label='Otsu Vol.')

# plot the precision-recall curve for global thresholding
th_range = np.delete(np.unique(sample_vol), 0)

precisions   = np.zeros((th_range.size))
recalls      = np.zeros((th_range.size))
for i, t in enumerate(th_range):
    # global thresholding
    threshed_global = (sample_vol <= t)
    #if i == 70:
        #plt.imshow(threshed_global[35], cmap='gray')
    met = mt.metric(sample_gr, threshed_global)

    precisions[i] = met.precision()
    recalls[i] = met.TPR()

if(recalls[i] == 1):
    print('recall is 1 at threhsold:', i)
plt.plot(recalls, precisions, color='c', label='global')

plt.legend(loc='lower left', fontsize='small')

#%%
for i in range(5):
    print(i+1, ' tau:', taus[i], end='')
    met = mt.metric(sample_gr, all_bfrangi[i])
    auc = met.return_auc()
    print(' auc:', auc, end='')
    met1 = mt.metric(sample_gr, all_final[i])
    print(' dice:', met1.dice())
 