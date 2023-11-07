import numpy as np
import matplotlib.pyplot as plt
import metric as mt

#%%
kesm_vol = np.load('whole_volume.npy')
kesm_grtruth = np.load('ground_truth.npy')
#%%
micro_vol = np.load('micro_raw_600x700x1010.npy')
micro_grtruth = np.load('micro_grtruth_600x700x1010.npy')
#%%
TPR       = np.zeros((2, 256))
FPR       = np.zeros((2, 256))
recall    = np.zeros((2, 256))
precision = np.zeros((2, 256))
#%%
for t in range(256):
    print(t, end=' ')
    # global thresholding for KESM data
    # threshed_kesm = (kesm_vol <= t)
    # met_kesm = mt.metric(kesm_grtruth, threshed_kesm)
    
    # global thresholding for Micro-CT data
    threshed_micro = (micro_vol >= t)
    met_micro = mt.metric(micro_grtruth, threshed_micro)
    
    # TPR[0, t] = met_kesm.TPR
    # FPR[0, t] = met_kesm.FPR
    # precision[0, t] = met_kesm.precision()
    # recall[0, t] = TPR[0, t]

    TPR[1, t] = met_micro.TPR
    FPR[1, t] = met_micro.FPR
    precision[1, t] = met_micro.precision()
    recall[1, t] = TPR[1, t]
    
    # if (t < np.min(kesm_vol)):
    #     precision[0, t] = 1
        
#%%
plt.figure(1)
#plt.rcParams.update({'font.size': 12})
# plt.plot(recall[0], precision[0], marker='.', label='KESM')
plt.plot(recall[1], precision[1], marker='.', label='Micro-CT')

plt.xlabel('Recall')
plt.ylabel('Precision')

plt.legend(loc='lower left')
plt.grid()

plt.show()

#%%
plt.figure(2)
# plt.plot(FPR[0], TPR[0], marker='.', label='KESM')
plt.plot(FPR[1], TPR[1], marker='.', label='Micro-CT')

plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend(loc='lower left')
plt.grid()

plt.show()


