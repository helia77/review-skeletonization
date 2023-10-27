import numpy as np
import matplotlib.pyplot as plt
import metric as mt

#%%
micro_grtruth = np.load('micro_grtruth.npy')
micro_vol = np.load('micro_volume.npy')

kesm_vol = np.load('kesm_raw_volume_200x.npy')
kesm_grtruth = np.load('kesm_gr_truth_200x.npy')

TPR       = np.zeros((2, 256))
FPR       = np.zeros((2, 256))
recall    = np.zeros((2, 256))
precision = np.zeros((2, 256))

for t in range(255):
    threshed_kesm = (kesm_vol <= t)
    met_kesm = mt.metric(kesm_grtruth, threshed_kesm)
    
    threshed_micro = (micro_vol >= t)
    met_micro = mt.metric(micro_grtruth, threshed_micro)
    
    TPR[0, t] = met_kesm.TPR
    FPR[0, t] = met_kesm.FPR
    precision[0, t] = met_kesm.precision()
    recall[0, t] = TPR[0, t]

    TPR[1, t] = met_micro.TPR
    FPR[1, t] = met_micro.FPR
    precision[1, t] = met_micro.precision()
    recall[1, t] = TPR[1, t]
    
#%%
# plt.figure(1)
plt.plot(recall[1], precision[1], marker='.', label='Micro-CT')
plt.plot(recall[0], precision[0], marker='.', label='KESM')

plt.xlabel('Recall')
plt.ylabel('Precision')

plt.legend(loc='lower left')

# plt.figure(2)
# plt.plot(FPR[0], TPR[0], label='KESM')
# plt.plot(FPR[1], TPR[1], label='Micro-CT')

# plt.xlabel('FPR')
# plt.ylabel('TPR')

# plt.legend(loc='lower left')
plt.show
#%%
# plot the intensity histogram of Micro-ct volume data
flat_volume = micro_vol.flatten()

# Plot the intensity histogram
n, bins, patches = plt.hist(flat_volume, bins=100)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.title('Intensity Histogram')

plt.axvline(x=bins[np.argmax(n)], color='r', linestyle='dashed', linewidth=2)
plt.text(bins[np.argmax(n)], -190000, round(bins[np.argmax(n)],2), rotation=270, color='red', ha='center')
plt.grid()

plt.show()

#%%
# plot the intensity histogram of KESM volume data
flat_volume = kesm_vol.flatten()

# Plot the intensity histogram
n, bins, patches = plt.hist(kesm_vol.ravel(), 256)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.title('Intensity Histogram of KESM')

plt.axvline(x=bins[np.argmax(n)], color='r', linestyle='dashed', linewidth=2)
plt.text(bins[np.argmax(n)], -30000, round(bins[np.argmax(n)],2), rotation=270, color='red', ha='center')
plt.grid()

plt.show()


