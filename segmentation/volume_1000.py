# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:39:02 2023

@author: helioum
"""

import manage_data as md
import numpy as np
import os
#%%
folder_path = 'C:/Users/helioum/Desktop/region3_helia'
num_images = 772
volume_1000 = md.load_images(folder_path, num_images, True, grayscale=True)

#%%
# vol_ x _ y _ z : x horizontal, y vertical, z depth
vol_0_0_0 = volume_1000[0:200, 0:200, 0:200]
vol_1_0_0 = volume_1000[0:200, 0:200, 200:400]
vol_2_0_0 = volume_1000[0:200, 0:200, 400:600]
vol_3_0_0 = volume_1000[0:200, 0:200, 600:800]
vol_4_0_0 = volume_1000[0:200, 0:200, 800:1000]

# md.numpy_to_nrrd(vol_0_0_0, 'Data/vol_0_0_0.nrrd')
# md.numpy_to_nrrd(vol_1_0_0, 'Data/vol_1_0_0.nrrd')
# md.numpy_to_nrrd(vol_2_0_0, 'data/vol_2_0_0.nrrd')
# md.numpy_to_nrrd(vol_3_0_0, 'Data/vol_3_0_0.nrrd')
# md.numpy_to_nrrd(vol_4_0_0, 'Data/vol_4_0_0.nrrd')

vol_1 = np.concatenate((vol_0_0_0, vol_1_0_0, vol_2_0_0, vol_3_0_0, vol_4_0_0), axis=2)
# segmentation done
#%%
vol_0_1_0 = volume_1000[0:200, 200:400, 0:200]
vol_1_1_0 = volume_1000[0:200, 200:400, 200:400]
vol_2_1_0 = volume_1000[0:200, 200:400, 400:600]
vol_3_1_0 = volume_1000[0:200, 200:400, 600:800]
vol_4_1_0 = volume_1000[0:200, 200:400, 800:1000]

# md.numpy_to_nrrd(vol_0_1_0, 'Data/vol_0_1_0.nrrd')
# md.numpy_to_nrrd(vol_1_1_0, 'Data/vol_1_1_0.nrrd')
# md.numpy_to_nrrd(vol_2_1_0, 'Data/vol_2_1_0.nrrd')
# md.numpy_to_nrrd(vol_3_1_0, 'Data/vol_3_1_0.nrrd')
# md.numpy_to_nrrd(vol_4_1_0, 'Data/vol_4_1_0.nrrd')

vol_2 = np.concatenate((vol_0_1_0, vol_1_1_0, vol_2_1_0, vol_3_1_0, vol_4_1_0), axis=2)
# segmentation done
#%%
vol_0_0_1 = volume_1000[200:400, 0:200, 0:200]
vol_1_0_1 = volume_1000[200:400, 0:200, 200:400]
vol_2_0_1 = volume_1000[200:400, 0:200, 400:600]
vol_3_0_1 = volume_1000[200:400, 0:200, 600:800]
vol_4_0_1 = volume_1000[200:400, 0:200, 800:1000]

# md.numpy_to_nrrd(vol_0_0_1, 'Data/vol_0_0_1.nrrd')
# md.numpy_to_nrrd(vol_1_0_1, 'Data/vol_1_0_1.nrrd')
# md.numpy_to_nrrd(vol_2_0_1, 'Data/vol_2_0_1.nrrd')
# md.numpy_to_nrrd(vol_3_0_1, 'Data/vol_3_0_1.nrrd')
# md.numpy_to_nrrd(vol_4_0_1, 'Data/vol_4_0_1.nrrd')

vol_3 = np.concatenate((vol_0_0_1, vol_1_0_1, vol_2_0_1, vol_3_0_1, vol_4_0_1), axis=2)
# segmentation done
#%%
vol_0_1_1 = volume_1000[200:400, 200:400, 0:200]
vol_1_1_1 = volume_1000[200:400, 200:400, 200:400]
vol_2_1_1 = volume_1000[200:400, 200:400, 400:600]
vol_3_1_1 = volume_1000[200:400, 200:400, 600:800]
vol_4_1_1 = volume_1000[200:400, 200:400, 800:1000]

# md.numpy_to_nrrd(vol_0_1_1, 'Data/vol_0_1_1.nrrd')
# md.numpy_to_nrrd(vol_1_1_1, 'Data/vol_1_1_1.nrrd')
# md.numpy_to_nrrd(vol_2_1_1, 'Data/vol_2_1_1.nrrd')
# md.numpy_to_nrrd(vol_3_1_1, 'Data/vol_3_1_1.nrrd')
# md.numpy_to_nrrd(vol_4_1_1, 'Data/vol_4_1_1.nrrd')

vol_4 = np.concatenate((vol_0_1_1, vol_1_1_1, vol_2_1_1, vol_3_1_1, vol_4_1_1), axis=2)
# segmentation done
#%%
path = 'Data/Segmentation (nrrd)'

vol_11 = md.nrrd_to_numpy(os.path.join(path, 'vol_0_0_0.nrrd'))
vol_12 = md.nrrd_to_numpy(os.path.join(path, 'vol_1_0_0.nrrd'))
vol_13 = md.nrrd_to_numpy(os.path.join(path, 'vol_2_0_0.nrrd'))
vol_14 = md.nrrd_to_numpy(os.path.join(path, 'vol_3_0_0.nrrd'))
vol_15 = md.nrrd_to_numpy(os.path.join(path, 'vol_4_0_0.nrrd'))

vol_21 = md.nrrd_to_numpy(os.path.join(path, 'vol_0_1_0.nrrd'))
vol_22 = md.nrrd_to_numpy(os.path.join(path, 'vol_1_1_0.nrrd'))
vol_23 = md.nrrd_to_numpy(os.path.join(path, 'vol_2_1_0.nrrd'))
vol_24 = md.nrrd_to_numpy(os.path.join(path, 'vol_3_1_0.nrrd'))
vol_25 = md.nrrd_to_numpy(os.path.join(path, 'vol_4_1_0.nrrd'))

vol_31 = md.nrrd_to_numpy(os.path.join(path, 'vol_0_0_1.nrrd'))
vol_32 = md.nrrd_to_numpy(os.path.join(path, 'vol_1_0_1.nrrd'))
vol_33 = md.nrrd_to_numpy(os.path.join(path, 'vol_2_0_1.nrrd'))
vol_34 = md.nrrd_to_numpy(os.path.join(path, 'vol_3_0_1.nrrd'))
vol_35 = md.nrrd_to_numpy(os.path.join(path, 'vol_4_0_1.nrrd'))

vol_41 = md.nrrd_to_numpy(os.path.join(path, 'vol_0_1_1.nrrd'))
vol_42 = md.nrrd_to_numpy(os.path.join(path, 'vol_1_1_1.nrrd'))
vol_43 = md.nrrd_to_numpy(os.path.join(path, 'vol_2_1_1.nrrd'))
vol_44 = md.nrrd_to_numpy(os.path.join(path, 'vol_3_1_1.nrrd'))
vol_45 = md.nrrd_to_numpy(os.path.join(path, 'vol_4_1_1.nrrd'))

vol_z2 = np.concatenate((vol_41, vol_42, vol_43, vol_44, vol_45), axis=2)
vol_z1 = np.concatenate((vol_31, vol_32, vol_33, vol_34, vol_35), axis=2)
vol_y1 = np.concatenate((vol_21, vol_22, vol_23, vol_24, vol_25), axis=2)
vol_y0 = np.concatenate((vol_11, vol_12, vol_13, vol_14, vol_15), axis=2)

vol = np.zeros((400, 400, 1000))
#%%
vol[0:200, 0:200, 0:1000] = vol_y0
vol[200:400, 0:200, 0:1000] = vol_z1
vol[0:200, 200:400, 0:1000] = vol_y1
vol[200:400, 200:400, 0:1000] = vol_z2
#%%
#md.numpy_to_nrrd(vol, 'open_me.nrrd')
np.save('ground_truth.npy', vol)
#%%
whole_vol = np.zeros((400, 400, 1000))
whole_vol[0:200, 0:200, 0:1000] = vol_1
whole_vol[200:400, 0:200, 0:1000] = vol_3
whole_vol[0:200, 200:400, 0:1000] = vol_2
whole_vol[200:400, 200:400, 0:1000] = vol_4

np.save('whole_volume.npy', whole_vol)





