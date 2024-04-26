import numpy as np
import scipy as sp
import manage_data as md
import matplotlib.pyplot as plt

# This is a Python implementation of the NetMets geometry metric:
#   https://doi.org/10.1186/1471-2105-13-S8-S7

# This implementation is based on work done by Dr. Jiaming Guo

def gaussian(X, sigma):
    return np.exp(-0.5 * (X ** 2 / sigma ** 2))

# generate point clouds representing both networks
# HELIA: create an Nx3 array storing the 3D positions of each centerline pixel
skeleton = md.nrrd_to_numpy('lee.nrrd')
gr_skeleton = md.nrrd_to_numpy('centerline.nrrd')

#%%
# find the voxel position of each centerline pixel
gr_pixels = np.argwhere(gr_skeleton).astype(np.float32)
skeleton_pixels = np.argwhere(skeleton).astype(np.float32)
# map all positions to [0, 1]
gr_pixels[:, 0] /= 1023.0
gr_pixels[:, 1:3] /= 511.0

skeleton_pixels /= 199.0
#%%

#P_T = np.array(T.pointcloud(sigma/subdiv))
#P_GT = np.array(GT.pointcloud(sigma/subdiv))
P_T = skeleton_pixels
P_GT = gr_pixels

sigma = 10
#tunable constants
subdiv = 4                                   # fraction of sigma used to sample each network
shadow = 10                                  # thickness of the shadow network when displaying geometric results

# generate KD trees for each network
GT_tree = sp.spatial.cKDTree(P_GT)
T_tree = sp.spatial.cKDTree(P_T)

# query each KD tree to get the corresponding geometric distances
[T_dist, _] = GT_tree.query(P_T)
[GT_dist, _] = T_tree.query(P_GT)

# convert distances to Gaussian metrics
T_metric = gaussian(T_dist, sigma)
GT_metric = gaussian(GT_dist, sigma)

#calculate the TPR and FPR
print("FNR = " + str(1 - np.mean(GT_metric)))
print("FPR = " + str(1 - np.mean(T_metric)))

plt.subplot(1, 2, 1)
plt.scatter(P_GT[:, 0], P_GT[:, 1], s=sigma*shadow, c="grey")
plt.scatter(P_T[:, 0], P_T[:, 1], s=sigma, c=T_metric, cmap = "plasma")

plt.subplot(1, 2, 2)
plt.scatter(P_T[:, 0], P_T[:, 1], s=sigma*shadow, c="grey")
plt.scatter(P_GT[:, 0], P_GT[:, 1], s=sigma, c=GT_metric, cmap = "plasma")