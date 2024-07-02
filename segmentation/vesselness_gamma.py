# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:24:27 2024

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage
from scipy.signal import find_peaks


csfont = {'fontname':'Times New Roman'}

N = 4000
S = 4
ds = 1
gamma = 2
size = 4 * (pow(2, S) - 1) + pow(2, S-1) * 3                         #((S * (S+1))) *2 + 5*S
x = np.linspace(0, size, N)
dx = x[1] - x[0]
vessels = np.zeros((N,))

chosen_range = [1, 2, 4, 8]
norm_range = range(1, S+1)

# Create the vessels simulation (step functions)
x0 = 0
for xi in range(N):
    sum_si = 0
    for i, si in enumerate(chosen_range):
        before =  4*(pow(2, i+1) - 1) - pow(2, i) #2 * pow(si*ds, 2) + si*ds
        if x[xi] > before and x[xi] < before + si*ds:
            vessels[xi] = 1

# Plot the step functions in the 1st and 4th plot        
fig, axs = plt.subplots(5, 1, figsize=(8, 10))
axs[0].plot(x, vessels, color='k')  
axs[4].plot(x, vessels, color='k')
axs[0].set_title('(a) Sample Step Response and Gaussian Blur',**csfont)

s = np.zeros((3, S, N))
blurred = np.zeros((S, N))

# Store each second derivative with scale Gamma=i+1 in s[i]
# s size is 3 x number of sigmas x N
for si in range(S): 
    sigma = (si+1)*ds/dx
    blurred[si, :] = sp.ndimage.gaussian_filter(vessels, sigma, order=0)
    axs[0].plot(x, blurred[si, :], label='$\\sigma$='+str(si+1))
    s[0, si, :] = pow(sigma, 1) * sp.ndimage.gaussian_filter(vessels, sigma, order=2)
    s[1, si, :] = pow(sigma, 2) * sp.ndimage.gaussian_filter(vessels, sigma, order=2)
    s[2, si, :] = pow(sigma, 3) * sp.ndimage.gaussian_filter(vessels, sigma, order=2)


min_points = []                                                                     # includes the minimum point for each Gamma (3)
V_values = np.zeros((3, S, N))
max_points = np.zeros((3, N))
min_all = np.zeros((3, N))

# Find the peak values (minimum in this case) of each plot (Gamma)
for j in range(3):                                                                  # for each Gamma
    max_points[j] = np.max(-s[j], axis=0)                                           # get the maximum of all outputs for a gamma
    peaks = find_peaks(max_points[j], height=0.005)[0]                              # get the peak points of the maximum
    min_points.append(peaks)                                                        # save the peak points for each gamma
    min_all[j] = np.min(s[j], axis=0)                                               # get the minimum curve of all outputs for a gamma


# Calculate the vesselness function equivalent 1 - exp(-S^2/2c^2) for each Gamma and sigma
c_values = [0.0022, 0.22, 45]
final_value = []
for j in range(3):
    for si in range(S):
        S2 =  pow(s[j, si], 2)                                                # making each element squared
        V_values[j, si] = 1 - np.exp(-(S2)/(2*pow(c_values[j], 2)))
        
    # Get the max V for each gamma
    final_value.append(np.max(V_values[j], axis=0))
    
min_points[0] = np.delete(min_points[0], 3)         # a peak that is unnecessary

# Plot the second derivatives for each Gamma + the minimum response dashed line
for j in range(3):
    # Plot the dashed line minimum
    axs[j+1].plot(x[min_points[j]], min_all[j, min_points[j]], 'k', linestyle='dashed')
    for i in range(S):
        # Plot the 2nd derivatives of the Gaussians for each Gamma
        axs[j+1].plot(x, s[j, i, :], label='$\\sigma$='+str(i+1))
        
# Plot the last curve: the Maximum Vesselness response
for j in range(3):
    axs[4].plot(x, final_value[j], label='$\\gamma$='+str(j+1)+', c='+str(c_values[j]))

# Plot settings
titles = ['(a) Step Function with Blurred Gaussian', '(b) 2nd-Order Gaussian, scaled $\\gamma$=1', 
          '(c) 2nd-Order Gaussian, scaled $\\gamma$=2', '(d) 2nd-Order Gaussian, scaled $\\gamma$=3',
          '(e) Step Function and maximum 1-D vesselness function']
for i in range(5):
    axs[i].legend(loc='lower right', fontsize='x-small')
    axs[i].set_title(titles[i],**csfont)
    axs[i].tick_params(axis='both', labelsize=8)

axs[4].legend(loc='upper right', fontsize='x-small')
plt.tight_layout()
plt.show()










