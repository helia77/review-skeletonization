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
csfont = {'fontname':'Times New Roman'}

DATA = 'KESM'

alphac_scores = {"KESM": 'alpha2_c50_fscores.npy', "LSM": 'alpha_c_LSM_fscores.npy', "Micro": 'alpha2_c50_micro_fscores.npy'}

beta_scores = {"KESM": 'beta_all_fscore.npy', "LSM": 'beta_all_fscore_LSM.npy', "Micro": 'beta_all_fscore_micro.npy'}
beta_precisions = {"KESM": 'beta_all_precision.npy', "LSM": 'beta_all_precision_LSM.npy', "Micro": 'beta_all_precision_micro.npy'}
beta_recalls = {"KESM": 'beta_all_recall.npy', "LSM": 'beta_all_recall_LSM.npy', "Micro": 'beta_all_recall_micro.npy'}

tau_scores = {"KESM": 'tau_all_fscores.npy', "LSM": 'tau_all_fscore_LSM.npy', "Micro": 'tau_all_fscore_micro.npy'}
tau_precisions = {"KESM": 'tau_all_precision.npy', "LSM": 'tau_all_precision_LSM.npy', "Micro": 'tau_all_precision_micro.npy'}
tau_recalls = {"KESM": 'tau_all_recall.npy', "LSM": 'tau_all_recall_LSM.npy', "Micro": 'tau_all_recall_micro.npy'}

#%%
# alpha and c values
alphac_fscores = np.load('Final npy Data/Figure 2/' + DATA + '/' + alphac_scores[DATA])
alpha_range = np.load('Final npy Data/Figure 2/alpha_range.npy')   #0-2
c_range = np.load('Final npy Data/Figure 2/c_range.npy')           #0-50

# beta values
beta_fscores = np.load('Final npy Data/Figure 2/' + DATA + '/' + beta_scores[DATA])
beta_precision = np.load('Final npy Data/Figure 2/' + DATA + '/' + beta_precisions[DATA])[1:]
beta_recall = np.load('Final npy Data/Figure 2/' + DATA + '/' + beta_recalls[DATA])[1:]
beta_range = np.load('Final npy Data/Figure 2/beta_range.npy')

tau_all_fscores = np.load('Final npy Data/Figure 3/' + DATA + '/' + tau_scores[DATA])
tau_all_precision = np.load('Final npy Data/Figure 3/' + DATA + '/' + tau_precisions[DATA])
tau_all_recall = np.load('Final npy Data/Figure 3/' + DATA + '/' + tau_recalls[DATA])
tau_range = np.load('Final npy Data/Figure 3/' + DATA + '/tau_range.npy')
#%%
plt.figure(2, figsize=(10,10))
im = plt.imshow(alphac_fscores, cmap='YlGnBu', vmin=0, vmax=1.05)
plt.contour(alphac_fscores, levels=[0.45, 0.65, 0.75], linewidths=2.5, colors='black')
scale = 1.1                            # the scale size for micro plot (due to colorbar size difference)
# plot colorbar
if DATA == "Micro":
    cbar = plt.colorbar(im, fraction=0.0463, pad=0.03)                             # adding the colobar on the right
    cbar.set_label('DICE', fontsize=30/scale, **csfont)
    cbar.ax.tick_params(labelsize=20/scale)

# set ticks values
step = 9
xtick_positions = np.arange(0, len(alpha_range), step)
ytick_positions = np.arange(0, len(c_range), step)
alpha_values = [f'{alpha_range[i]:.3f}' for i in xtick_positions]
alpha_values[0], alpha_values[-1] = '0', '2'
plt.xticks(xtick_positions, alpha_values, fontsize=30/scale, rotation=90, **csfont)
plt.yticks(ytick_positions, [f'{c_range[i]:.0f}' for i in ytick_positions], fontsize=30/scale, **csfont)

# draw the maximum f-score on plot
idxmax_c, idxmax_alpha = np.unravel_index(np.argmax(alphac_fscores), alphac_fscores.shape)
plt.scatter(idxmax_alpha, idxmax_c, marker='o', color='red', s=150/scale, label='Max DICE')

# Annotate the maximum F-score point
plt.annotate(f'$\\alpha$={0.2}, $c$={c_range[idxmax_c]:.2f}\nDICE={np.max(alphac_fscores):.1f}',
              xy=(idxmax_alpha+1, idxmax_c), weight='bold',
              xytext=(idxmax_alpha + 42, idxmax_c+ 32),
              arrowprops=dict(edgecolor='grey', arrowstyle='->'),
              bbox=dict(boxstyle="round,pad=0.3", edgecolor='grey', facecolor='white', alpha=1),
              ha='center',  # Horizontal alignment
              va='center',  # Vertical alignment
              fontsize=25/scale, **csfont)

# details
plt.legend(handletextpad=0.2,
           loc='upper right', prop={"family":'Times New Roman', "size":"22.32", "weight":"bold"})
plt.xlabel('$\\alpha$', fontsize=35/scale, **csfont)
plt.ylabel('$c$', fontsize=35/scale, **csfont)
plt.xticks(**csfont)
plt.yticks(**csfont)
plt.tight_layout()
plt.gca().invert_yaxis()
#plt.subplots_adjust(left=0.1, right=0.9, top=1, bottom=0)

print('Highest occurs at alpha', alpha_range[idxmax_alpha], 'c ', c_range[idxmax_c], '\nDICE=',
      alphac_fscores[idxmax_c, idxmax_alpha])

#%%
# For beta
plt.figure(2, figsize=(6,6))

# Plot the F-score contour plot
space = np.linspace(0, 1, 100)
PR, RE = np.meshgrid(space, space)
denum = PR + RE
denum[np.where(PR + RE == 0.0)] = np.nextafter(0, 1)
tau_fscores_contour = (2*PR*RE)/(denum)
tau_fscores_contour = np.nan_to_num(tau_fscores_contour, nan=0.0)
#CS = plt.contourf(RE, PR, tau_fscores_contour, levels=np.linspace(0, 1, 11), cmap='OrRd')
CS = plt.contour(RE, PR, tau_fscores_contour, cmap='OrRd', linewidths=4.0)
manual_locations = [(0.15,0.15), (0.3, 0.3), (0.4, 0.4), (0.6, 0.6), (0.78, 0.7), (0.9, 0.9)]
plt.clabel(CS, colors='k', inline=True, fontsize=13, manual=manual_locations)

#plt.plot(beta_range, beta_fscores)
plt.plot(beta_recall, beta_precision, linewidth=4, color='k')

# Plot the maximum f-score point
idxmax = np.argmax(beta_fscores)
plt.scatter(beta_recall[idxmax], beta_precision[idxmax], marker='o', color='k', s=90, label='Max DICE')

# Plot the dashed lines
# plt.plot([beta_recall[idxmax], beta_recall[idxmax]], [0, beta_precision[idxmax]], 'k', linestyle='dashed')
# plt.plot([0, beta_recall[idxmax]], [beta_precision[idxmax], beta_precision[idxmax]], 'k', linestyle='dashed')

# Annotate the maximum F-score point
plt.annotate(f'$\\beta$={beta_range[idxmax]:.2f}\nDICE={np.max(beta_fscores):.3f}',
              xy=(beta_recall[idxmax], beta_precision[idxmax]), weight='bold',
              xytext=(beta_recall[idxmax] - 0.31, beta_precision[idxmax] - 0.07),
              arrowprops=dict(facecolor='black', arrowstyle='->'),
              bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'),
              ha='center',  # Horizontal alignment
              va='center',  # Vertical alignment
              fontsize=18, **csfont)

print('Highest occurs at beta ', beta_range[idxmax], 'DSC:', beta_fscores[idxmax])
plt.xticks(fontsize=18, **csfont)
plt.yticks(fontsize=18, **csfont)

plt.xlabel('Recall', fontsize=25, **csfont)
plt.ylabel('Precision', fontsize=25, **csfont)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.legend(handletextpad=0.2, borderpad=0.25, handlelength=1.1,
           loc=(0.01, 0.01), prop={"family":'Times New Roman', "size":"18", "weight":"bold"})
plt.show()

# plt.figure(3)
# plt.plot(beta_range, beta_fscores)
# plt.scatter(beta_range[idxmax], beta_fscores[idxmax], marker='o', color='red', s=40)
#%%
# Figure (3) - tau
plt.figure(3, figsize=(6,6))

# plot the F-score contour plot
space = np.linspace(0, 1, 100)
PR, RE = np.meshgrid(space, space)
denum = PR + RE
denum[np.where(PR + RE == 0.0)] = np.nextafter(0, 1)
tau_fscores_contour = (2*PR*RE)/(denum)
tau_fscores_contour = np.nan_to_num(tau_fscores_contour, nan=0.0)
# CS = plt.contourf(RE, PR, tau_fscores_contour, levels=np.linspace(0, 1, 11), cmap='OrRd')
CS = plt.contour(RE, PR, tau_fscores_contour, cmap='OrRd', linewidths=4.0)
manual_locations = [(0.15,0.15), (0.3, 0.3), (0.4, 0.4), (0.6, 0.6), (0.65, 0.87), (0.9, 0.9)]
plt.clabel(CS, colors='k', inline=True, fontsize=13, manual=manual_locations)

plt.plot(tau_all_recall, tau_all_precision, linewidth=4, color='k')

# plot the maximum F-score point
plt.xlabel('Recall', fontsize=25, **csfont)
plt.ylabel('Precision', fontsize=25, **csfont)
idxmax = np.argmax(tau_all_fscores)
plt.scatter(tau_all_recall[idxmax], tau_all_precision[idxmax], marker='o', color='k', s=90, label='Max DICE')

# Annotate the maximum F-score point
plt.annotate(f'$\\tau$ = {tau_range[idxmax]:.2f}',
              xy=(tau_all_recall[idxmax], tau_all_precision[idxmax]), weight='bold',
              xytext=(tau_all_recall[idxmax] - 0.23, tau_all_precision[idxmax] - 0.066),
              arrowprops=dict(facecolor='black', arrowstyle='->'),
              fontsize=19, **csfont)

# Set the plot to start from (0,0)
#plt.xlim(0, max(tau_all_recall)+1.05)
#plt.ylim(0, max(tau_all_precision)+1.05)
plt.xticks(fontsize=18,**csfont)
plt.yticks(fontsize=18, **csfont)
print('Highest occurs at tau ', tau_range[idxmax], 'DICE:', tau_all_fscores[idxmax])
plt.tight_layout()
plt.legend(handletextpad=0.2, borderpad=0.25, handlelength=1.1,
           loc=(0.01, 0.01), prop={"family":'Times New Roman', "size":"18", "weight":"bold"})
plt.grid(alpha=0.3)
plt.show()

#%%
plt.plot(tau_range, tau_all_fscores)