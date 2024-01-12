# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:48:08 2024

@author: helioum
"""

import numpy as np
import frangi as frg
import matplotlib.pyplot as plt
import time
import metric as mt
import cthresholding as cp_th
import plot_curves as pltc
import sklearn.metrics as sk
import pandas as pd
import manage_data as md

#%%
results_path = 'C:/Users/helioum/Desktop/CNN_data'
gr_truth_path = 'C:/Users/helioum/Desktop/annotated_data'

results = md.load_images(results_path, 'all', stack=True, grayscale=True)
gr_truth = md.load_images(gr_truth_path, 'all', stack=True, grayscale=True)

#%%

met = mt.metric(gr_truth, results)
