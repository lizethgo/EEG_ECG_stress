#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 23:26:00 2020

scatters number_batches (belonging to stress) vs ECG % increase for 24 cases
we can oserve that some individuals has a larger number of batches (stress) + larger ECG % -- more stressed 


@author: lizeth
"""

#### This files reads the csv files generated in classification_3.py
#### It gathers cvs files from all cases generated in classification_3.py
#### It gets mean of all statistical data to observe which group is the most prevalent in each batch 
#### This will do the histograms based on each experiment
#### cases 13, 15, 17 excluded
import numpy as np
import pywt
import matplotlib.pyplot as plt
import matplotlib
import pyedflib
#from scipy.fftpack as fft
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
from matplotlib.backends.backend_pdf import PdfPages

n_channels = 19
n_level = 5
case_ = 18

### data shape is data_[n_levels*n_channels, n_batches]
### data_[0:19,:] refers to all channels, level = 1
### data_[20:39,:] refers to all channels, level = 2 and so on ..

### re-arrangin data to data_arr[n_channels, n_levels, n_batches]
n_case = 24
case_init = 6
case_end = 29
n_batches = 83
tot_data = np.zeros((n_case,n_level, n_channels, n_batches))
	
for i in range(case_init, case_end+1):
	data_ = pd.read_csv('GROUPS_case_{}.csv'.format(i), delimiter=',', header=None)
	data_ = data_.iloc[:,0:83].values
	for j in range(0,n_channels):
		#print((i*n_level)+n_level)
		tot_data[i-case_init,:,j,:] = data_[j*n_level:(j*n_level)+n_level,:] 

### vector will store the values for all gropus and it will serve to plot a histogram for all cases
### vector shape will be vector[cases, batches] for each channel for each level
### 
		
pdf_file=PdfPages('CLASSIFICATION_ALL_CASES_.pdf')
ifig = 1
E = [12,19,31,62,79,81,84] ### This is the list of Experiments
b_prev = 0
n_level = 3
n_channel = 11

#level 1 channel 10
# level 2 channel 4
# level 2 channel 6
# level 2 channel 8
# level 3 channel 12
# level 3 channel 11
# level 3 channel 7
# level 3 channel 1




number_batches = [None]*24
for case in range(0, n_case):
    vector = tot_data[case,n_level,n_channel,:] + 1
    vector2 = np.copy(vector)
    vector2 = np.where(vector2==1, 4, vector2)  # group 3
    vector2 = np.where(vector2==3, 1, vector2)  # group 1 -- stress
    vector2 = np.where(vector2==4, 3, vector2)
    
 
    
    #ax1.suptitle('Distribution per batch for channel {} and level {}'.format(j,i))
    b_prev = 0
    for x,b in enumerate(E):
        
        if x == 2:
            number_batches[case] = np.count_nonzero(vector2[b_prev:b] == 3)
        elif x>2:
            number_batches[case] = number_batches[case] + np.count_nonzero(vector2[b_prev:b] == 3)

        b_prev = b

dataset_3 = pd.read_csv("REFERENCIA_ECG_METRICS.csv", delimiter=' ', header=None)
dataset_3 = (dataset_3.iloc[5:29,:].values)
#avg_val = np.average(dataset_3,1)
max_val = np.average(dataset_3,1)


plt.scatter(max_val, number_batches)

    
# number_batches = np.zeros((24,7))
#for case in range(0, n_case):
#    vector = tot_data[case,n_level,n_channel,:] + 1
#    vector2 = np.copy(vector)
#    vector2 = np.where(vector2==1, 4, vector2)  # group 3
#    vector2 = np.where(vector2==3, 1, vector2)  # 
#    vector2 = np.where(vector2==4, 3, vector2)
#    print(vector2)
#    
# 
#    
#    #ax1.suptitle('Distribution per batch for channel {} and level {}'.format(j,i))
#    b_prev = 0
#    for x,b in enumerate(E):
#        print(np.count_nonzero(vector2[b_prev:b] == 3))
#        print('case and x', case, x)
#        number_batches[case, x] = np.count_nonzero(vector2[b_prev:b] == 2)
#        #print(number_batches)
#        b_prev = b
#
#
#dataset_3 = pd.read_csv("REFERENCIA_ECG_METRICS.csv", delimiter=' ', header=None)
#dataset_3 = (dataset_3.iloc[5:29,:].values)
#max_val = np.max(dataset_3,1)
#max_sort = np.sort(max_val)
#max_ind = np.argsort(max_val)
#
#
#for i in range(0,5):
#    plt.scatter(dataset_3[:,i+2], number_batches[:,i+2])
#    plt.show()
    



