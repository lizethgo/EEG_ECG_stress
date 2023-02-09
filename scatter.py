# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 13:21:18 2021

@author: 20195088
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 23:26:00 2020

same as classification_EEG_per_case but per experiment 

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
import matplotlib.patches as mpatches

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
n_level = 0
n_channel = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]  # most relevant channels

#level 1 channel 10
# level 2 channel 4
# level 2 channel 6
# level 2 channel 8
# level 3 channel 12
# level 3 channel 11
# level 3 channel 7
# level 3 channel 1


dataset_3 = pd.read_csv("REFERENCIA_ECG_METRICS.csv", delimiter=' ', header=None)


colors1=['red', 'green', 'green', 'green', 'green', 'green', 'green', 'orange', 'orange', 'green', 'green', 'orange', 'orange', 'green','orange', 'orange','green', 'red','green', 'green','green', 'green', 'red', 'red']
colors1.reverse() 
    
    #ax1.suptitle('Distribution per batch for channel {} and level {}'.format(j,i))
fig, ax = plt.subplots(3,2, figsize=(10,10))
fig0, ax1 = plt.subplots(3,2, figsize=(10,10))
b_prev = 0
for x,b in enumerate(E):
    print(x)
    
    number_batches = [0]*24
    for case in range(0, n_case):
        for y,c in enumerate(n_channel):
            vector2 = tot_data[case,n_level,c,:] + 1        
            number_batches[case] = np.count_nonzero(vector2[b_prev:b] == 3) + number_batches[case]
            #number_batches[case] = number_batches[case] / (b-b_prev)
            #print(number_batches[case])
        if x == 0:
            colors = 'blue'

            
        number_batches[case] = number_batches[case] / ((b-b_prev)*len(n_channel))
    b_prev = b
    
    
    
    #print(number_batches)
    dataset_3 = pd.read_csv("REFERENCIA_ECG_METRICS.csv", delimiter=' ', header=None)
    dataset_3 = (dataset_3.iloc[5:29,x].values)
    #max_val = np.max(dataset_3)

    
    
    #plt.figure(x+1)
    #plt.scatter(dataset_3, number_batches)
    #plt.xlabel('R-R increase (%)')
    #plt.ylabel('# of occurrences of Group 3 (stress)')
    #plt.xticks(fontsize=12)
    #plt.yticks(fontsize=12)
    #plt.title('Experiment_{}'.format(x))
    
    if x != 1:
    
        if x <= 3:
            if x == 0:
                x_1 = x
            else:
                x_1 = x - 1
            x_2 = 0
        else:
            x_1 = x - 4
            x_2 = 1
            
        
        ax[x_1, x_2].scatter(dataset_3, number_batches, color=colors)
        ax[x_1, x_2].set_title('Experiment_{}'.format(x+1),  fontsize=14)
        ax[x_1, x_2].grid(linestyle=':', linewidth='0.5', color='red')
        ax[x_1, x_2].set_xlim(xmin=-20, xmax=40)
        ax[x_1, x_2].set_ylim(ymin=0, ymax=1)
        
        if x == 3 or x == 6:
            ax[x_1, x_2].set_xlabel(r'$\mu_{ECG}$', fontsize=12)
        
        if x==0 or x==2 or x==3:
            ax[x_1, x_2].set_ylabel(r'$\mu_{EEG}$', fontsize=12)   
            
        ### metrics for stress
            
        metrics = 0.5*(dataset_3/100) + 0.5*np.asarray(number_batches)
        N = np.arange(0,24)
            
        ax1[x_1, x_2].barh(N, metrics, color=colors1)
        ax1[x_1, x_2].set_title('Experiment_{}'.format(x+1),  fontsize=14)
        ax1[x_1, x_2].grid(linestyle=':', linewidth='0.5', color='red')
        ax1[x_1, x_2].set_xlim(xmin=0, xmax=1)
        ax1[x_1, x_2].set_yticks(N)
        ax1[x_1, x_2].set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24'],fontsize=10)
        ax1[x_1, x_2].grid(linestyle=':', linewidth='0.5', color='black')
        green_patch = mpatches.Patch(color='green', label='Mild')
        red_patch = mpatches.Patch(color='red', label='Moderate')
        yellow_patch = mpatches.Patch(color='orange', label='Severe')
        #ax[x_1, x_2].legend(handles=[green_patch, yellow_patch, red_patch, ])
        ax1[x_1, x_2].legend(handles=[green_patch, yellow_patch, red_patch, ])
        if x == 3 or x == 6:
            ax1[x_1, x_2].set_xlabel(r'$\mu_{stress}$', fontsize=12)
        
        if x==0 or x==2 or x==3:
            ax1[x_1, x_2].set_ylabel('cases', fontsize=10)


    

            
    
        

    
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
    



