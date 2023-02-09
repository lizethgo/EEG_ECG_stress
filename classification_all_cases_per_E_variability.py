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
	data_ = data_.iloc[:,0:n_batches].values
	for j in range(0,n_channels):
		#print((i*n_level)+n_level)
		tot_data[i-case_init,:,j,:] = data_[j*n_level:(j*n_level)+n_level,:] 

### vector will store the values for all gropus and it will serve to plot a histogram for all cases
### vector shape will be vector[cases, batches] for each channel for each level
###
#pdf_file=PdfPages('CLASSIFICATION_ALL_CASES.pdf')
#ifig = 1
E = [12,19,31,62,79,81,84] ### This is the list of Experiments



n_exp = 7
### tot_data[case, channel, level, batches]
var = np.zeros((n_case, n_level, n_channels, n_exp))
n_group = 3

for case_ in range(0,n_case):
        for i in range(0,n_level):
                for j in range(0, n_channels):
                        vector = tot_data[case_,i,j,:] + 1
                        b_prev = 0
 	
                        for x,b in enumerate(E):
                                unique, counts = np.unique(vector[b_prev:b], return_counts=True)
                                result = np.where(counts == np.amax(counts))
                                var[case_, i, j, x] = unique[result[0][0]]
                                b_prev = b

                                #print(unique, counts, unique[result[0][0]], x)


var_plot = np.zeros((n_level, n_channels, n_exp, n_group))

for i in range(0,n_level):
        for j in range(0, n_channels):
                for x in range(0, n_exp):
                        counts, bins = np.histogram(var[:, i, j, x], bins=[1,2,3,4])
                        #unique, counts = np.unique(var[:, i, j, x], return_counts=True)
                        #print(unique, counts)
                        #if len(counts) < 3:
                                #counts = np.insert(counts,np.shape(counts),np.zeros(n_group-len(counts)))
                                #print(counts)
                        var_plot[i,j,x,:]=counts

width = 0.5

for i in range(0,n_level):
        for j in range(0, n_channels):
                ax2 = plt.figure(0, figsize=(4,8))
                #ax3 = plt.figure(1, figsize=(8,2))
                #plt.bar(np.arange(0,n_exp), var_plot[i,j,:,0], color='red', edgecolor='white', width=0.9)
                #plt.bar(np.arange(0,n_exp), var_plot[i,j,:,1], bottom=var_plot[i,j,:,0], color='green', edgecolor='white', width=0.9)
                #plt.bar(np.arange(0,n_exp), var_plot[i,j,:,2], bottom=[i+j for i,j in zip(var_plot[i,j,:,0],var_plot[i,j,:,1])], color='blue', edgecolor='white',width=0.9)
                #plt.show()
                #for x in range(0, n_exp):
                ax2.add_subplot(n_channels,1,j+1)
                plt.bar(np.arange(0,n_exp), var_plot[i,j,:,0], color='red', edgecolor='white', width=width)
                plt.bar(np.arange(0,n_exp), var_plot[i,j,:,1], bottom=var_plot[i,j,:,0], color='green', edgecolor='white', width=width)
                plt.bar(np.arange(0,n_exp), var_plot[i,j,:,2], bottom=[i+j for i,j in zip(var_plot[i,j,:,0],var_plot[i,j,:,1])], color='blue', edgecolor='white',width=width) 
        plt.show()
                

                                

                                
                        








