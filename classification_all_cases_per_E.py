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
        
pdf_file=PdfPages('CLASSIFICATION_ALL_CASES_debug.pdf')
# ifig = 1
# E = [12,19,31,62,79,81,84] ### This is the list of Experiments
# b_prev = 0
# for i in range(0,n_level):
#     for j in range(0, n_channels):
#         vector = tot_data[:,i,j,:] + 1
#         plt.rc('ytick',labelsize=7)
#         ax1 = plt.figure(ifig, figsize=(20,3))
#         ax1.suptitle('Distribution per batch for channel {} and level {}'.format(j,i))
#         b_prev = 0
#         for x,b in enumerate(E):
#             #print(x)
#             #print(b)
#             ax1.add_subplot(1,len(E),x+1)
#             #hist, bins = np.histogram(vector[:,b],bins=[0,1,2])
#             #plt.bar(bins[-1],hist, width=0.5,color='#0504aa')
#             print(vector[:,b_prev:b].flatten())
#             print('DEBUS MSG b_prev is: ',b_prev)
#             nbins = 3
#             colors = plt.get_cmap('winter')(np.linspace(0,1,nbins))
#             
#             n, bins, patches = plt.hist(vector[:,b_prev:b].flatten(), bins=3, color='#0504aa', alpha=0.7, rwidth=0.85)
#             #plt.rc('xtick',lebelsize=8)
#             idx = 0
#             for patch, color in zip(patches, colors):
#                 patch.set_facecolor(color)
#             b_prev = b

#         #plt.show()
#         pdf_file.savefig(ax1)
#         ifig = ifig + 1
# pdf_file.close()





E = [12,19,31,62,79,81,84] ### This is the list of Experiments
b_prev = 0


fig, ax1 = plt.subplots(nrows=n_level, ncols =len(E), sharex = True, figsize=(12,8))
#ax1.suptitle('Distribution per batch for channel {} and level {}'.format(j,i))
plt.rc('ytick',labelsize=7)
cmaps = ['winter', 'viridis', 'plasma', 'rainbow', 'inferno']


for i in range(0,n_level):
        vector = tot_data[:,i,:,:] + 1
                
        b_prev = 0
        for x,b in enumerate(E):
            #print(x)
            #print(b)
            #ax1.add_subplot(1,len(E),x+1)
            #hist, bins = np.histogram(vector[:,b],bins=[0,1,2])
            #plt.bar(bins[-1],hist, width=0.5,color='#0504aa')
            print(vector[:,:,b_prev:b].flatten())
            print('DEBUS MSG b_prev is: ',b_prev)
            nbins = 3
            #colors = plt.get_cmap('winter')(np.linspace(0,1,nbins))
            colors = plt.get_cmap('viridis')(np.linspace(0,1,nbins))
            colors = colors[::-1]
            colors = ['red', 'black', 'blue']
        
            n, bins, patches = ax1[i,x].hist(vector[:,:,b_prev:b].flatten(), bins=3, color='#0504aa', alpha=0.7, rwidth=0.85)
            ax1[i,x].tick_params(axis='y', labelsize=12 )
            ax1[i,x].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            ax1[i,x].yaxis.offsetText.set_fontsize(12)
            plt.tight_layout()
            #plt.rc('xtick',lebelsize=8)
            idx = 0
            
            if x == 0:
                ax1[i,x].set_ylabel('level {}'.format(i+1), fontsize=18)
            if i == 4:
                ax1[i,x].set_xlabel('E{}'.format(x+1), fontsize=18)
            for patch, color in zip(patches, colors):
                patch.set_facecolor(color)
            b_prev = b

        #plt.show()
        #pdf_file.savefig(ax1)
        #ifig = ifig + 1
pdf_file.close()

