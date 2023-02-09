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
n_case = 3
case_init = 6
case_end = 8
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
fig, ax = plt.subplots(nrows=len(E), ncols =n_case, sharex = True)
x = [1,2,3,4]
labels = ['1', '2', '3']
plt.xticks(x, labels)

for case in range(0, n_case):
    vector2 = tot_data[case,n_level,n_channel,:] + 1
    #vector2 = np.copy(vector)
    #vector2 = np.where(vector2==1, 4, vector2)
    #vector2 = np.where(vector2==3, 1, vector2)
    #vector2 = np.where(vector2==4, 3, vector2)
    
    
    #ax1.suptitle('Distribution per batch for channel {} and level {}'.format(j,i))
    b_prev = 0
    for x,b in enumerate(E):
        
        ax[x,case].set_ylim(ymin=0, ymax=(b-b_prev))
        ax[x,case].yaxis.set_major_locator(plt.MaxNLocator(1))
        nbins = 3
        colors = plt.get_cmap('winter')(np.linspace(0,1,nbins))
        #colors = ['k', 'r', 'b', 'r']
        n, bins, patches = ax[x,case].hist(vector2[b_prev:b].flatten(), bins=[1,2,3,4], color='#0504aa', alpha=0.7, rwidth=0.85, align='left')
        idx = 0
        for patch, color in zip(patches, colors):
            patch.set_facecolor(color)
        b_prev = b

#ax.set_xlim(xmin=1, xmax=35086)

#pdf_file.savefig(fig)
plt.savefig('test.eps', format='eps')
#ifig = ifig + 1
#fig.tight_layout()
plt.show()
pdf_file.close()

