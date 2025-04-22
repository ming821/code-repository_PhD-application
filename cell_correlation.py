# -*- coding: utf-8 -*-
"""

Script for Computational Neuroscience-Encoding and Decoding course, day6


    
Caution:
*******
Due to shared variables and dataset,the code is structured to run in order,
proceeding from the top to the bottom.
*******

Date: 12/12/2022
By zhiming.jia
"""
#%% set environment    
import numpy as np                       # import all the functions that will be used
import scipy                             # in the subsequent steps.
import matplotlib.pyplot as plt
#%% 6.1 Crosscorrelation cell1&cell2

data = scipy.io.loadmat('Day_06_1')      # load data
spikes = data['spikes'].squeeze()        # extract cell 1 and cell2

spike_cell1 = spikes[0]*1000             # conver seconds to milliseconds                 
spike_cell2 = spikes[1]*1000
bin = 2                                  # bin width of 2 milliseconds.
bin_edges = np.arange(-75,75+bin,bin)    # bin edges, -75ms to 75ms with the
                                         # fixed bin width value
bin_centers = bin_edges[0:-1] + bin/2    # bin centers
spike_num = np.zeros(len(bin_centers))   # zeros array for storing histo values
                                         # in loop

for i in range(len(spike_cell1)):        # loop over all spikes in cell 1
    spike_shift = spike_cell2-spike_cell1[i]
                                         # shiftted spike times 
    hist,_ = np.histogram(spike_shift,bins=bin_edges)
                                         # histogram into bin edges
    spike_num += hist                    # summation
    
spike_corre = spike_num/len(spike_cell1)/(bin/1000) # calculate correlation

plt.figure(figsize=(8,6))                # creat figure with 8*6 aspect ratio  
plt.plot(bin_centers,spike_corre,'b',alpha=0.6)
plt.xlabel('ΔT[ms]',fontsize=12)     
                                         # label the x-axis
plt.ylabel('Spikerate(spikes/s)',fontsize=12)         
                                         # label the y-axis                                    
plt.title('Correlation cell 2-1',fontsize=15)     # name the title
                                         
plt.xlim(bin_centers[0],bin_centers[-1]) # set x-axis limits                          
plt.show()                              

#%% 6.2 Crosscorrelation-all 4 cells
m = 1                                    # indice used for subplot, grown in loop 
plt.figure(figsize=(16,16))                # define figure size: 8*8 aspect ratio  
for a in range(len(spikes)):             # loop over 4 cells
    for b in range(len(spikes)):
        spike_num = np.zeros(len(bin_centers))   
                                         # zeros array for storing histo values
        spike_cell_r = spikes[a]*1000    # reference cell
        spike_cell_c = spikes[b]*1000    # cell compare with reference   
             
        for i in range(len(spike_cell_r)):    
                                         # loop over all spikes in reference cell
            spike_shift = spike_cell_c-spike_cell_r[i]
                                         # shiftted spike times 
            hist,_ = np.histogram(spike_shift,bins=bin_edges)
                                         # histogram into bin edges
            spike_num += hist            # summation
        spike_corre = spike_num/len(spike_cell1)/(bin/1000) 
                                         # calculate correlation
        plt.subplot(4,4,m)               # subplot
        plt.plot(bin_centers,spike_corre,'b',alpha=0.6)
        plt.xticks([])                   # remove x ticks
        plt.title(f'cell {b+1}-{a+1}',fontsize=12)
                                         # name the title with the given cell
        plt.xlim(bin_centers[0],bin_centers[-1])   
                                         # set x limits
        if m in(1,5,9,13):               # only lable y-axis in specific subplot
            plt.ylabel('Spikerate(spikes/s)',fontsize=10)
        if m>=13:                        # only lable x-axis in bottom panel
            plt.xlabel('ΔT[ms]',fontsize=10)
            plt.xticks([-75,-30,0,30,75])# set x ticks in bottom panel
        m += 1
plt.show()

#%% 6.3 Autocorrelation
bin = 0.25                               # new bin width,0.25 milliseconds
bin_edges = np.arange(0.25,50+bin,bin) 
                                         # bin edges
bin_centers = bin_edges[0:-1] + bin/2    # bin centers
spike_num = np.zeros(len(bin_centers))   # zeros array for storing histo values

plt.figure(figsize=(10,6))               # define figure size: 8*8 aspect ratio

for a in range(len(spikes)):             # loop over all 4 cells
    spike_cell = spikes[a]*1000          # extract spike from same cell 
    spike_num = np.zeros(len(bin_centers)) 
    for i in range(len(spike_cell)):     # calculate the autocorrelation
        spike_shift = spike_cell-spike_cell[i]
                                         # shiftted spike times 
        hist,_ = np.histogram(spike_shift,bins=bin_edges)
                                         # histogram into bin edges
        spike_num += hist                # summation
    spike_corre = spike_num/len(spike_cell1)/(bin/1000) 
    
    plt.subplot(2,2,a+1)                 # 4 subplots for 4 cells 
    plt.plot(bin_centers,spike_corre,'b')
    plt.xlabel('ΔT-cell1 [ms]',fontsize=12)     
                                         # label the x-axis
    plt.ylabel('Spikerate-Cell2 (spikes/s)',fontsize=12)          
                                         # label the y-axis   
    plt.xlim(bin_centers[0],bin_centers[-1])   
                                         # set x limits                                     
    plt.title(f'AutoCorre cell-{a+1}',fontsize=15)
    
plt.tight_layout()                       # tight layout figure
plt.show()