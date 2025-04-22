# -*- coding: utf-8 -*-

"""
Script for Computational Neuroscience-Encoding and Decoding course, caontains
Raster and PSTH plot functions.

     
Date: 5/12/2022
By zhiming.jia
"""
#%% set environment        
import numpy as np                # import all the functions that will be used
import matplotlib.pyplot as plt   # in the subsequent steps.

#%% Raster
def rasterplot(spikes, triggers, start, stop):
    """ 
    Generates a raster plot based on given spike train and triggers within 
    a specified time window. 

    Args:
        spikes (numpy array_)   : 1D numpy array contains spike times.
        triggers (numpy array_) : 1D numpy array contains trigger times.
        start (float)           : Defined start time in seconds
        stop (float)            : Defined stop time in seconds
        
    Returns:
        Peristimulus time spike raster plot.
    """
    plt.figure(figsize=(12, 20))        # create a figure with a 12:20 aspect ratio
    
    for i in range(len(triggers)):      # loop over all triggers
        rel_spike = spikes-triggers[i]  # relative spikes
        sel_rel_spike = rel_spike[(rel_spike>start) & (rel_spike<stop)]
                                        # selected relative spikes
        y = np.ones(np.shape(sel_rel_spike))*i
                                        # same length array used for plotting
        plt.plot(sel_rel_spike,y,'.k')  # plot spike time as blackdots
    plt.xlabel('Time[seconds]',fontsize=15)     
                                        # label the x-axis
    plt.ylabel('Trigger number',fontsize=15)   
                                        # label the y-axis
    plt.title('Raster plot',fontsize=18)# name the title
    plt.show()
        
#%% PSTH
def PSTH(spikes,triggers,start,stop,bin,plot=False):
    """ 
    Generates a raster plot based on given spike train and triggers within 
    a specified time window. 

    Args:
        spikes (numpy array_)   : 1D numpy array contains spike times.
        triggers(numpy array_)  : 1D numpy array contains trigger times.
        start(float)            : Defined start time  in seconds.
        stop(float)             : Defined stop time  in seconds.
        bin(float)              : Specified bin sizes for histoplot[seconds].
    Returns:
        Peristimulus time histogram (PSTH)     
        """
    bin_edges = np.arange(start, stop, bin) # Bin edges array in selected time window
    bin_centers = bin_edges[:-1] + bin / 2  # bin centers array
    spike_sum = np.zeros(bin_centers.shape) # zeros array for sum spike numbers up
                                            # in the subsequent loop
                                            
    for i in range(len(triggers)):          # loop over all triggers
        rel_spike = spikes - triggers[i]    # relative spikes
        hist, _ = np.histogram(rel_spike, bins=bin_edges)
                                            # histocount relative spikes into
                                            # setted bin edges
        spike_sum += hist                   # Summing the spike numbers induced by
                                            # each trigger.

    spike_rate_ave = (spike_sum/len(triggers))/bin
                                            # normalize into spike rate (spikes per second).
    if plot:
        plt.plot(bin_centers, spike_rate_ave)
                                            # plot
        plt.xlim([start,stop])              # set x limits
        plt.xlabel('Time[seconds]',fontsize=12)     
                                            # label the x-axis
        plt.ylabel('average spike rate(spikes/sec)',fontsize=12)   
                                            # label the y-axis
        plt.title(f'PSTH,{bin*1000}ms window',fontsize=15)
                                            # name the title
        plt.show()
    return bin_centers,spike_rate_ave

    