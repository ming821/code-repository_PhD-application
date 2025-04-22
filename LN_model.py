# -*- coding: utf-8 -*-
"""
Script for Computational Neuroscience-Encoding and Decoding course, day3 & day4

Day3:
    The Linear model: use different filters(sta, kernels) to producing generator
     potential(repeat data) or linear response.(The sta is trained by continuous 
     data from Day_03_2.mat)

Day4:
    1.The Linear-Nolinear model: Reuse same method in day3, calculate the genetrator
     potential of continuous data(Linear). 
     
     2.Then histogram whole g-potemtial and slected g-potential(time bins with 
      spikes) separatekly. Calculating the spike probability by dividing the later 
      into first one(static nonlinearity).
     
     3.Fit it with cumulative normal distribution(nonlinearity)
     
     4. Use the fit parameters to calculate the spike probability for the repeated 
     stimulus. Plot PSTH(in spike probablity) with recordded data in same figure.
    
Caution:
*******
Due to shared variables and dataset,the code is structured to run in order,
proceeding from the top to the bottom.
*******

Date: 8/12/2022
By zhiming.jia
"""

#%% set environment    
import numpy as np                       # import all the functions that will be used
import scipy                             # in the subsequent steps.
import matplotlib.pyplot as plt
from Spiketrain_analysis import rasterplot
from Spiketrain_analysis import PSTH

#%%
# Day3
#%% Temporal linear response to a flash
stim=np.zeros(300)                       # generate a stimulus protocol
stim[101:201]=1                          # set stimulus from 101 t0 200 as 1
sta = np.load('sta_ming.npy')            # load sta of dataset in 2.1.1
gen_poten = np.zeros(len(stim))          # zeros array for collecting generator potential

for i in range(len(sta),len(stim)):      # loop over stimulus, from 29 to 299 indexes
                                         # which is the bin steps 30  to 300 
    gen_poten[i-1] = sum(sta*stim[i-len(sta):i])
                                         # multiply each indices of sta and stimulus 
                                         # separately. Then sum them up
                                         
plt.figure(figsize=(8,5))                # creat figure with 8*5 aspect ratio                                         
plt.plot(range(len(stim)),stim,'b',label='stimulus')   
                                         # plot stimulus in bllue line
plt.plot(range(len(stim)),gen_poten,'orange',label='G_potential')
                                         # plot generator potential in blue line
plt.xlabel('Time',fontsize=12)           # label the x-axis                                    
plt.ylabel('Spike rate/stimulus amplitude',fontsize=12)     
                                         # label the y-axis
plt.xlim(0,len(stim))                    # set limits of x-axis
plt.legend()                             # add  legend
plt.show()    
#%% Low- and Highpass filters & stimulus1
stim=np.zeros(300)                       # stimulus 1
stim[101:201]=1

kernel2=np.ones(20)*0.05                 # Lowpass filter
kernel1=np.concatenate([np.ones(2)*0.5, np.ones(2)*-0.5])
                                         # Highpass filter
lin_res1 = np.zeros(len(stim))           # zeros array used containing generator
lin_res2 = np.zeros(len(stim))           # potential(linear response)

for i in range(len(kernel1),len(stim)):  # loop over stimulus
    lin_res1[i] = sum(kernel1*stim[i-len(kernel1):i])
                                         # convolution
    
for i in range(len(kernel2),len(stim)):  # loop over stimulus
    lin_res2[i] = sum(kernel2*stim[i-len(kernel2):i])
                                         # convolution
                                         
plt.figure(figsize=(8,5))                # creat figure with 8*5 aspect ratio                                          
plt.plot(stim,'--b',label='stimulus')    # plot linear responses and stimulus 
                                         # in same figure with different color
plt.plot(lin_res1,'r',label='Highpass filter')
plt.plot(lin_res2,'orange',label='Lowpass filter')

plt.xlabel('Time',fontsize=12)           # label the x-axis                                    
plt.ylabel('Spike rate/stimulus amplitude',fontsize=12)     
                                         # label the y-axis
plt.xlim(0,len(stim))                    # set limits of x-axis
plt.legend()                             # add legend
plt.show()    

#%% Low- and Highpass filters & stimulus2
tx=np.arange(0, 500)                     # stimulus 2
tx=np.arange(0, 500)
stim=np.sin(tx)+np.sin(tx/100) 

kernel2=np.ones(20)*0.05                 # Lowpass filter
kernel1=np.concatenate([np.ones(2)*0.5, np.ones(2)*-0.5])
                                         # Highpass filter
lin_res1 = np.zeros(len(stim))           # zeros array used containing generator
lin_res2 = np.zeros(len(stim))           # potential(linear response)

for i in range(len(kernel1),len(stim)):  # loop over stimulus
    lin_res1[i] = sum(kernel1*stim[i-len(kernel1):i])
                                         # convolution
    
for i in range(len(kernel2),len(stim)):  # loop over stimulus
    lin_res2[i] = sum(kernel2*stim[i-len(kernel2):i])
                                         # convolution
                                         
plt.figure(figsize=(10,7))               # creat figure with 10*7 aspect ratio                                         
plt.plot(stim,'--b',label='stimulus')    # plot linear responses and stimulus 
                                         # in same figure with different color
plt.plot(lin_res1,'r',label='Highpass filter')
plt.plot(lin_res2,'orange',label='Lowpass filter')

plt.xlabel('Time',fontsize=12)           # label the x-axis                                    
plt.ylabel('Spike rate/stimulus amplitude',fontsize=12)     
                                         # label the y-axis
plt.xlim(0,len(stim))                    # set limits of x-axis
plt.legend()                             # add legend
plt.show()    

#%% Spatio-temporal linear response & Raster
data = scipy.io.loadmat('Day_03_2')      # load data
                                         # extract repeat spikes and triggers
spikes_repeats = data['spikes_repeats'].squeeze() 
trigger_repeats = data['trigger_repeats'].squeeze()    
triggers = trigger_repeats   # extract triggers 
start = 1                                # set start time to1s
stop = 5                                 # set stop time to 5s

rasterplot(spikes_repeats,trigger_repeats,start,stop)  
                                         # call funtion and plot raster plot
#%% Spatio-temporal linear response & Raster0.02s
bin = data['refresh_time'].squeeze()     # set bin size to refresh_time
bin_centers,spike_rate_ave = PSTH(spikes_repeats,trigger_repeats,start,stop,bin,plot='true')   
                                         # call funtion and plot PSTH
#%% Spatio-temporal linear response & STA of contimuous data(training data)
                                         # extract continuous spikes,triggers and stimulus
spikes_continuous = data['spikes_continuous'].squeeze()
trigger_continuous = data['trigger_continuous'].squeeze()    
stim_continuous = data['stim_continuous'].squeeze()        

spike_vector=np.histogram(spikes_continuous,bins=trigger_continuous)[0]
                                         # divide spikes into triggers, aim to 
                                         # making the same length with stimulus
spike_vector[0:30] = 0                   # discard the first 30 frames
sts_con = np.zeros((5,5,30))             # zeros 3D array for suming sta in loop
                                         # pixel 5*5
                                          
spike_index = np.argwhere(spike_vector>0)# find the indexes where has spike
spike_index = spike_index.squeeze()      # reduce to 1D array

for i in spike_index:                    # loop over all spike indexes
    sts_con += stim_continuous[:,:,i-30:i]
                                         # auming 30 frames stimulus before spikes
     
sta_con = sts_con/len(spike_index)       # calculate sta
#%% Spatio-temporal linear response & linear response                                         
stim = data['stim_repeats'].squeeze()    # extract repeat stimulus 
gen_poten = np.zeros(stim.shape[2])      # zeros array for collecting generator potential
                                         # same length as stimulus array

                                         # loop over stimulus
for i in range(sta_con.shape[2],stim.shape[2]):
    gen_poten[i] = sum(sta_con*stim[:,:,i-sta_con.shape[2]:i])
                                         # convolution using sta_con as the filter.

timea = np.arange(len(gen_poten))*bin    # time array same length as linear response 
plt.figure(figsize=(10,6))               # creat figure with 10*6 aspect ratio                                          
plt.plot(timea,gen_poten/np.max(gen_poten),'orange',label='G_potential')
plt.plot(bin_centers,spike_rate_ave/np.max(spike_rate_ave),'b',label='test data',alpha=0.6) 
                                         # plot the linear response and the PSTH in 
                                         # same figure,scale both to a maximum value 
                                         # of one
plt.xlabel('Time[s]',fontsize=15)        # label the x-axis                                    
plt.ylabel('Spike rate',fontsize=15)     # label the y-axis
plt.xlim(start,stop)                     # set limits of x-axis
plt.legend()                             # add legend

#%%
# Day4
#%% 4.1.1- generator potential by continuous stimulus
stim = data['stim_continuous'].squeeze() # extract repeat stimulus 
gpoten_con = np.zeros(stim.shape[2])     # zeros array for collecting generator potential
                                         # same length as stimulus array

                                         # loop over stimulus
for i in range(sta_con.shape[2],stim.shape[2]):
    gpoten_con[i] = sum(sta_con*stim[:,:,i-sta_con.shape[2]:i])
                                         # convolution using sta_con as the filter.
                                                                                                                         
hist_bin_edges=np.arange(-3.5,3.5+.25,.25)
hist_bin_centers=hist_bin_edges[:-1] + 0.25/2 
                                         # bin edges and centers for hisogram and
                                         # plotting 
spike_vector=np.histogram(spikes_continuous,bins=trigger_continuous)[0]
gpoten_con_pos = gpoten_con[spike_vector>0]
                                         # seleted generator potential where 
                                         # time bins with spikes
hist1, _ = np.histogram(gpoten_con, bins=hist_bin_edges)
hist2, _ = np.histogram(gpoten_con_pos, bins=hist_bin_edges)
                                         # histogram into bin edges separately
plt.figure(figsize=(8,5))                # creat figure with 8*5 aspect ratio 
plt.plot(hist_bin_centers,hist1,'blue',label='All')
plt.plot(hist_bin_centers,hist2,'orange',label='Positive part')
                                         # plot in same figure with lines                                   
plt.xlabel('Generator potential',fontsize=12)
                                         # label the x-axis                                    
plt.ylabel('Spike count',fontsize=12)     
                                         # label the y-axis
plt.xlim(hist_bin_centers[0],hist_bin_centers[-1])                   
                                         # set limits of x-axis
plt.legend()                             # add legend
plt.show()    

#%% 4.1.1- spike probability
spike_probablity = hist2/hist1           # calculate spike probablity of each bin

#%% 4.1.2-cumulative normal distribution function
def nl_cdf(x, mu, sigma, amp):
    """
    Scaled cumulative normal density function.
    Args:
    x: vector of x values
    mu: mean of normal distribution
    sigma: standard deviation of normal distribution
    amp: amplitude
    Returns:
    Vector of scaled cumulative normal density function values for x.
    """
    return amp * scipy.stats.norm.cdf(x, mu, sigma)

nl = spike_probablity                    # fit the spike probablity with function
initial_guess_params=[1,1,1]             # set starting values for the parameters
nl_params,_=scipy.optimize.curve_fit(nl_cdf, hist_bin_centers,nl, p0=initial_guess_params)
                                         # fit for spike probablity
                                         
plt.figure(figsize=(8,5))                # creat figure with 8*5 aspect ratio                                        
plt.plot(hist_bin_centers,nl,'.b',label='spike_probablity')      
plt.plot(hist_bin_centers,nl_cdf(hist_bin_centers,*nl_params),'r',label='fitting curve')
plt.xlabel('Generator potential',fontsize=12)
                                         # label the x-axis                                    
plt.ylabel('Spike probablity',fontsize=12)     
                                         # label the y-axis
plt.xlim(hist_bin_centers[0],hist_bin_centers[-1])                   
                                         # set limits of x-axis
plt.legend()                             # add legend
plt.show()    
# high varaition in curve terminal is because the high frequency with
# lower probablity
#%% 4.1.3 - plot spike probability, reuse gen_poten(repeat data)
# values>1, cause bin size is too big, allow several spikes in same bin
timea = np.arange(len(gen_poten))*bin    # time array same length as linear response 
plt.figure(figsize=(20,7))               # creat figure with 20*7 aspect ratio                                               
plt.plot(timea,nl_cdf(gen_poten,*nl_params),'orange',label='model')
                                         # apply same fitted parameters for generator
                                         # potnetial of repeat data with function,
                                         # calculate the model's spike probability                                          
plt.plot(bin_centers,spike_rate_ave*bin,'b',label='recorded data',alpha=0.6) 
                                         # spike rate average multiply bin, convert 
                                         # it to spike probability                                        
plt.xlabel('Time[seconds]',fontsize=15)
                                         # label the x-axis                                    
plt.ylabel('Spike probablity',fontsize=15)     
                                         # label the y-axis
plt.xlim(start,stop)                     # set limits of x-axis(time window)                                        
plt.legend()                             # add legend
plt.show()    