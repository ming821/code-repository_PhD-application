def networkFF_ming(Nin,Tlen,dt,Ls,deltaS,w,plotflag = False):

## Model of a feed-forward summing network in the auditory pathway
#   simulating the convergence of peripheral recepter neurons to a central neuron

# - Input: 
#  Nin    number of receptor input neurons converging to the central neuron(assume each side is equal )
#  Tlen   total simulated time length [ms]
#  dt     time step [ms] (use 0.01 as your default)
#  Ls     sound stimulus level [dB]
#  deltaS interaural intensity difference (IID)
#  w      inhibitory weight         
#   plotflag flag for plotting, default as False    
# 
# - Ouput:
#  Vout     membrane potential [mV] of the central output neuron, length(Vm) = length(dBin)
#  tspike   spike timings [ms] of the central output neuron
#  nspike   number of spikes in the central ouput neuron
# 
# *** Notes ***
#  This function internally calls the following functions: 
#   - alphafunc.m    : calculating alpha function 
#   - IFmodel        : integrate-and-fire model
#

## importing required functions/modules
    from numpy import array, zeros, ones, exp 
    from numpy.random import randn
    import matplotlib.pyplot as plt
    import numpy as np
    from alphafunc import alphafunc
    from IFmodel import IFmodel

## transducer functions (converting sound level into current)
    def transduction_simple(AdB):
        I = 15 * (AdB - 40) #  40 dB is the threshold
        I[ I<0   ] = 0      # min value is 0 pA
        I[ I>500 ] = 500    # max value is 500 pA (saturation)
        return I

    def transduction_sigmoid(AdB):
        Imax = 500 # max current value (saturation)
        Imin = 300 # min currentvalue (spontaneous)
        A0 = 60    # assuming 60 dB as average input level 
        Ak = 5.0   # scaling factor
        I = Imin + (Imax-Imin) / ( 1 + exp( -(AdB-A0)/Ak ) )
        return I

## time and input vectors
    Ntime = round(Tlen/dt)                       # number of time steps
    dBin_l = (Ls + deltaS/2) * ones( (Ntime,) )  # sound stimulus vector [dB]
    dBin_r = (Ls - deltaS/2) * ones( (Ntime,) )  # sound stimulus vector [dB]

    t = array(range(Ntime) )*dt                  # time vector [ms] for plotting

## parameters for the IF model 
    Cm      = 20  # capacitance [pF] 
    Rm      = 50  # resistance [Mohm] 
    Er      = -60 # resting potential [mV] 
    V0      = Er  # initial potential [mV] 
    Vth     = -43 # threshold [mV] 
    Vreset  = -75 # reset potential [mV] 
    Tref    = 1.0 # refractory period [ms] 

## parameters for alpha synapse
    talpha = 1.5     # [ms] time constant
    aalpha = 800/Nin # [pA] amplitude, scaled by the number of inputs

                     # inhibitory synapses
    talpha_ihn = 3   # [ms] time constant
    aalpha_ihn = w/Nin 
                     # [pA] amplitude, scaled by the number of inputs

## implementation of Lateral Inhibition(neuron1)
                                                             # the first neuron receives inhibitory inputs from the right and 
                                                             # excitatory inputs from the left side.
    
    Iin_l = transduction_sigmoid(dBin_l)                     # transduce the stimulus from dB into a current,.For the simple feed-forward simulation task, 
    Iin_r = transduction_sigmoid(dBin_r)                     # use transduction_simple. Use transduction_sigmoid for the lateral inhibition task
    
    Isyn_l = zeros( (Ntime,) )                               # make a vector for the summed synaptic outputs of the left side receptor neurons
    Isyn_r = zeros( (Ntime,) )                               # make a vector for the summed synaptic outputs of the right side receptor neurons
   
    for i in range(Nin):                                     # loop through the left receptor neurons
        noise = randn(len(Iin_l)) * 500                      # noise with mean of0 and std of 500 pA
        Inoise = Iin_l + noise                               # actual sound input on left side is sL = s + Δs/2.        
        _,Tsp,_,=IFmodel(Cm,Rm,Er,Inoise,V0,dt,Vth,Vreset,Tref)
                                                             # output spike timings of the IF mode, default is no plotting
        Nsp = zeros( (Ntime,) )                              # make a zero vector of length Ntime
        index = np.array([x / dt for x in Tsp])              # indexing all spike times
        Nsp[index.astype(int)- 1] = 1
        (Iout,ydum) = alphafunc(Nsp,dt,talpha,aalpha)        # calling the alpha function 
        Isyn_l = Isyn_l + Iout                               # sum the output of the i-th receptor neuron
    
    
    for i in range(Nin):                                     # loop through the right receptor neurons
        noise = randn(len(Iin_r)) * 500                      # noise with mean of0 and std of 500 pA
        Inoise = Iin_r + noise                               # actual sound input on right side is sL = s - Δs/2.                
        _,Tsp,_,=IFmodel(Cm,Rm,Er,Inoise,V0,dt,Vth,Vreset,Tref)
                                                             # output spike timings of the IF mode, default is no plotting
        Nsp = zeros( (Ntime,) )                              # make a zero vector of length Ntime
        index = np.array([x / dt for x in Tsp])              # indexing all spike times
        Nsp[index.astype(int)- 1] = 1
        (Iout,ydum) = alphafunc(Nsp,dt,talpha_ihn,aalpha_ihn)# calling the alpha function 
        Isyn_r = Isyn_r + Iout                               # sum the output of the i-th receptor neuron

    Isyn_1 = Isyn_l + Isyn_r                                 # the total synaptic current by summing each side up
    (Vout1,tspike1,nspike1) = IFmodel(Cm,Rm,Er,Isyn_1,V0,dt,Vth,Vreset,Tref)
                                                             # output spike timings of the IF mode, default is no plotting
    
    ## plotting
    if(plotflag):
        plt.figure(1)

        # model synaptic input 
        plt.subplot(2,1,1); plt.cla
        plt.plot(t,Isyn_1,'b-') 
        plt.xlim(0,Ntime*dt)
        plt.xlabel('time [ms]'); plt.ylabel('summed synaptic input [pA]')
        plt.title('simulated synaptic input')

        # response of the central neuron
        plt.subplot(2,1,2); plt.cla
        plt.plot(t,Vout1,'b-') # membrane potential
        if( len(tspike1)> 0 ): # if any spikes exist, show spike times
            plt.plot(tspike1,[Vth]*nspike1,'ro',fillstyle='none') 
        plt.xlim(0,Ntime*dt); plt.ylim(Vreset-5,Vth+5)
        plt.xlabel('time [ms]'); plt.ylabel('potential [mV]')
        plt.title( 'central neuron output: %d spikes' % nspike1 )

        plt.subplots_adjust(hspace=0.5) # adjustment of margins
        plt.show()
   

## for the symmetric neuron
                                                             # the second(symmetric) neuron receives inhibitory inputs from the left and 
                                                             # excitatory inputs from the right side.
    
    Iin_l = transduction_sigmoid(dBin_l)                     # transduce the stimulus from dB into a current,.For the simple feed-forward simulation task, 
    Iin_r = transduction_sigmoid(dBin_r)                     # use transduction_simple. Use transduction_sigmoid for the lateral inhibition task
    
    Isyn_l = zeros( (Ntime,) )                               # make a vector for the summed synaptic outputs of the left side receptor neurons
    Isyn_r = zeros( (Ntime,) )                               # make a vector for the summed synaptic outputs of the right side receptor neurons
   
    for i in range(Nin):                                     # loop through the left receptor neurons
        noise = randn(len(Iin_l)) * 500                      # noise with mean of0 and std of 500 pA
        Inoise = Iin_l + noise                               # actual sound input on left side is sL = s + Δs/2.        
        _,Tsp,_,=IFmodel(Cm,Rm,Er,Inoise,V0,dt,Vth,Vreset,Tref)
                                                             # output spike timings of the IF mode, default is no plotting
        Nsp = zeros( (Ntime,) )                              # make a zero vector of length Ntime
        index = np.array([x / dt for x in Tsp])              # indexing all spike times
        Nsp[index.astype(int)- 1] = 1
        (Iout,ydum) = alphafunc(Nsp,dt,talpha_ihn,aalpha_ihn)# calling the alpha function 
        Isyn_l = Isyn_l + Iout                               # sum the output of the i-th receptor neuron
    
    
    for i in range(Nin):                                     # loop through the right receptor neurons
        noise = randn(len(Iin_r)) * 500                      # noise with mean of0 and std of 500 pA
        Inoise = Iin_r + noise                               # actual sound input on right side is sL = s - Δs/2.                
        _,Tsp,_,=IFmodel(Cm,Rm,Er,Inoise,V0,dt,Vth,Vreset,Tref)
                                                             # output spike timings of the IF mode, default is no plotting
        Nsp = zeros( (Ntime,) )                              # make a zero vector of length Ntime
        index = np.array([x / dt for x in Tsp])              # indexing all spike times
        Nsp[index.astype(int)- 1] = 1
        (Iout,ydum) = alphafunc(Nsp,dt,talpha,aalpha)        # calling the alpha function 
        Isyn_r = Isyn_r + Iout                               # sum the output of the i-th receptor neuron

    Isyn_2 = Isyn_l + Isyn_r                                 # the total synaptic current by summing each side up
    (Vout2,tspike2,nspike2) = IFmodel(Cm,Rm,Er,Isyn_2,V0,dt,Vth,Vreset,Tref)
                                                             # output spike timings of the IF mode, default is no plotting
   

## plotting
    if(plotflag):
        plt.figure(2)

        # model synaptic input 
        plt.subplot(2,1,1); plt.cla
        plt.plot(t,Isyn_2,'b-') 
        plt.xlim(0,Ntime*dt)
        plt.xlabel('time [ms]'); plt.ylabel('summed synaptic input [pA]')
        plt.title('simulated synaptic input')

        # response of the central neuron
        plt.subplot(2,1,2); plt.cla
        plt.plot(t,Vout2,'b-') # membrane potential
        if( len(tspike2)> 0 ): # if any spikes exist, show spike times
            plt.plot(tspike2,[Vth]*nspike2,'ro',fillstyle='none') 
        plt.xlim(0,Ntime*dt); plt.ylim(Vreset-5,Vth+5)
        plt.xlabel('time [ms]'); plt.ylabel('potential [mV]')
        plt.title( 'central neuron output: %d spikes' % nspike2 )

        plt.subplots_adjust(hspace=0.5) # adjustment of margins
        plt.show()

## end of function
    return (Vout1, tspike1, nspike1,Vout2, tspike2, nspike2)
