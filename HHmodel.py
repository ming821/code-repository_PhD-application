def HHmodel2(Iext,dt,GNa,GK,ENa,EK,plotflag = False):

## Hodgkin-Huxley model 
# - Input: 
#  Iext : external input vector [pA] 
#  dt : time step [ms] 
#  GNa: [mS/cm^2] Na conductance density
#  GK : [mS/cm^2] K conductance density
#  ENa: [mV] Na reversal potential 
#  EK : [mV] K reversal potential 
# 
# - Output: 
#   v : simulated membrane potential [mV]
#   t : time vector [ms]
# 
# *** Notes *** 
#  - Iext is a vector not a scalar (as in RCeuler.m).
#  - The output vectors have the same lengths of the input vector Iext. 
#    Namely, length(v) = length(t) = length(Iext). 
#  - For stable calculation, the time step dt should be 0.005 [ms] or smaller.  
#  - Activation/inactivation functions are defined before the main part of the code
#

## importing required functions/modules
    from numpy import array, empty, exp 
    import matplotlib.pyplot as plt

## flag for plotting
#    plotflag = False  # True:plot, False:no plot 

# transition rate functions
    def alphaM(v): 
        vr = -65.0; return 0.1 * (v-vr-25) / ( 1 - exp(-(v-vr-25)/10) )
    def betaM(v):
        vr = -65.0; return 4.0 * exp(-(v-vr)/18)
    def alphaH(v):
        vr = -65.0; return 0.07 * exp(-(v-vr)/20)
    def betaH(v): 
        vr = -65.0; return 1.0 / ( 1 + exp( -(v-vr-30)/10 ) )
    def alphaN(v):
        vr = -65.0; return 0.01 * (v-vr-10) / ( 1 - exp(-(v-vr-10)/10) )
    def betaN(v):
        vr = -65.0; return 0.125 * exp(-(v-vr)/80)

# steady-state values of m,h,n,a,b
    def m_infty(v):
        return alphaM(v) / ( alphaM(v) + betaM(v) ) 
    def h_infty(v):
        return alphaH(v) / ( alphaH(v) + betaH(v) ) 
    def n_infty(v):
        return alphaN(v) / ( alphaN(v) + betaN(v) )

## parameters
    Smemb = 3000 # [um^2] surface area of the membrane  
    Cmemb = 1.0  # [uF/cm^2] membrane capacitance density 
    Cm = Cmemb * Smemb * 1e-8 # [uF] membrane capacitance 

    #GNa = 120.0 # [mS/cm^2] Na conductance density
    #GK = 36.0 # [mS/cm^2] K conductance density 
    GL = 0.3 # [mS/cm^2] leak conductance density 

    gNa = GNa * Smemb * 1e-8 # Na conductance [mS]
    gK = GK * Smemb * 1e-8 # K conductance [mS]
    gL = GL * Smemb * 1e-8 # leak conductance [mS]

    #ENa = +50.0 # [mV] Na reversal potential 
    #EK  = -80.0 # [mV] K reversal potential 
    EL  = -55.0 # [mV] leak reversal potential 

## time vector and data arrays
    Nsteps = len(Iext) # number of time steps
    t = array( range( Nsteps ) )*dt  # time vector for plotting
    v = empty( (Nsteps,) ) # [mV] membrane potential 
    m = empty( (Nsteps,) ) # Na activation variable
    h = empty( (Nsteps,) ) # Na inactivation variable
    n = empty( (Nsteps,) ) # K activation variable 

## initial values
    Vinit = -65 # [mV] initial potential
    v[0] = Vinit # initial membrane potential
    m[0] = m_infty(Vinit) # initial m
    h[0] = h_infty(Vinit) # initial h
    n[0] = n_infty(Vinit) # initial n

## calculate membrane response step-by-step 
    for j in range(Nsteps-1): 

        # ionic currents: g[mS] * V[mV] = I[uA]
        INa = gNa * m[j]**3 * h[j] * (ENa-v[j])
        IK = gK * n[j]**4 * (EK-v[j])
        IL = gL * (EL-v[j])

        # derivatives:  I[uA] / C[uF] * dt[ms] = dv[mV]
        dv_dt = ( INa + IK + IL + Iext[j]*1e-6 ) / Cm
        dm_dt = (1-m[j])* alphaM(v[j]) - m[j]*betaM(v[j])
        dh_dt = (1-h[j])* alphaH(v[j]) - h[j]*betaH(v[j])
        dn_dt = (1-n[j])* alphaN(v[j]) - n[j]*betaN(v[j])

        # calculate next step
        v[j+1] = v[j] + dv_dt * dt
        m[j+1] = m[j] + dm_dt * dt
        h[j+1] = h[j] + dh_dt * dt
        n[j+1] = n[j] + dn_dt * dt

## plotting
    if(plotflag):

        # open new figure window
        plt.figure(figsize=(6,8)) 

        # membrane potential
        plt.subplot(4,1,(1,2)); plt.cla
        plt.plot(t,v) 
        plt.xlim(0,max(t)+1)
        plt.ylim(-100,50)
        plt.ylabel('potential [mV]')

        # activation/inactivation variables
        plt.subplot(4,1,3); plt.cla
        plt.plot(t,m,'b-',label='m') # Na activation
        plt.plot(t,h,'r-',label='h') # Na inactivation 
        plt.plot(t,n,'g-',label='n') # K activation 
        plt.xlim(0,max(t)+1)
        plt.ylim(0,1)
        plt.ylabel('variable')
        plt.legend(loc='upper right')

        # external stimulus
        plt.subplot(4,1,4); plt.cla
        plt.plot(t,Iext,'k-')
        plt.xlim(0,max(t)+1)
        ymargin = max([max(abs(Iext)),10])
        plt.ylim( min(Iext)-0.2*ymargin, max(Iext)+0.2*ymargin )
        plt.ylabel('input [pA]')
        plt.xlabel('time [ms]')

        plt.show()

## returning results
    return (v,t)
