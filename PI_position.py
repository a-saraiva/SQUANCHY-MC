#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:50:33 2019

@author: SQUANCHY-MC
"""


import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy
from fast_histogram import histogram1d


# %%

"""Parameters, units: ueV, nm, ns"""

N= 5000 #time_steps and number of tried changes during one sweep.
beta = 0.1
delta= beta/N #length of time_step
h=5 # tried position change (step size) during Metropolis is from [-h/2,h/2]
path=np.random.uniform(-20,20,N) #Initial path: "Hot start"
#path=np.zeros(N) #Initial path: "Cold start"

num_path=3000 #number of sweeps
path_list = [[0]] * num_path #path at the end of each sweep

m= 5.686 * 10 **(-6) # ueV ns^2/nm^2 - electron mass
hbar = 0.6582 # ueV * ns
a = 0.543 #nm - lattice constant in silicon

pot_step= 3 * 10**(6) #ueV - Si/SiO2 interface
E_z = 0.5 * 10**(3) #ueV/nm - Electric field in z direction

# %%

"""Different potentials, all one dimensional"""

# Potential box with length 2a and height V
def pot_box(x,a,V):
    if -a<x<a:
        return(0)
    else: return(V)

# Heterostructure interface potential (z-direction) leading to Airy functions as wave function.
def airy(x,slope,V):
    if x<0:
        return(V)
    else: return(slope*x)
    
# Double dot around +x_0 and -x_0, frequency w.
def double_HO(x,x_0,w):
    return(min(0.5*m*w**2*(x-x_0)**2,0.5*m*w**2*(x+x_0)**2))
    

# %%

"""
One sweep consists of trying N times changing a random element in the array path (size of this change depends on h),
each time accepting/refusing according to Metropolis. 
"""

def sweep(path,h,N):
    
    s=0 # Action change during the sweep
    accrate=0 # accuracy rate
    index=np.random.uniform(1,N-1,N) # Order in which path changes are tried. Endpoints (0 and N-1) kept fixed.
    rand=np.random.uniform(0,1,2*N) # Creating 2N random numbers. First N used for proposing change, other N for Metroplis acceptance.
    
    
    for i in range(0,N):
        tau=int(index[i])
        tau_m=int((tau-1))
        tau_p=int((tau+1))
        x_new=path[tau]+h*(rand[i]-0.5) #propose change of path[tau] in symmetric way.
        
        """
        Calculating change in action due to replacing path[tau] by x_new. Different potentials considered.
        """
        
        #s_old=0.5*m*((path[tau_p]-path[tau])**2+(path[tau]-path[tau_m])**2)+0.5*m*w**2*path[tau]**2
        s_old=((0.5*m)/(delta))*((path[tau_p]-path[tau])**2+(path[tau]-path[tau_m])**2)+delta*airy(path[tau],E_z,pot_step)
        #s_old=0.5*m*((path[tau_p]-path[tau])**2+(path[tau]-path[tau_m])**2)+pot_box(path[tau],3,5)
        #s_old=0.5*m*((path[tau_p]-path[tau])**2+(path[tau]-path[tau_m])**2)+double_pot_box(path[tau],3.5)
        #s_old=0.5*m*((path[tau_p]-path[tau])**2+(path[tau]-path[tau_m])**2)+E(path[tau])
        
        #s_new=0.5*m*((path[tau_p]-x_new)**2+(x_new-path[tau_m])**2)+0.5*m*w**2*x_new**2
        s_new=((0.5*m)/(delta))*((path[tau_p]-x_new)**2+(x_new-path[tau_m])**2)+delta*airy(x_new,E_z,pot_step)
        #s_new=0.5*m*((path[tau_p]-x_new)**2+(x_new-path[tau_m])**2)+pot_box(x_new,3,5)
        #s_new=0.5*m*((path[tau_p]-x_new)**2+(x_new-path[tau_m])**2)+double_pot_box(x_new,3.5)
        #s_new=0.5*m*((path[tau_p]-path[tau])**2+(path[tau]-path[tau_m])**2)+E(x_new)
        
        delta_s=s_new-s_old
        
        
        if delta_s<0: #always accepting if action lowered
            path[tau]=x_new
            accrate+=(1/N)
            s+=delta_s
        elif rand[i+N]<math.exp(-(delta_s/hbar)): #otherwise accepte with specific probability.
            path[tau]=x_new
            accrate+=(1/N)
            s+=delta_s
    return(path,accrate,s)
        
    
    
# %%

"""
Main loops, creating a list of paths in the previously defined path_list,
which after thermalization follow the distribution used for Metropolis
"""

"""
Use perform_therm if interested in convergence behaviour; To check how many sweeps needed
to convergence (thermalization).
- path is initial path, can be either hot/cold.
- N is number of timeslices (fixed)
"""

def perform_therm(path,N):
   
    for j in range(0,num_path):
        sweep(path,h,N)
        path_list[j]=deepcopy(path)
        
"""
Use perform to extract converged results. 

n_conv: number of sweeps which have to be ignored for thermalization. (1000)

To avoid correlations between two following paths (separated by one sweep),
one can consider every n_sav'th path, and ignore all between.

Standart: n_sav=1
"""

def perform(path,N,n_conv,n_sav=1):
    for i in range(0,n_conv):
        sweep(path,h,N)
    for j in range(0,num_path):
        path_list[j]=deepcopy(path)
        for k in range(0,n_sav):
            sweep(path,h,N)


# %%
            
"""
Kind of fast function to create histogram.

path: Accept arrays, lists, list of lists,..

Sandart bounderies of histogram given by max/min of path value, but can be changed
if one wants to zoom in.
""" 

def histo(path,x_min=np.amin(path),x_max=np.amax(path)):
    number_bins=1000
    
    histo = histogram1d(path,range=[x_min,x_max],bins=number_bins)
    h=np.linspace(x_min,x_max,number_bins)
    plt.plot(h,histo,".")
    plt.xlabel("position [nm]")
    plt.ylabel("|Psi(x)|^2")
    
    
# %%
    
"""
Once a sequence of paths created (using perform), mean and stand_dev functions
plot the mean/standart deviation of each path in the sequence.
"""
def mean():
    x_exp=np.zeros(num_path)
    x=np.linspace(0,num_path-1,num_path)
    x_mean=0
    for i in range(0,num_path):
        for j in range(0,N):
            x_exp[i]+=path_list[i][j]
        x_exp[i]=x_exp[i]/N
    for i in range(0,num_path):
        x_mean+=x_exp[i]
    x_mean=x_mean/num_path
    const=np.ones(num_path)*x_mean
    plt.figure()
    plt.plot(x,x_exp,".")
    plt.plot(x,const)
    plt.xlabel("sweep")
    plt.ylabel("mean")
    plt.show    
    

def stand_dev():
    x_exp=np.zeros(num_path)
    x=np.linspace(0,num_path-1,num_path)
    x_square=0
    #x_mean=0
    for i in range(0,num_path):
        x_mean=0
        for j in range(0,N):
            x_mean+=path_list[i][j]
        x_mean=x_mean/N
        
        for j in range(0,N):
            x_exp[i]+=(path_list[i][j]-x_mean)**2
        x_exp[i]=np.sqrt(x_exp[i]/(N-1))
    for i in range(0,len(x_exp)):
        x_square+=x_exp[i]
    x_square=x_square/num_path
    square=np.ones(num_path)*x_square
    plt.figure()
    plt.plot(x,x_exp,".")
    plt.plot(x,square)
    plt.xlabel("sweep")
    plt.ylabel("standart dev")
    plt.show    
