#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:24:08 2019

@author: schlattnerfrederic
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:38:42 2019

@author: schlattnerfrederic
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy
from fast_histogram import histogram1d


# %%

"""Parameters, units: ueV, nm, ns"""

N= 10000 #time_steps

#delta= 10**(-5)
beta=1
delta= beta/N
#beta = N*delta # ueV-1


h=2
h_mom=1
path=np.random.uniform(0,5,N+1)
path_mom=np.random.uniform(-1,1,N)

m= 5.686 * 10 **(-6) # ueV ns^2/nm^2
hbar = 0.6582 # ueV * ns
a = 0.543 #nm 

epsilon= 1.414 * 10**(6) #ueV
u = 0.6828 * 10**(6) #ueV
v = 0.6119 * 10**(6) #ueV

pot_step= 3 * 10**(6) #ueV
E_field = 2 * 10**(3) #ueV/nm

num_path=10000 #number of paths considered
path_list = [[0]] * num_path #list with saved position paths
path_list_mom = [[0]] * num_path #list with saved momentum paths

# %%

"""
Defining potential and dispersion relation
"""

def pot_box(x,a,V):
    if -a<x<a:
        return(0)
    else: return(V)
    
def airy(x,slope,V):
    return((V/(np.exp(30*x)+1))+slope*x)
    #if x<0:
     #   return(V)
    #else: return(slope*x)
    
def E(k):
    return((hbar*k)**2/(2*m))
    
# %%
    
def sweep_pos_mom(path,path_mom,h,h_mom):
    
    accrate_pos=0
    accrate_mom=0
        
    index=np.random.uniform(1,N,N)
    index_mom=np.random.uniform(0,N,N)
    
    rand=np.random.uniform(0,1,2*N)
    rand_mom=np.random.uniform(0,1,2*N)
    
    for i in range(0,N):
        
        #Position
        
        tau=int(index[i])
        x_new=path[tau]+h*(rand[i]-0.5)
        
        
        delta_pot=airy(x_new,E_field,pot_step)-airy(path[tau],E_field,pot_step)
        so1=path_mom[tau]*((path[tau+1]-path[tau])/delta)
        so2=path_mom[tau-1]*((path[tau]-path[tau-1])/delta)
        sn1=path_mom[tau]*((path[tau+1]-x_new)/delta)
        sn2=path_mom[tau-1]*((x_new-path[tau-1])/delta)
        
        #if np.sign(math.cos(delta*sn1)*math.cos(delta*sn2)/(math.cos(delta*so1)*math.cos(delta*so2)))==-1:
        #    continue
        
        if -(delta/hbar)*(delta_pot)>10:
            path[tau]=x_new
            accrate_pos+=(1/N)
            continue
        
        if rand[i+N]<math.exp(-(delta/hbar)*(delta_pot))*math.cos(delta*sn1)*math.cos(delta*sn2)/(math.cos(delta*so1)*math.cos(delta*so2)):
        
        #if math.log(rand[i+N])<(-(delta/hbar)*(delta_pot))*math.log(math.cos(delta*sn1)*math.cos(delta*sn2)/(math.cos(delta*so1)*math.cos(delta*so2))):
                
                path[tau]=x_new
                accrate_pos+=(1/N)
        
                
        #Momentum
        
        zau=int(index_mom[i])
        k_new=path_mom[zau]+h_mom*(rand_mom[i]-0.5)
        
        delta_E=E(k_new)-E(path_mom[zau])
        Eo=path_mom[zau]*((path[zau+1]-path[zau])/delta)
        En=k_new*((path[zau+1]-path[zau])/delta)
        
        
        #if np.sign(math.cos(delta*En)/math.cos(delta*Eo))==-1:
        #    continue
        
        
        if rand_mom[i+N]<math.exp(-(delta/hbar)*(delta_E))*math.cos(delta*En)/math.cos(delta*Eo):
        #if math.log(rand_mom[i+N])<(-(delta/hbar)*(delta_E))+math.log(math.cos(delta*En)/math.cos(delta*Eo)):
                
                path_mom[zau]=k_new
                accrate_mom+=(1/N)
        
                
    return(path,path_mom,accrate_pos,accrate_mom)


# %%
def perform_PS():
    #h=h_1
    #h_mom=h_mom_1
    for i in range(0,2000):
        sweep_pos_mom(path,path_mom,h,h_mom)
        #h=h*sweep_pos_mom(path,path_mom,h,h_mom)[2]/(idrate)
        #h_mom=h_mom*sweep_pos_mom(path,path_mom,h,h_mom)[2]/(idrate)
    for j in range(0,num_path):
        path_list[j]=deepcopy(path)
        path_list_mom[j]=deepcopy(path_mom)
        sweep_pos_mom(path,path_mom,h,h_mom)
        #for k in range(0,3):
            #sweep_pos_mom(path,path_mom,h,h_mom)
            #h=h*sweep_pos_mom(path,path_mom,h,h_mom)[2]/(idrate)
            #h_mom=h_mom*sweep_pos_mom(path,path_mom,h,h_mom)[2]/(idrate)
        

def perform_PS_therm():
    
    for j in range(0,num_path):
        sweep_pos_mom(path,path_mom,h,h_mom)
        
        path_list[j]=deepcopy(path)
        path_list_mom[j]=deepcopy(path_mom)


    
# %%
def fig_mean():
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
    print(x_mean)
# %%
def fig_mean_mom():
    x_exp=np.zeros(num_path)
    x=np.linspace(0,num_path-1,num_path)
    x_mean=0
    for i in range(0,num_path):
        for j in range(0,N):
            x_exp[i]+=path_list_mom[i][j]
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
    print(x_mean)
    

# %%
def fig_stand_dev():
    x_exp=np.zeros(num_path)
    x=np.linspace(0,num_path-1,num_path)
    x_square=0
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
    print(x_square)
    plt.figure()
    plt.plot(x,x_exp,".")
    plt.plot(x,square)
    plt.xlabel("sweep")
    plt.ylabel("standart dev")
    plt.show    
    
# %%
def fig_stand_dev_mom():
    x_exp=np.zeros(num_path)
    x=np.linspace(0,num_path-1,num_path)
    x_square=0
    for i in range(0,num_path):
        k_mean=0
        for j in range(0,N):
            k_mean+=path_list_mom[i][j]
        k_mean=k_mean/N
        for j in range(0,N):
            x_exp[i]+=(path_list_mom[i][j]-k_mean)**2
        x_exp[i]=np.sqrt(x_exp[i]/(N-1))
    for i in range(0,len(x_exp)):
        x_square+=x_exp[i]
    x_square=x_square/num_path
    square=np.ones(num_path)*x_square
    print(x_square)
    plt.figure()
    plt.plot(x,x_exp,".")
    plt.plot(x,square)
    plt.xlabel("sweep")
    plt.ylabel("standart dev")
    plt.show 

# %%

def histo(path):
    print("N=", N)
    print("beta=", beta)
    print("num_path= ", num_path)
    
    #x_min=np.amin(path)
    #x_max=np.amax(path)
    
    x_min=0
    x_max=8
    
    number_bins=100
    
    histo = histogram1d(path,range=[x_min,x_max],bins=number_bins)
    h=np.linspace(x_min,x_max,number_bins)
    plt.plot(h,histo,".")
    plt.xlabel("position [nm]")
    plt.ylabel("|Psi(x)|^2")    
    

# %%
"""    
def histo(path):
    
    print("N=", N)
    print("beta", beta)
    
    number_bins=200
    x_min=np.amin(path)
    x_max=np.amax(path)
    #x_min=-2
    #x_max=1
    histo = histogram1d(path,range=[x_min,x_max],bins=number_bins)
    h=np.linspace(x_min,x_max,number_bins)
    #e_k=[E(i) for i in h]
    #plt.figure()
    plt.plot(h,histo)
    #plt.plot(h,e_k)
    #plt.xlabel("position")
    #plt.ylabel("probability")
    #plt.show
    #print(w)
    
# %%
"""