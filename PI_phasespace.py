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

#from scipy.interpolate import interp1d
import numpy as np
#import PyQt5
#import qtpy

#import sip

#import scipy
#from scipy import signal
#import seaborn as sns
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib.colors as colors
#import Labber
import math
#import peakutils
#from peakutils.plot import plot as pplot
#from sklearn.cluster import KMeans
#from fractions import Fraction
#import matplotlib as mpl

from copy import deepcopy
import random 
import copy
from scipy.stats import linregress
#from mpl_toolkits.mplot3d import Axes3D
from fast_histogram import histogram1d



N= 1000 #time_steps


m= 5.686 * 10 **(-6) # ueV ns^2/nm^2
#m=1
w= 1 #frequency

hbar = 0.6582 # ueV * ns
a = 0.543 #nm 
beta = 10 # ueV-1

epsilon= 1.414 * 10**(6) #ueV
u = 0.6828 * 10**(6) #ueV
v = 0.6119 * 10**(6) #ueV

pot_step= 3 * 10**(6) #ueV
E_field = 0.5 * 10**(3) #ueV/nm

delta= beta/N

#hbar=1
#delta=1

h=5
h_mom=0.8
idrate=0.6

#path=np.zeros(N) #path in position space
#path_mom=np.zeros(N)
#path=np.random.uniform(1,5,N)
path_mom=np.random.uniform(-20,20,N)
#path_mom[0]=0
#path_mom[N-1]=0

path=np.random.uniform(0,20,N+1)
#path[0]=0
#path[N-1]=0

#path_mom=np.random.uniform(-5,5,N)

#path_mom=2.65*np.ones(N) # path in momentum space
#path_mom=2*math.pi*np.ones(N) # path in momentum space
#for i in range(N):
#    path_mom[i]=np.random.uniform(-3,3)

#for i in range(0,len(path)):
#    path[i]=2*np.random.uniform(0,1)
#    path_mom[i]=2*np.random.uniform(0,1)
  

h_linspace=np.linspace(0,10,40) #to plot h's
acc=np.zeros(len(h_linspace)) #accuracy in position
acc_mom=np.zeros(len(h_linspace)) #accuracy in momentum

num_path=2000 #number of paths considered
path_list = [[0]] * num_path #list with saved position paths
path_list_mom = [[0]] * num_path #list with saved momentum paths



"""
Defining potential and dispersion relation
"""

def pot_box(x,a,V):
    if -a<x<a:
        return(0)
    else: return(V)
    
def airy(x,slope,V):
    return((V/(np.exp(x)+1))+slope*x)
    #if x<0:
     #   return(V)
    #else: return(slope*x)
    
    
      

#def E(k):
#    return((epsilon-2*v*math.cos((k/2))+2*u*math.cos(k)))
    
def E(k):
    return((hbar*k)**2/(2*m))
    
#energy=0
#test=[math.cos((delta/hbar)*path_mom[i]*path[i])*math.exp(-(delta/hbar)*(E(path_mom[i])+airy(path[i],0.1,2))) for i in range(0,len(path))]
#energy=np.prod(test)
#for i in range(0,N-1):
    #action+=0.5*m*((path[i+1]-path[i])**2)+airy(path[i],0.1,2)
#energy=[energy]
    
def figure4():
    l=np.linspace(1,num_path-1,num_path)
    #minus_act=[i * -1 for i in action]
    #exp_act=np.exp(minus_act)
    #energy=[np.zeros(len(action))
    
    #for i in range(0,len(action)):
    #    energy[i]=np.log(np.sum(exp_act[:i])/i)
    
    #act_diff=np.zeros(len(l))
    
    #for i in range(0,len(l)):
        #path=np.zeros(l[i]) 
        #action_list=np.zeros(l[i])
        #path=np.random.uniform(-5,5,l[i])
        #act_diff[i]= sweep(path,h,l[i])[2]
        #act_diff[i]=np.mean(action_list)
    plt.figure()
    plt.plot(l,action,".")
    #plt.plot(l,energy,".")
    #plt.plot(x,square)
    plt.xlabel("sweep")
    plt.ylabel("action_diff")
    plt.show 
    
#def E(k):
 #   return(k**2/(2*m))    

#def E(k):
#    return(0.5*m*w**2*k**2)
    
#def E(k):
#    return(0)
    

# %%
"""
k=np.linspace(-20,20,100)
L= [airy(i,E_field,pot_step) for i in k]
#E2= [(math.cos((i-2*math.pi)/2)+math.cos(i-2*math.pi)) for i in k]
#E3= [5*(math.cos((i-2*math.pi)/2)+math.cos(i-2*math.pi)) for i in k]
plt.figure
#plt.plot(k,E1)
plt.plot(k,L)
#plt.plot(k,E3)
plt.xlabel("x")
plt.ylabel('double')
plt.show
"""
# %%



"""
def func(x):
    return(((x+(2*math.pi))%(4*math.pi))-(2*math.pi))
    
k=np.linspace(-40,40,100)
func_k=np.linspace(-40,40,100)

for i in range(len(k)):
    func_k[i]=func(func_k[i])
    
plt.figure()
plt.plot(k,func_k)



k=np.linspace(-2*math.pi,2*math.pi,1000)
E1= [math.cos(i/2)+math.cos(i) for i in k]
E2= [math.cos((i-2*math.pi)/2)+math.cos(i-2*math.pi) for i in k]
E3= [((i%(2*math.pi)) for i in k]
plt.figure
#plt.plot(k,E1)
plt.plot(k,E2)
plt.plot(k,E3)
plt.xlabel("x")
plt.ylabel('double')
plt.show
"""

"""
Defining sweeps in position and momentum space for accuracy measurments
"""

def sweep(path,h):
    
    accrate=0
    index=np.zeros(N)
    for i in range(0,N):
        index[i]=int(np.random.uniform(0,1)*N)

    rand=np.zeros(2*N)
    for i in range(0,2*N):
        rand[i]=np.random.uniform(0,1)
    
    for i in range(0,N):
        tau=int(index[i])
        tau_m=int((tau-1+N)%N)
        tau_p=int((tau+1)%N)
        x_new=path[tau]+h*(rand[i]-0.5)
        s_old=path_mom[tau]*(path[tau_p]-path[tau])+(path[tau]-path[tau_m])+pot_box(path[tau],1,2)+E(path_mom[tau])
        s_new=path_mom[tau]*(path[tau_p]-x_new)+(x_new-path[tau_m])+pot_box(x_new,1,2)+E(path_mom[tau])
        delta_s=s_new-s_old
        
        if rand[i+N]<math.exp(-delta_s):
            path[tau]=x_new
            accrate+=(1/N)
    #h=h*accrate/(idrate)
    return(path,accrate)
        
    #h=h*accrate/(idrate)


def sweep_mom(path_mom,h_mom):
    
    accrate=0
    index=np.zeros(N)
    for i in range(0,N):
        index[i]=int(np.random.uniform(0,1)*N)

    rand=np.zeros(2*N)
    for i in range(0,2*N):
        rand[i]=np.random.uniform(0,1)
    
    for i in range(0,N):
        tau=int(index[i])
        tau_m=int((tau-1+N)%N)
        tau_p=int((tau+1)%N)
        k_new=path_mom[tau]+h_mom*(rand[i]-0.5)
        #k_new=((path_mom[tau]+h_mom*(rand_mom[i]-0.5))%(4*math.pi))-(2*math.pi)
        s_old=path_mom[tau]*(path[tau_p]-path[tau])+(path[tau]-path[tau_m])+pot_box(path[tau],1,2)+E(path_mom[tau])
        s_new=k_new*(path[tau_p]-path[tau])+(path[tau]-path[tau_m])+pot_box(path[tau],1,2)+E(k_new)
        delta_s=s_new-s_old
        
        if rand[i+N]<math.exp(-delta_s):
            path_mom[tau]=k_new
            accrate+=(1/N)
    return(path,accrate)

#%%
    
def sweep_pos_mom(path,path_mom,h,h_mom):
    
    accrate_pos=0
    accrate_mom=0
    #fail=0
    
    #Defining random numbers for position
    #index=np.zeros(N)
    #for i in range(0,N):
    #    index[i]=int(np.random.uniform(0,1)*N)
        
    index=np.random.uniform(1,N,N)
    index_mom=np.random.uniform(0,N,N)

    rand=np.zeros(2*N)
    for i in range(0,2*N):
        rand[i]=np.random.uniform(0,1)
    
    #Defining random numbers for momentum
    
    rand_mom=np.zeros(2*N)
    for i in range(0,2*N):
        rand_mom[i]=np.random.uniform(0,1)
    
    """
    M=0
    for i in range(0,N-1):    
        M+=E(path_mom[i])+airy(path[i],E_field,pot_step)-1j*(path_mom[i]*((path[i+1]-path[i])/delta))
    
    M+=E(path_mom[N-1])+airy(path[N-1],E_field,pot_step)
    """
    
    for i in range(0,N):
        
        #Position
        
        tau=int(index[i])
        x_new=path[tau]+h*(rand[i]-0.5)
        
        
        delta_pot=airy(x_new,E_field,pot_step)-airy(path[tau],E_field,pot_step)
        so1=path_mom[tau]*((path[tau+1]-path[tau])/delta)
        so2=path_mom[tau-1]*((path[tau]-path[tau-1])/delta)
        sn1=path_mom[tau]*((path[tau+1]-x_new)/delta)
        sn2=path_mom[tau-1]*((x_new-path[tau-1])/delta)
        
        if np.sign(math.cos(delta*sn1)*math.cos(delta*sn2)/(math.cos(delta*so1)*math.cos(delta*so2)))==-1:
            continue
        
        #if rand[i+N]<math.exp(-(delta/hbar)*(delta_pot))*math.cos(delta*sn1)*math.cos(delta*sn2)/(math.cos(delta*so1)*math.cos(delta*so2)):
        
        if math.log(rand[i+N])<(-(delta/hbar)*(delta_pot))*math.log(math.cos(delta*sn1)*math.cos(delta*sn2)/(math.cos(delta*so1)*math.cos(delta*so2))):
                
                path[tau]=x_new
                
                #M+=delta_M
                accrate_pos+=(1/N)
        
                
        #   
        """
        s_old=((0.5*m)/(delta))*((path[tau+1]-path[tau])**2+(path[tau]-path[tau-1])**2)+delta*airy(path[tau],E_field,pot_step)
        s_new=((0.5*m)/(delta))*((path[tau+1]-x_new)**2+(x_new-path[tau-1])**2)+delta*airy(x_new,E_field,pot_step)
        delta_s=s_new-s_old
        
        
        if delta_s<0:
            path[tau]=x_new
            accrate_pos+=(1/N)
            #s+=delta_s
        elif rand[i+N]<math.exp(-(delta_s/hbar)):
            path[tau]=x_new
            accrate_pos+=(1/N)
            #s+=delta_s
        """
        #
                
        #Momentum
        
        zau=int(index_mom[i])
        k_new=path_mom[zau]+h_mom*(rand_mom[i]-0.5)
        
        delta_E=E(k_new)-E(path_mom[zau])
        Eo=path_mom[zau]*((path[zau+1]-path[zau])/delta)
        En=k_new*((path[zau+1]-path[zau])/delta)
        
        
        """
        if delta_E<0:
            path_mom[zau]=k_new
            accrate_pos+=(1/N)
            #s+=delta_s
        elif rand_mom[i+N]<math.exp(-(delta*delta_E/hbar)):
            path_mom[zau]=k_new
            accrate_pos+=(1/N)
            #s+=delta_s
        """
        
        
        
        if np.sign(math.cos(delta*En)/math.cos(delta*Eo))==-1:
            continue
        
        
        #if rand_mom[i+N]<math.exp(-(delta/hbar)*(delta_E))*math.cos(delta*En)/math.cos(delta*Eo):
        if math.log(rand_mom[i+N])<(-(delta/hbar)*(delta_E))+math.log(math.cos(delta*En)/math.cos(delta*Eo)):
                
                path_mom[zau]=k_new
                
                #M+=delta_M
                accrate_mom+=(1/N)
        
                
    return(path,path_mom,accrate_pos,accrate_mom)

# %%

def sweep_pos_mom_1(path,path_mom,h,h_mom):
    
    accrate=0
    
    #fail=0
    
    #Defining random numbers for position
    #index=np.zeros(N)
    #for i in range(0,N):
    #    index[i]=int(np.random.uniform(0,1)*N)
        
    index=np.random.uniform(1,N-1,N)
    #index_mom=np.random.uniform(0,N,N)

    rand=np.zeros(2*N)
    for i in range(0,2*N):
        rand[i]=np.random.uniform(0,1)
    
    #Defining random numbers for momentum
    
    rand_mom=np.zeros(2*N)
    for i in range(0,2*N):
        rand_mom[i]=np.random.uniform(0,1)
        
    for i in range(0,N):
    
    
        tau=int(index[i])
        x_new=path[tau]+h*(rand[i]-0.5)
        k_new=path_mom[tau]+h_mom*(rand_mom[i]-0.5)
        
        s_old_H=(E(path_mom[tau])+airy(path[tau],E_field,pot_step))
        s_new_H=(E(k_new)+airy(x_new,E_field,pot_step))
        
    
            
        
        s_old_1=path_mom[tau]*((path[tau]-path[tau-1])/delta)
        s_old_2=path_mom[tau+1]*((path[tau+1]-path[tau])/delta)
        s_new_1=k_new*((x_new-path[tau-1])/delta)
        s_new_2=path_mom[tau+1]*((path[tau+1]-x_new)/delta)
        
            #delta_M=(s_new_H-s_old_H)-1j*(s_new-s_old)*hbar
            #delta_M=(s_new_H-s_old_H)
        
            #print(s_new_2+s_old_2)
    
            #if delta_M<0:
             #   path[tau]=x_new
              #  path_mom[tau]=k_new
               # M+=delta_M
                #accrate+=(1/N)
                
            #if math.cos((delta)*(s_new-s_old))<0:
            #    fail+=(1/N)
            #    continue
                
            
        if rand[i+N]<math.exp(-(delta/hbar)*(s_new_H-s_old_H))*math.cos(delta*s_new_1)*math.cos(delta*s_new_2)/(math.cos(delta*s_old_1)*math.cos(delta*s_old_2)):
                
             path[tau]=x_new
             path_mom[tau]=k_new
             #M+=delta_M
             accrate+=(1/N)
                
    
    
    """
        if tau==0:
            s_old_2=path_mom[tau]*((path[tau+1]-path[tau])/delta)
            s_new_2=k_new*((path[tau+1]-x_new)/delta)+path_mom[tau-1]
            
            delta_M=(s_new_H-s_old_H)-1j*(s_new_2-s_old_2)
            #print(s_new_2+s_old_2)
            #delta_M=(s_new_H-s_old_H)
            if math.cos((delta/hbar)*(s_new_2-s_old_2))<0:
                fail+=(1/N)
                continue
    
            if math.log(rand[i+N])<(-(delta/hbar)*(s_new_H-s_old_H))*math.log(math.cos((delta/hbar)*(s_new_2-s_old_2))):
                path[tau]=x_new
                path_mom[tau]=k_new
                M+=delta_M
                accrate+=(1/N)
            
            #elif math.log(rand[i+N])<(-(delta/hbar)*delta_M):
             #   path[tau]=x_new
              #  path_mom[tau]=k_new
               # M+=delta_M
                #accrate+=(1/N)
            
        elif tau==N-1:
            
            s_old_2=path_mom[tau-1]*((path[tau]-path[tau-1])/delta)
            s_new_2=path_mom[tau-1]*((x_new-path[tau-1])/delta)
            
            delta_M=(s_new_H-s_old_H)-1j*(s_new_2-s_old_2)
            #delta_M=(s_new_H-s_old_H)
            
            #print(s_new_2+s_old_2)
            
            #if delta_M<0:
             #   path[tau]=x_new
              #  path_mom[tau]=k_new
               # M+=delta_M
                #accrate+=(1/N)
            
            if math.cos((delta/hbar)*(s_new_2-s_old_2))<0:
                fail+=(1/N)
                continue
            
            if math.log(rand[i+N])<(-(delta/hbar)*(s_new_H-s_old_H))*math.log(math.cos((delta/hbar)*(s_new_2-s_old_2))):
                path[tau]=x_new
                path_mom[tau]=k_new
                M+=delta_M
                accrate+=(1/N)
    """
            
       
    """
    M=0
    for i in range(0,N-1):    
        M+=(path_mom[i]*(path[i+1]-path[i]))
    
    for i in range(0,N):
        tau=int(index[i])
        x_new=path[tau]+h*(rand[i]-0.5)
        k_new=path_mom[tau]+h_mom*(rand_mom[i]-0.5)
        
        s_old_H=(E(path_mom[tau])+airy(path[tau],E_field,pot_step))
        s_new_H=(E(k_new)+airy(x_new,E_field,pot_step))
        
        M_new=M-path_mom[tau]*(path[tau+1]-path[tau])+path_mom[tau-1]*(path[tau]-path[tau-1])+k_new*(path[tau+1]-x_new)+path_mom[tau-1]*(x_new-path[tau-1])
        
        exp_M=np.exp(-1j*(delta/hbar)*M)
        exp_M_new=np.exp(-1j*(delta/hbar)*M_new)
            
        L=exp_M.real/exp_M_new.real
        
        #print(s_new_H-s_old_H)
       
        
        if abs(s_new_H-s_old_H)>10:
            path[tau]=x_new
            path_mom[tau]=k_new
            accrate+=(1/N)
            
        elif rand[i+N]<math.exp(-(delta/hbar)*(s_new_H-s_old_H))*L:
            path[tau]=x_new
            path_mom[tau]=k_new
            accrate+=(1/N)
    """
        
    
    """
        L=(M/M_new).real
        
        if L >1:
            path[tau]=x_new
            path_mom[tau]=k_new
            accrate+=(1/N)
        
        elif rand[i+N]<L:
            path[tau]=x_new
            path_mom[tau]=k_new
            accrate+=(1/N)
    """
        
            
            
    
    """
    for i in range(0,N):
        tau=int(index[i])
        tau_m=int((tau-1+N)%N)
        tau_p=int((tau+1)%N)
        #tau_m=int((tau-1))
        #tau_p=int((tau+1))
        x_new=path[tau]+h*(rand[i]-0.5)
        #k_new=path_mom[tau]+h_mom*(rand_mom[i]-0.5)
        #k_new=(((path_mom[tau]+h_mom*(rand_mom[i]-0.5))+(2*math.pi))%(4*math.pi))-(2*math.pi)
        k_new=(path_mom[tau]+h_mom*(rand_mom[i]-0.5))
        
        #if k_new<2*math.pi:
        #    k_new+=4*math.pi
        #if k_new>2*math.pi:
        #    k_new-=4*math.pi
        
        #s_old=path_mom[tau]*((path[tau_p]-path[tau])+(path[tau]-path[tau_m]))+E(path_mom[tau])+0.5*m*(w)**2*path[tau]**2
        #s_old=E(path_mom[tau])+0.5*m*(w)**2*path[tau]**2
        #s_old=-path_mom[tau]*((path[tau_p]-path[tau])+(path[tau]-path[tau_m]))+E(path_mom[tau])+airy(path[tau],0.1,2)
        #s_old=path_mom[tau]*((path[tau_p]-path[tau])+(path[tau]-path[tau_m]))-E(path_mom[tau])-pot_box(path[tau],2,2)
        #s_new=k_new*((path[tau_p]-x_new)+(x_new-path[tau_m]))+E(k_new)+0.5*m*(w)**2*x_new**2
        #s_new=E(k_new)+0.5*m*(w)**2*x_new**2
        #s_new=-k_new*((path[tau_p]-x_new)+(x_new-path[tau_m]))+E(k_new)+airy(x_new,0.1,2)
        #s_new=k_new*((path[tau_p]-x_new)+(x_new-path[tau_m]))-E(k_new)-pot_box(x_new,2,2)
        #delta_s=s_new-s_old
        #print(delta_s)
        
        s_old_H=(delta/hbar)*(E(path_mom[tau])+airy(path[tau],E_field,pot_step))
        #s_old_2=(delta/hbar)*path_mom[tau]*((path[tau_p]-path[tau])+(path[tau]-path[tau_m]))
        s_old_1=(delta/hbar)*path_mom[tau]*(path[tau_p]-path[tau])
        #s_old_2=(delta/hbar)*path_mom[tau_m]*(path[tau]-path[tau_m])
        
        s_new_H=(delta/hbar)*(E(k_new)+airy(x_new,E_field,pot_step))
        #s_new_2=(delta/hbar)*k_new*((path[tau_p]-x_new)+(x_new-path[tau_m]))
        s_new_1=(delta/hbar)*k_new*(path[tau_p]-x_new)
        s_new_2=(delta/hbar)*path_mom[tau_m]*(x_new-path[tau_m])
        
        
        
        #delta_s=s_new_1-s_old_1
        #print(p2/p1)
        
        #if abs(s_new_H-s_old_H)>10:
         #   continue
        
        delta_s=s_new_H-s_old_H
        #print(delta_s)
        
        
         
        #if abs(delta_s)<10:
            #if rand[i+N]<math.exp(-(s_new_1-s_old_1))*math.cos(s_new_2)/math.cos(s_old_2):
        #if rand[i+N]<math.exp(-(s_new_1-s_old_1)):
    """
        
    """
        if delta_s<-100:
            continue
        compare = math.exp(-(s_new_H-s_old_H))*math.cos(s_new_1-s_old_1)
        if rand[i+N]<compare:
            path[tau]=x_new
            path_mom[tau]=k_new
            accrate+=(1/N)
        
        
        
        #print(delta_s)
        #if delta_s <0:
        #    path[tau]=x_new
        #    path_mom[tau]=k_new
        #    accrate+=(1/N)
            #s+=delta_s
        
        
        
        
        if delta_s<-100:
            path[tau]=x_new
            path_mom[tau]=k_new
            accrate+=(1/N)
            fail+=(1/N)
            
        elif rand[i+N]<math.exp(-(s_new_H-s_old_H))*math.cos(s_new_1-s_old_1+s_new_2):
            path[tau]=x_new
            path_mom[tau]=k_new
            accrate+=(1/N)
    """
        
        
    return(path,path_mom,accrate)
    

    
# %%
def acc(h,hh):
    saved=np.zeros(num_path)
    for i in range(0,num_path):
        saved[i]=sweep_pos_mom(path,path_mom,h,hh)[2]
    x=np.linspace(1,num_path,num_path)
    plt.figure()
    plt.plot(x,saved)
    plt.xlabel("sweep")
    plt.ylabel("accuracy")
    plt.show
        



def accuracy(h_linspace):
    for i in range(0,100):
        sweep(path,h_mom)
    acc=np.zeros(len(h_linspace))
    for k in range(0,len(h_linspace)):
        for i in range(0,10):
            for j in range(0,100):
                sweep(path,h_linspace[k])
            acc[k]+=sweep(path,h_linspace[k])[1]
        acc[k]=acc[k]/10
    
    plt.figure
    plt.plot(h_linspace,acc)
    plt.xlabel("h")
    plt.ylabel("acceptance rate")
# %%
def accuracy_mom(h_linspace):
    for i in range(0,100):
        sweep_mom(path_mom,h_mom)
    acc_mom=np.zeros(len(h_linspace))
    for k in range(0,len(h_linspace)):
        for i in range(0,10):
            for j in range(0,100):
                sweep_mom(path_mom,h_linspace[k])
            acc_mom[k]+=sweep_mom(path_mom,h_linspace[k])[1]
        acc_mom[k]=acc_mom[k]/10
    
    plt.figure
    plt.plot(h_linspace,acc_mom)
    plt.xlabel("h")
    plt.ylabel("acceptance rate")

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
        
def perform():
    for i in range(0,4000):
        sweep(path,h)
    for j in range(0,num_path):
        path_list[j]=deepcopy(path)
        for k in range(0,10):
            sweep(path,h)
            
def perform_PS_therm():
    #h=h1
    #h_mom=h_mom1
    for j in range(0,num_path):
        sweep_pos_mom(path,path_mom,h,h_mom)
        #h=h*sweep_pos_mom(path,path_mom,h,h_mom)[2]/(idrate)
        #h_mom=h_mom*sweep_pos_mom(path,path_mom,h,h_mom)[2]/(idrate)
        path_list[j]=deepcopy(path)
        path_list_mom[j]=deepcopy(path_mom)
       
        #for k in range(0,12):
            #sweep_pos_mom(path,path_mom,h,h_mom)
            #h=h*sweep_pos_mom(path,path_mom,h,h_mom)[2]/(idrate)
            #h_mom=h_mom*sweep_pos_mom(path,path_mom,h,h_mom)[2]/(idrate)

ll=np.linspace(0,N,N)

def paths(h):
    plt.figure()
    for i in range(0,5):
        plt.plot(ll,path)
        sweep(path,h)
    plt.xlabel("time")
    plt.ylabel("position")
    


def x_2(array):
    squared=np.zeros(len(array))
    for i in range(0,len(array)):
        squared[i]=array[i]**2
    return(squared)


def fig_prob_density():
    parabola_x=np.linspace(-2,2,50)
    parabola_y=x_2(parabola_x)
    plt.figure()
    plt.hist(path_list,50)
    plt.plot(parabola_x,parabola_y)
    plt.xlabel("position")
    plt.ylabel("probability")
    plt.show
    
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
    
def stand(m,w):
    R=1+(w**2/2)-w*np.sqrt(1+(w**2)/4)
    a=1/(2*m*w*np.sqrt(1+0.25*w**2))*((1+R**N)/(1-R**N))
    return(a)
    
# %%
    
def histo(path):
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
acc_p=np.zeros((10,10))

def acc_plane():
    path=np.zeros(N)
    path_mom=np.zeros(N)
    
    h=np.linspace(0,0.25,10)
    h_mom=np.linspace(0,25,10)
    for i in range(0,len(h)):
        for j in range(0,len(h_mom)):
            for l in range(0,100):
                sweep_pos_mom(path,path_mom,h[i],h_mom[j])
            acc_p[i][j]=sweep_pos_mom(path,path_mom,h[i],h_mom[j])[2]
            path=np.zeros(N)
            path_mom=np.zeros(N)
            
    
    plt.figure()
    plt.pcolormesh(h, h_mom, acc_p,cmap="inferno")
    plt.colorbar()
    #plt.gca().set_aspect("equal") # x- und y-Skala im gleichen Maßstaab
    plt.show()
            

# %%
   
    
    
# %%
"""
coeff=[2.5,4,6,8,10]
plt.figure()
for i in range(0,6):
    perform_PS_therm(i)
    histo(path_list_mom)
    fig_stand_dev()
    fig_stand_dev_mom()
    path=np.zeros(N) #path in position space
    path_mom=np.zeros(N)
"""

# %%
"""
plt.figure()
histo(path_list_mom[0:2])
histo(path_list_mom[5:10])
histo(path_list_mom[15:20])
histo(path_list_mom[25:30])
plt.show  
"""

# %%
"""
a=0
for i in range(0,len(path_mom)):
    a+=path_mom[i]
    print(a/len(path_mom))
"""

# %%