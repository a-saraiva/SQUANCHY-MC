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
#from mpl_toolkits import mplot3d


# %%

"""Parameters, units: ueV, nm, ns"""

N = 1000 #time_steps and number of tried changes during one sweep.
beta = 0.1
tau = beta/N #length of time_step
bias = 1 # Turn on/off force bias scheme (1 or 0)
interaction = 1 # Turn on/off electron-electron interaction (1 or 0)

if bias == 0:
    print('No Bias')
elif bias == 1:
    print('Bias')
if interaction == 0:
    print('No Interaction')
elif interaction == 1:
    print('Interaction')

# %%

#path1=np.zeros((3,N)) #Initial path: "Cold start"
#path2=np.zeros((3,N))

path1 = np.random.uniform(-2,2,(3,N))
path2 = np.random.uniform(-2,2,(3,N))
'''for i in range(N):
    if path1[2,i] < 0:
        path1[2,i] = -1 * path1[2,i]
    #if path2[2,i] < 0:
     #   path2[2,i] = -1 * path2[2,i]'''

num_path = 100 #number of sweeps

path_list1 = [0] * num_path #path at the end of each sweep
path_list1[0] = deepcopy(path1)
path_list2 = [0] * num_path #path at the end of each sweep
path_list2[0] = deepcopy(path2)

action_list1 = [0] * num_path
action_list2 = [0] * num_path
action_list = [0] * num_path

accrate_list1 = [0] * num_path
accrate_list2 = [0] * num_path
accrate_list = [0] * num_path

m = 5.686 * 10 **(-6) # ueV ns^2/nm^2 - electron mass
hbar = 0.6582 # ueV * ns
a = 0.543 #nm - lattice constant in silicon
w = 10**(4) #ns-1
q = 1
lamda = 0.5 * hbar**2 / m

pot_step = 3 * 10**6 #ueV - Si/SiO2 interface
E_z = 2 * 10**3 #ueV/nm - Electric field in z direction

# %%
"""Different potentials, all one dimensional"""    
def V_2d(x,y):
    return 0.5 * m * w**2 * (x**2 + y**2)

def dV_2d(x,y):
    return [m * w**2 * x, m * w**2 * y, 0]

# Heterostructure interface potential (z-direction) leading to V_airy functions as wave function.
a=10
def V_airy(z,slope,V):
    '''if z<0:
        return(V)
    else:
        return(slope * z)'''
    if z<0:
        return(V / (math.exp(a*z) + 1) - slope * z)
    elif z>2:
        return(slope *z)
    else:
        return(V / (math.exp(a*z) + 1) + slope * z)
    #return V / (math.exp(10*z) + 1) + slope * z
    #return (0.5*m*w**2*z**2)
    
def dV_airy(z,slope,V):
    '''if z<0:
        return([0,0,0])
    else:
        return([0,0,slope])'''
    if z<0:
        return([0,0,-a * V * math.exp(a*z) / ((math.exp(a*z) + 1)**2) - slope])
    elif z>2:
        return([0,0,slope])
    else:
        return([0,0,-a * V * math.exp(a*z) / ((math.exp(a*z) + 1)**2) + slope])
    #return [0, 0, - 10 * V * math.exp(a*z) / ((math.exp(a*z) + 1)**2) + slope]
    #return[0,0,m * w**2*z]

def dist(r1,r2):
    return(np.sqrt(np.dot(np.subtract(r1,r2),np.subtract(r1,r2))))

def ee(r1,r2):
    return(1.23*10**5/dist(r1,r2))


# %%
    
def plot_potential(potl):
    z = np.linspace(-5,5,1000)
    V = np.zeros(1000)
    for i in range(0,1000):
        V[i] = potl(z[i],E_z,pot_step)
    plt.plot(z,V,color='g')
    histo1(2,path_list1)
    #plt.xlim(0,2)
    plt.ylim(0,10000)
#plot_potential(V_airy)
'''for i in range(0,1000):
    plt.plot(z[i],dV_airy(z[i],E_z,pot_step)[2],'.')
#plt.xlim(0.95,1.05)
plt.show()'''

# %%
    
def init_action():
    for i in range(N-1):
        action_list1[0] += np.dot(path1[:,i+1]-path1[:,i],path1[:,i+1]-path1[:,i]) + V_airy(path1[2,i],E_z,pot_step) + V_2d(path1[0,i],path1[1,i])
        action_list2[0] += np.dot(path2[:,i+1]-path2[:,i],path2[:,i+1]-path2[:,i]) + V_airy(path2[2,i],E_z,pot_step) + V_2d(path2[0,i],path2[1,i])
    action_list1[0] += V_airy(path1[2,N-1],E_z,pot_step) + V_2d(path1[0,N-1],path1[1,N-1])
    action_list2[0] += V_airy(path2[2,N-1],E_z,pot_step) + V_2d(path2[0,N-1],path2[1,N-1])
    #print(action_list[0])
#init_action()
#init_action(path2)

# %%
'''    
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = V_2d(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#X = np.arange(-10, 10, 0.25)
#Y = np.arange(-5, 5, 0.25)
#X, Y = np.meshgrid(X, Y)
fig.show()

z=np.linspace(-50,200,N)
for i in range(N):
    plt.plot(z[i],dV_airy(z[i],E_z,pot_step),'.')
plt.show()
'''
# %%

def plot_meshgrid(path):

    H, xedges, yedges = np.histogram2d(path[0,:],path[1,:],bins=10)
    H = H.T  # Let each row list bins with common y range.
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(132, title='pcolormesh: actual edges',aspect='equal')
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H)

    
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    
    # Make data.
    #X1 = np.arange(-5, 5, 0.25)
    #Y2 = np.arange(-5, 5, 0.25)
    #X, Y = np.meshgrid(X, Y)
    #R = np.sqrt(X**2 + Y**2)
    #Z = np.sin(R)
    plt.show()
    
#plot_meshgrid(path1)
#plot_meshgrid(path2)

# %%

"""
One sweep consists of trying N times changing a random element in the array path (size of this change depends on h),
each time accepting/refusing according to Metropolis. 
"""

delta_t = 0.25 # need to adjust such that average acceptance rate is 0.5
h = 2 # tried position change (step size) during Metropolis is from [-h/2,h/2]
#h = delta_t

def sweep():
    
    s=0 # Action change during the sweep
    s1=0
    s2=0
    
    accrate=0 # accuracy rate
    accrate1=0
    accrate2=0
    
    index=np.random.uniform(1,N-1,N) # Order in which path changes are tried. Endpoints (0 and N-1) kept fixed.
    metropolis_comparison = np.random.uniform(0,1,N)
    
    rand_x_1=np.random.uniform(0,1,N) # Creating N random numbers.
    rand_y_1=np.random.uniform(0,1,N)
    rand_z_1=np.random.uniform(0,1,N)
    
    rand_x_2=np.random.uniform(0,1,N) # Creating N random numbers.
    rand_y_2=np.random.uniform(0,1,N)
    rand_z_2=np.random.uniform(0,1,N)
    
    for i in range(0,N):
        time=int(index[i])
        time_m=int((time-1))
        time_p=int((time+1))
                
        """
        Calculating change in action due to replacing path[time] by x_new. Different potentials considered.
        """
        
        so1_p = np.dot((path1[:,time_p]-path1[:,time]),(path1[:,time_p]-path1[:,time]))
        so1_m = np.dot((path1[:,time]-path1[:,time_m]),(path1[:,time]-path1[:,time_m]))
        
        so2_p = np.dot((path2[:,time_p]-path2[:,time]),(path2[:,time_p]-path2[:,time]))
        so2_m = np.dot((path2[:,time]-path2[:,time_m]),(path2[:,time]-path2[:,time_m]))
        
        T_old1 = 0.5 * m * (so1_p + so1_m) / tau**2
        T_old2 = 0.5 * m * (so2_p + so2_m) / tau**2
        T_old = T_old1 + T_old2
        
        def V_spring(path):
            return np.dot(path[:,time_p] - path[:,time], path[:,time_p] - path[:,time]) / (4*lamda*tau) # different time?
        
        def dV_spring(path):
            return (path[:,time_p] - path[:,time] + path[:,time_m] - path[:,time]) / (4*lamda*tau)
        
        V_spring_old1 = V_spring(path1)
        V_spring_old2 = V_spring(path2) 
        dV_spring_old1 = dV_spring(path1)
        dV_spring_old2 = dV_spring(path2)
        
        def V_field(point): # Scalar potential of field @ position
            return np.add(V_airy(point[2],E_z,pot_step), V_2d(point[0],point[1]))

        def dV_field(point): # 3d gradient of field @ position
            return np.add(dV_airy(point[2],E_z,pot_step), dV_2d(point[0],point[1]))
        
        V_field_old1 = V_field(path1[:,time])
        V_field_old2 = V_field(path2[:,time])
        dV_field_old1 = dV_field(path1[:,time])
        dV_field_old2 = dV_field(path2[:,time])

        V_interaction_old = interaction * ee(path1[:,time],path2[:,time])
        
        V_eff_old1 = V_field_old1 + V_spring_old1 
        V_eff_old2 = V_field_old2 + V_spring_old2
        V_eff_old = V_eff_old1 + V_eff_old2 + V_interaction_old
         
        dV_eff_old1 = dV_spring_old1 - 0.5 * tau * dV_field_old1
        dV_eff_old2 = dV_spring_old2 - 0.5 * tau * dV_field_old2
        
        E_old1 = T_old1 + V_field_old1 + V_interaction_old
        E_old2 = T_old2 + V_field_old2 + V_interaction_old
        E_old = E_old1 + E_old2
        
        s_old1 = tau * E_old1
        s_old2 = tau * E_old2
        s_old = s_old1 + s_old2
        
        delta_r1 = delta_t * dV_eff_old1
        delta_r2 = delta_t * dV_eff_old2

        x_new_1 = path1[0,time] + bias * delta_r1[0] + h * (rand_x_1[i]-0.5)
        y_new_1 = path1[1,time] + bias * delta_r1[1] + h * (rand_y_1[i]-0.5)
        z_new_1 = path1[2,time] + bias * delta_r1[2] + h * (rand_z_1[i]-0.5)
        
        x_new_2 = path2[0,time] + bias * delta_r2[0] + h * (rand_x_2[i]-0.5)
        y_new_2 = path2[1,time] + bias * delta_r2[1] + h * (rand_y_2[i]-0.5)
        z_new_2 = path2[2,time] + bias * delta_r2[2] + h * (rand_z_2[i]-0.5)
        
        r_new_1 = [x_new_1,y_new_1,z_new_1]
        r_new_2 = [x_new_2,y_new_2,z_new_2]

        sn1_p = np.dot((path1[:,time_p]-r_new_1),(path1[:,time_p]-r_new_1))
        sn1_m = np.dot((r_new_1-path1[:,time_m]),(r_new_1-path1[:,time_m]))
        
        sn2_p = np.dot((path2[:,time_p]-r_new_2),(path2[:,time_p]-r_new_2))
        sn2_m = np.dot((r_new_2-path2[:,time_m]),(r_new_2-path2[:,time_m]))
        
        T_new1 = 0.5 * m * (sn1_p + sn1_m) / tau**2
        T_new2 = 0.5 * m * (sn2_p + sn2_m) / tau**2
        T_new = T_new1 + T_new2
        
        V_field_new1 = V_field(r_new_1)
        V_field_new2 = V_field(r_new_2)
        V_field_new = V_field_new1 + V_field_new2
       
        V_interaction_new = interaction * ee(r_new_1,r_new_2)

        E_new1 = T_new1 + V_field_new1 + V_interaction_new
        E_new2 = T_new2 + V_field_new2 + V_interaction_new
        E_new = E_new1 + E_new2
        
        s_new1 = tau * E_new1
        s_new2 = tau * E_new2
        s_new = s_new1 + s_new2
        
        delta_s1 = s_new1 - s_old1
        delta_s2 = s_new2 - s_old2
        delta_s = s_new - s_old
        
        def transition_exponent(old_point,new_point):
            return np.dot(new_point - old_point - delta_t * dV_field(old_point), new_point - old_point - delta_t * dV_field(old_point)) / (2 * delta_t)
        
        mu1 = transition_exponent(r_new_1,path1[:,time])
        mu2 = transition_exponent(r_new_2,path2[:,time])
        nu1 = transition_exponent(path1[:,time],r_new_1)
        nu2 = transition_exponent(path2[:,time],r_new_2)

        def accept(old_path,new_path):
            old_path[0] = new_path[0]
            old_path[1] = new_path[1]
            old_path[2] = new_path[2]
        
        '''if delta_s < 0: # always accepting if action lowered
            accept(path1[:,time],r_new_1)
            accept(path2[:,time],r_new_2)
            s += delta_s
            accrate += 1/N'''
        if delta_s1 < 0: # always accepting if action lowered
            accept(path1[:,time],r_new_1)
            #accept(path2[:,time],r_new_2)
            s1 += delta_s1
            accrate1 += 1/N
            s += delta_s1
            accrate += 1/N
        elif nu1 - mu1 + E_old1 - E_new1 > 0: # avoides exponential calculation
            accept(path1[:,time],r_new_1)
            #accept(path2[:,time],r_new_2)
            s1 += delta_s1
            accrate1 += 1/N
            s += delta_s1
            accrate += 1/N
        if delta_s2 < 0: # always accepting if action lowered
            #accept(path1[:,time],r_new_1)
            accept(path2[:,time],r_new_2)
            s2 += delta_s2
            accrate2 += 1/N
            s += delta_s2
            accrate += 1/N
        elif nu2 - mu2 + E_old2 - E_new2 > 0: # avoides exponential calculation
            #accept(path1[:,time],r_new_1)
            accept(path2[:,time],r_new_2)
            s2 += delta_s2
            accrate2 += 1/N
            s += delta_s2
            accrate += 1/N
    #print(accrate)
    return(accrate,accrate1,accrate2,s1,s2,s)

#sweep()    
    
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

def perform_therm():
    for j in range(1,num_path):     
        sweep_path = sweep()
        accrate_list[j] = sweep_path[0]
        accrate_list1[j] = sweep_path[1]
        accrate_list2[j] = sweep_path[2]
        action_list1[j] = action_list1[j-1] + sweep_path[3]
        action_list2[j] = action_list2[j-1] + sweep_path[4]
        action_list[j] = action_list[j-1] + sweep_path[5]
        path_list1[j] = deepcopy(path1)
        path_list2[j] = deepcopy(path2)
    print(np.mean(accrate_list),np.mean(accrate_list1),np.mean(accrate_list2))
    print(action_list[num_path-1])
perform_therm()

#plot_meshgrid(path1)
#plot_meshgrid(path2)    

# %%

def histo1(a,path_list):
    
    li=[]
    for i in range(0,num_path):
        li.append(path_list[i][a,:])
    x_min=np.amin(li)
    x_max=np.amax(li)
    
    #print("N=", N)
    #print("beta", beta)
    
    number_bins=100000
    
    histo = histogram1d(li,range=[x_min,x_max],bins=number_bins)
    h=np.linspace(x_min,x_max,number_bins)
    plt.plot(h,histo)
    #plt.xlim(0.75,1.25)    
    plt.xlabel("position [nm]")
    plt.ylabel("$|\Psi(x)|^2$")

histo1(0,path_list1)
histo1(0,path_list2)
plt.show()
histo1(1,path_list1)
histo1(1,path_list2)
plt.show()
histo1(2,path_list1)
histo1(2,path_list2)
plt.show()

# %%

#plot_potential(V_airy)
histo1(2,path_list1)
histo1(2,path_list2)
plt.xlim(0.75,1.25)
plt.show()

# %%
    
"""
Once a sequence of paths created (using perform), mean and stand_dev functions
plot the mean/standart deviation of each path in the sequence.
"""
def mean(path_list):
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
    
def action():
    x=np.linspace(0,num_path-1,num_path)
    plt.figure()
    plt.plot(x,action_list,'.')
    plt.plot(x,action_list1,'.')
    plt.plot(x,action_list2,'.')
    plt.xlabel("sweep")
    plt.ylabel("action")
    plt.show()
    
action()

def stand_dev(a):
    x_exp=np.zeros(num_path)
    x=np.linspace(0,num_path-1,num_path)
    x_square=0
    #x_mean=0
    for i in range(0,num_path):
        x_mean=0
        for j in range(0,N):
            x_mean+=path_list1[i][a,j]
        x_mean=x_mean/N
        
        for j in range(0,N):
            x_exp[i]+=(path_list1[i][a,j]-x_mean)**2
        x_exp[i]=np.sqrt(x_exp[i]/(N-1))
    for i in range(0,len(x_exp)):
        x_square+=x_exp[i]
    x_square=x_square/num_path
    square=np.ones(num_path)*x_square
    print(x_square)
    #print(x_exp[0],x_exp[10000])
    '''plt.figure()
    plt.plot(x,x_exp,".")
    plt.plot(x,square)
    plt.xlabel("sweep")
    plt.ylabel("standart dev")
    plt.show()'''
    
stand_dev(0)
stand_dev(1)