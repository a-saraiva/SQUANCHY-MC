#!/usr/bin/env python
# coding: utf-8

# # Path Integral Montecarlo
# 
# 
# ## Setup
# This first part of the code is used to set up the input parameters for the ***code***. 
# 
# ### List of corrections to the previous code
#     
#    1. Added __exchange update__
#    2. **Changed displacement constant** for a **vector variable** initialized as h = [2,1,.08]. It makes sense, since the tree dimensions are at a different scale. 
#    3. Created **look-up tables** where the potentials are saved in a refined grid
#    4. **Sweep**:
#        - Created new vectorized form of sweep. Instead sweeping one path at the time, it moves randomly several time components before accepting the sweep (I still need to include vectorized V_field to make it work properly)
#        - Removed definitions from the method code. They were making the algorithm work slowly
#        - By having a list of the action for every site in the path, we could avoid submitting doing the computations twice at each sweep. 
#    5. **Vectorized potentials** (Any non-included potential already admits vectorization)
#        - V_step 
#        - ee - electron-electron interaction
#        - V_hrl
#        - V_doubledot
#        - double_H0
#        - V_HOld 
#        - V_het
#    6. **Find_action** 
#         a) 100 times more efficient with vectorization
#         b) Modified representation of time derivative (increased precision)
#    7. Included **3D plots** of HRL potential
#    8. Included changes in methods for interpolation ( from 'linear' to 'nearest neighbors'). 50% faster. Slight change at each computation in the third significant figure. *(not necessary with the look-up tables)*
#     
# 
# ### Questions
#    - Do you have result for the exchange coupling to make comparisons?
#    - How are we going to sort the sign problem?
#     
# ### Ideas
#    - Create a main function could give legibility to the code. General constants will be removed to avoid errors. 
#    - Creating  a voltage function that could stand several modifications to the Hamiltonian depending on the desired code. This should be a simple way to add more flexibility to the program. 
#    - It is necessary to implement periodic boundary conditions in order to avoid random paths getting out of the region where the HRL potential is defined. (Having second thoughts. It is unlikely that the paths could escape from the given region. However,  )
#    - Random sampling kinetic energy as a Gaussian distribution could help improving the efficiency of the algorithm
#    - We can implement a vectorized displacement just a in the center of mass sweep in worm PIMC paper
#    - No need to compute old variables at each sweep. Save them in a vector and modify them when the sweep is accepted. (Experiment showed a decrease in runtime by 30%)
#    - As far as I observe, the new implementation of the code, which promises to be 130 times faster, is more similar to the worm algorithm, than to the previous idea of it. Wouldn't it be better to use the already implement PIMC- Del Maestro's code? If not, my suggestion is to create a code that at be end would integrate the ideas of worm algorithms. ** The worm code specializes on bosonic systems with a large number of particles. This is not our case ** 
# 
# 
# 
# 
#     
# 

# In[176]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:50:33 2019

@author: SQUANCHY-MC
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy import stats

# RC - params . Better plots
def SetPlotParams():
    '''
     Inputs:
     Return:
    '''
    plt.rcParams.update({'font.size': 25})
    plt.rc("text", usetex = True)
    plt.rc("font", family = "serif")
    plt.rcParams['figure.subplot.hspace'] = 0.3
    plt.rcParams['figure.subplot.wspace'] = 0.1
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['axes.formatter.limits'] = -3 , 3
    plt.rc('xtick.major', pad = 1)
    plt.rc('ytick.major', pad = 1)
    plt.rcParams['axes.labelpad'] = 15
    plt.rcParams['axes.formatter.use_mathtext'] = False
    plt.rcParams['figure.autolayout'] = True
    # plt.rcParams['figure.figsize'] =  8.3, 6.8
    plt.rcParams['figure.figsize'] =  5, 6
# %%
SetPlotParams()


# -------------------------------- Methods for path generation, crossings and staying----------------------------------------

def generate_paths_N_crossings(N, crossings):
    '''Generates random paths of 2 electrons with a defined number of crossings'''
    pathx1 = np.zeros(N) 
    pathx2 = np.zeros(N) 
    
    # arr = np.arange(100,N-100,200)
    # np.random.shuffle(arr)
    # List= np.sort(arr[:crossings])
    # CrossingPoints = np.zeros(crossings+2)
    # CrossingPoints[0] = 0
    # CrossingPoints[-1] = N
    # CrossingPoints[1:-1] = List
    CrossingPoints = np.linspace(0,N, crossings +2)

    CrossingPoints = CrossingPoints.astype(int)
    mean_x1 = -20
    mean_x2 = -30
    x_std  = 14
    for i in range(crossings+1):
        if i%2 == 0:
            # pathx1[CrossingPoints[i]:CrossingPoints[i+1]] = np.random.normal(loc=mean_x1
                                                                            # ,scale=x_std
                                                                            # ,size=CrossingPoints[i+1]-CrossingPoints[i])
            # pathx2[CrossingPoints[i]:CrossingPoints[i+1]] = np.random.normal(loc=mean_x2
                                                                            # ,scale=x_std
                                                                            # ,size=CrossingPoints[i+1]-CrossingPoints[i])

            pathx1[CrossingPoints[i]:CrossingPoints[i+1]] = np.random.uniform(-60,-20,CrossingPoints[i+1]-CrossingPoints[i])
            pathx2[CrossingPoints[i]:CrossingPoints[i+1]] = np.random.uniform(-40,5,CrossingPoints[i+1]-CrossingPoints[i])
        else:
            # pathx1[CrossingPoints[i]:CrossingPoints[i+1]] = np.random.normal(loc=mean_x2
                                                                            # ,scale=x_std
                                                                            # ,size= CrossingPoints[i+1]-CrossingPoints[i])
            # pathx2[CrossingPoints[i]:CrossingPoints[i+1]] = np.random.normal(loc=mean_x1
                                                                            # ,scale=x_std
                                                                            # ,size= CrossingPoints[i+1]-CrossingPoints[i])

            pathx1[CrossingPoints[i]:CrossingPoints[i+1]] = np.random.uniform(-40,5,CrossingPoints[i+1]-CrossingPoints[i])
            pathx2[CrossingPoints[i]:CrossingPoints[i+1]] = np.random.uniform(-60,-20,CrossingPoints[i+1]-CrossingPoints[i])

    # pathy1 = np.random.normal(loc=0, scale=10, size=N)
    # pathy2 = np.random.normal(loc=0, scale=10, size=N)
    pathy1 = np.random.uniform(-12,12,N)
    pathy2 = np.random.uniform(-12,12,N)
    
    pathz1 = np.random.normal(loc=-1, scale=0.5, size=N)
    pathz2 = np.random.normal(loc=-1, scale=0.5, size=N)
    
    # pathz2 = np.random.uniform(-6,-0.5,N)
    # pathz1 = np.random.uniform(-6,-0.5,N)

    path1 = np.array([pathx1,pathy1,pathz1])
    path2 = np.array([pathx2,pathy2,pathz2])
    Perm = (-1)**(crossings)
    
    return path1 , path2, Perm




# ##------------------------------------- Vectorized Potentials-------------------------------------------------------------


    '''From now on all potentials are modified to include vectorization'''

def V_step(z_path):
    return(pot_step / (np.exp(-(4/a)*z_path) + 1))

def dist(path1,path2):
    minus = path1-path2
    return np.sqrt(minus[0,:]**2 + minus[1,:]**2 + minus[2,:]**2)
#     return np.sqrt(np.matmul(minus.T,minus))

def ee(path1,path2):
    return(eec/dist(path1,path2))

#---------------------------------------------------GRID - Potentials--------------------------------------------------

def roundGrid(vec,grid_step):
    '''transforms the values in grid into the approximated integer number on it'''
    return vec//grid_step + (vec%grid_step)//(grid_step/2)

#Define Grid and minimum array
# pathLenght = 100
# XS, YS , ZS, Grid =  RefineGrid(xs,ys,zs)
# Z_max = np.argmax(ZS)
def CreateMinArray(pathLenght):
    '''creates a 3*N array of min values of XS to displace the grid 
    Inputs:
    Return: 3*N array where mins[xs,:] = min(xs) 
    '''
    mins= np.array([min(XS),min(YS),min(ZS)])
    min_array = np.zeros((3,pathLenght))
    min_array[0,:] +=  roundGrid(mins[0],dXS) 
    min_array[1,:] +=  roundGrid(mins[1],dXS)  
    min_array[2,:] +=  roundGrid(mins[2],dZS) 
    min_array = min_array.astype(int)
    return min_array


def ComputePotential(Grid_Func, path, min_array):
    '''
    Computes the potential interpolating the path over an integer look up table
    Inputs: Grid_Func -- Look-up table of already computed potential over the grid
            path - N-array 
    Return: The Grid 
    '''
    #Get indices of the paths in XS,YS,ZS coordinates
    argsXY = roundGrid(path[:-1,:],dXS)  - min_array[:-1,:]
    argsXY = argsXY.astype(int)
    argsZ = roundGrid(path[-1,:],dZS) - min_array[-1,:]
#     argZ = np.round(path[-1,:]*10).astype(int) - min_array[-1,:]
    argZ = argsZ.astype(int)
    argZ[argZ > Z_max] = Z_max
#     print(argZ)
    return Grid_Func[argsXY[0],argsXY[1],argZ] + V_step(path[2,:])
    # ------------------------------COMPUTING PATH ENERGY ---------------------------------------

def timeDerivative(path):
    ''' Gets the time derivative of the path 
     Inputs: path (3*N)
     Return: derivative (3*N-4)
    '''
    return (path[:,1:]-path[:,:-1]) 
    
    #return (1/2)*path[:,2:]-path[:,:-2]
    # if np.random.rand(1) > 1:
    #     k1 = 8*(path[:,2:]-path[:,:-2]) 
    #     k2 = k1[:,1:-1] - 1 * (path[:,4:]-path[:,:-4])
    #     return (1/12)*k2
    # else:


def KineticEnergy(MassVec,path):
    '''Computes kinetic energy of path of an electron path
       inputs: MassVec(3*1-array)
               path(3*N-array)
       return: Energy vector(N-array)
    '''
    # return (.5/(4*tau2))*np.matmul(np.transpose(MassVec),(path[:,2:]-path[:,:-2])**2)
    return np.sum((.5/tau2)*np.matmul(np.transpose(MassVec),(timeDerivative(path))**2))

def PotentialEnergy(path1,path2,GridFunc, minArray):
    ''' Computes potential energy of a pair of electron paths
     Inputs: paths (3*N)array
             GridFuction ---Requirements for compute potential
     Return:

    '''
    V_interaction =  np.sum(ee(path1,path2))
    #Action
    pot1 = np.sum(ComputePotential(GridFunc, path1, minArray))  # + V_interaction
    pot2 = np.sum(ComputePotential(GridFunc, path2, minArray))   # + V_interaction
    return V_interaction +pot1 +pot2


def find_Grid_action(path1,path2,GridFunc, minArray):
    '''Computes the action given two electron paths
       inputs: Path1 , Path2: 3*N-arrays
       outputs: Action(real)
    '''
    #Kinetic energy
    kin1 = KineticEnergy(m,path1)
    kin2 = KineticEnergy(m,path2)
    #Potential energy
    Pot = PotentialEnergy(path1,path2,GridFunc, minArray) 
    return tau*(kin1 + kin2 + Pot)  #+ 2 * np.log(2 * np.pi * tau / meff)

    
def KineticTotalEnergy(MassVec,path1,path2, Perm):
    '''Computes kinetic energy of the total periodic path of an electron path
       inputs: MassVec(3*1-array)
               path(3*N-array)
       return: Energy vector(N-array)
    '''
    
    if Perm == 1:

        Rpath1 = np.roll(path1,-2,(1))
        k1 = (.5/(4*tau2))*np.matmul(np.transpose(MassVec),(Rpath1-path1)**2)
        
        Rpath2 = np.roll(path2,-2,(1))
        k2 = (.5/(4*tau2))*np.matmul(np.transpose(MassVec),(Rpath2-path2)**2)

        return np.sum(k1 + k2)
    
    else:
        BigPath = np.concatenate((path1,path2), axis = 1)
        RBigPath = np.roll(BigPath,-2,(1))
        return np.sum((.5/(4*tau2))*np.matmul(np.transpose(MassVec),(RBigPath-BigPath)**2))
       
def PotentialTotalEnergy(path1,path2,GridFunc, minArray):
    '''Computes kinetic energy of the total periodic path of an electron path
       inputs: MassVec(3*1-array)
               path(3*N-array)
       return: Energy vector(N-array)
    '''
    #Interaction
    V_interaction =  np.sum(ee(path1,path2))
    #Potential energy
    path1Potential = np.sum(ComputePotential(GridFunc, path1,minArray))
    path2Potential = np.sum(ComputePotential(GridFunc, path2,minArray))
    return V_interaction + path1Potential + path2Potential
       


def find_Grid_Total_action(path1,path2,Perm,GridFunc, minArray):
    '''Computes the action given two electron paths
       inputs: Path1 , Path2: 3*N-arrays
       outputs: Action(real)
    '''
    #Kinetic energy
    s1 = tau*KineticTotalEnergy(m,path1,path2, Perm)
    
    #Action
    s1 += tau*PotentialTotalEnergy(path1,path2,GridFunc, minArray)
    return s1
    

'''------------------------------------- Updates --------------------------------'''


'''Gaussian Update'''
def getGaussianDistribution(path,Mvec):
    ''' Sample new path according to the kinetic propagator in the gaussian distribution
     Inputs: path (3*N array) , Mvec(3-array)
     Return: randPath(3*(N-2) arra)

    '''
    init = path[:,0]
    end = path[:,-1]
    
    timelength = len(path[0,:])
    times = np.linspace(1,  timelength-2, timelength-2)

    #Reshaping in matrix form for multiplication
    times1 = times.reshape(1,-1) 
    init1 = init.reshape(-1,1) 
    end1 = end.reshape(-1,1) 
    Mvec1 = Mvec.reshape(-1,1)
    
    #harmonic mean for variance computation
    timeArr = np.array([times1,times1[:,::-1]])[:,0,:]
    harm = stats.hmean(timeArr,axis = 0).reshape(-1,1)
    
    #Computing expected mean and variance
    rmean = (1/(timelength-1))*(np.matmul(init1,times1[:,::-1])+ np.matmul(end1,times1 ))
    sigma = 0.5*(tau*hbar)*(np.matmul((1/Mvec1),harm.T))
    
    #Creating new path
    newPath = np.zeros(np.shape(path))
    newPath[:,1:-1] = np.random.normal(rmean, np.sqrt(sigma))
    #Fixing boundaries
    newPath[:,0] = np.copy(init)
    newPath[:,-1] = np.copy(end)
    return newPath

def gaussianUpdate(path1,path2,GridFunc,min_array):
    '''Samples a gaussian probability according to the kinetic propagator and replaces it for path1 if it reduces the
       potential energy
       Inputs: Path1 , Path2: 3*N-arrays
               Sold - old action (float)
       outputs: newPath1, newPath2 (3N-arrays)
                delta_s - difference between new action and old action
    '''
    
    #sampling new paths according to gaussian distribution
    # if  np.random.rand(1) > .5  :
    new_path1 = getGaussianDistribution(path1,m)
    new_path2 = path2
    old_pot = np.sum(ComputePotential(GridFunc, path1, min_array))
    new_pot = np.sum(ComputePotential(GridFunc, new_path1, min_array))
    # else:
    #     new_path1 = path1
    #     new_path2 = getGaussianDistribution(path2,m)
    #     old_pot = np.sum(ComputePotential(GridFunc, path2, min_array))
    #     new_pot = np.sum(ComputePotential(GridFunc, new_path2, min_array))
    
    old_int =  np.sum(ee(path1,path2))
    new_int =  np.sum(ee(new_path1,new_path2))
    #add action
    delta_s = tau*( (new_pot + new_int) - (old_pot + old_int))

    if delta_s < 0: # always accepting if action lowered
        return new_path1 , new_path2 , delta_s
    elif np.random.rand(1) < np.exp(-(delta_s/hbar)): #otherwise accept with PI probability.
        return new_path1 , new_path2 , delta_s
    else: 
        return path1 , path2, 0

'''CM-Update'''

def CMupdate(path1,path2,GridFunc,min_array):
    ''' Sweeps the Center of mass of path1 in order to reduce the potential leaving the energy static
        Inputs: path1, path2
                GridFunc ,min_array : Necessary to compute potential
        Output: newPath1, newPath2
    '''
    N = len(path1[0,:])
    # select a 3-random vector for displacement
    rand_r1 = (np.random.rand(3)-.5)
    rand_r1[0] = h[0]*rand_r1[0]
    rand_r1[1] = h[1]*rand_r1[1]
    rand_r1[2] = h[2]*rand_r1[2]
    
    new_path1 = np.zeros(np.shape(path1))
    new_path2 = np.zeros(np.shape(path1))
    #Sweep
    new_path1[0,:] = path1[0,:] + rand_r1[0]
    new_path1[1,:]  = path1[1,:] + rand_r1[1]
    new_path1[2,:]  = path1[2,:] + rand_r1[2]
    new_path2 = path2 
    
    #Compute action
    old_pot = np.sum(ComputePotential(GridFunc, path1, min_array))
    new_pot = np.sum(ComputePotential(GridFunc, new_path1, min_array))
    
    old_int =  np.sum(ee(path1,path2))
    new_int =  np.sum(ee(new_path1,new_path2))
    
    delta_s = tau*( (new_pot + new_int) - (old_pot + old_int))
    
    #Acceptance condition
    if delta_s < 0: # always accepting if action lowered
        path1 = new_path1
        return  delta_s
    elif np.random.rand(1) < np.exp(-(delta_s/hbar)): #otherwise accept with PI probability.
        path1 = new_path1     
        return  delta_s
    else:
        return 0
    



def exchangeUpdate( Perm , path1 ,path2 ):
    '''
    Inputs:
    '''
    N = len(path1[0,:])


    randTimes= np.random.randint(1,N-1,2)
    randTimes= np.sort(randTimes)
    throw = True
    while (randTimes[0] == randTimes[1]) :
         randTimes= np.random.randint(1,N-1,2)
         randTimes= np.sort(randTimes)

    delta_s = 0
    
    for randomTime in randTimes:
        #Computing previous kinetic components at randomTime
        k1 = (.5/(tau))*np.matmul(np.transpose(m),(path1[:,randomTime]-path1[:,randomTime-1])**2)
        k2 = (.5/(tau))*np.matmul(np.transpose(m),(path2[:,randomTime]-path2[:,randomTime-1])**2)
        
        #Computing new kinetic components after exchange at randomTime
        nk1 = (.5/(tau))*np.matmul(np.transpose(m),(path2[:,randomTime]-path1[:,randomTime-1])**2)
        nk2 = (.5/(tau))*np.matmul(np.transpose(m),(path1[:,randomTime]-path2[:,randomTime-1])**2)
        #adding terms to action 
        delta_s = delta_s + (nk1 + nk2) - (k1 + k2)
        
    if delta_s < 0: # always accepting if action lowered
        TimeInterval = np.arange(randTimes[0],randTimes[1]) 
        tempPath = np.copy(path1[:,TimeInterval])
        path1[:,TimeInterval] = np.copy(path2[:,TimeInterval])
        path2[:,TimeInterval] = np.copy(tempPath)  
        return delta_s ,   Perm  #permutation variable changes signs

    elif np.random.rand(1) < np.exp(-(delta_s/hbar)): #otherwise accept with PI probability.
        TimeInterval = np.arange(randTimes[0],randTimes[1]) 
        tempPath = np.copy(path1[:,TimeInterval])
        path1[:,TimeInterval] = np.copy(path2[:,TimeInterval])
        path2[:,TimeInterval] = np.copy(tempPath) 
        return delta_s ,  Perm
        
    else:
        return 0 , Perm
        
    
def sweepUpdate(path1 ):
    '''
    Inputs:
    '''
    N = len(path1[0,:])
    shape = np.shape(path1)
    new_path1 = np.zeros(shape)

    ''' OLd sweep IMPlmentation '''
    #  # Creating 3N random numbers between [-h/2,h/2]
    rand_r1 = (np.random.rand(3,N-2)-.5)
    rand_r1[0,:] = h[0]*rand_r1[0,:]
    rand_r1[1,:] = h[1]*rand_r1[1,:]
    rand_r1[2,:] = h[2]*rand_r1[2,:]

    #Adding random numbers to create new path 
    new_path1[:,1:-1] = path1[:,1:-1] + rand_r1

    #Fix the path boundaries so that the change in kinetic energy is not considered
    endPath = np.array([0,-1])

    new_path1[:,endPath] = np.copy(path1[:,endPath])
         
    return new_path1

        # Creating 3 random numbers between [-h/2,h/2]
    # rand_r1 = (np.random.rand(3)-.5)

    # rand_r1[0] = h[0]*rand_r1[0]
    # rand_r1[1] = h[1]*rand_r1[1]
    # rand_r1[2] = h[2]*rand_r1[2]

    # new_path1[0,1:-1] = path1[0,1:-1] + rand_r1[0]
    # new_path1[1,1:-1]  = path1[1,1:-1] + rand_r1[1]
    # new_path1[2,1:-1]  = path1[2,1:-1] + rand_r1[2]

def antiSymSweep(path1,path2,GridFunc,min_array):
    '''Computes the action given two electron paths
       Inputs: Path1 , Path2: 3*N-arrays
               Sold - old action (float)
       outputs: newPath1, newPath2 (3N-arrays)
                delta_s - difference between new action and old action
    '''
    new_path1 = sweepUpdate(path1)
    new_path2 = path2
    kin_old = KineticEnergy(m,path1)
    kin_new = KineticEnergy(m,new_path1)

    old_pot = np.sum(ComputePotential(GridFunc, path1, min_array))
    new_pot = np.sum(ComputePotential(GridFunc, new_path1, min_array))

    old_int =  np.sum(ee(path1,path2))
    new_int =  np.sum(ee(new_path1,new_path2))
    delta_s = tau*( (kin_new+ new_pot + new_int) - (kin_old+ old_pot + old_int))

    if delta_s < 0: # always accepting if action lowered
        return new_path1 , new_path2 , delta_s
    elif np.random.rand(1) < np.exp(-(delta_s/hbar)): #otherwise accept with PI probability.
        return new_path1 , new_path2 , delta_s
    else:
        return path1 , path2, 0


def sweep(path1,path2,GridFunc,min_array):
    '''Computes the action given two electron paths
       Inputs: Path1 , Path2: 3*N-arrays
               Sold - old action (float)
       outputs: newPath1, newPath2 (3N-arrays)
                delta_s - difference between new action and old action
    '''
    if np.random.rand(1) > .5:
        new_path1 = sweepUpdate(path1)
        new_path2 = path2
        kin_old = KineticEnergy(m,path1)
        kin_new = KineticEnergy(m,new_path1)

        old_pot = np.sum(ComputePotential(GridFunc, path1, min_array))
        new_pot = np.sum(ComputePotential(GridFunc, new_path1, min_array))
    else:
        new_path1 = path1
        new_path2 = sweepUpdate(path2)
        kin_old = KineticEnergy(m,path2)
        kin_new = KineticEnergy(m,new_path2)

        old_pot = np.sum(ComputePotential(GridFunc, path2, min_array))
        new_pot = np.sum(ComputePotential(GridFunc, new_path2, min_array))

    old_int =  np.sum(ee(path1,path2))
    new_int =  np.sum(ee(new_path1,new_path2))
    #add action
    delta_s = tau*( (kin_new+ new_pot + new_int) - (kin_old+ old_pot + old_int))

    if delta_s < 0: # always accepting if action lowered
        return new_path1 , new_path2 , delta_s
    elif np.random.rand(1) < np.exp(-(delta_s/hbar)): #otherwise accept with PI probability.
        return new_path1 , new_path2 , delta_s
    else:
        return path1 , path2, 0
    


# -------------------------------------------PIMC-------------------------------------------------

def PIMC( NumRun
        , pathLength
        , NumMeasures
        , T_length
        , crossings
        , error
        , exchProb
        ,V_HRL):
    '''Execute path integral montecarlo
     Inputs: NumRun (int) - max num runs
             pathLenght - number of time beads
             numMeasures- number of runs between S-measurements
             T_length - length of the sub-paths used for sweep 
             #exchProb = 0  #This variables controls the probability of doing exchange update. Default 0 if exchanged update is not allowed

     Return: p1,p2 - Sampled paths
             S_arr - array of measurements

    '''
    
    #Create 2 random paths with definite number of crossings. Perm is the permutation number 
    p1 , p2 , Perm = generate_paths_N_crossings(pathLength, crossings)
    Perm = Perm

    #Create arrays that will be used to measure convergence
    S_arr = []
    Kin_arr = []
    Pot_arr = []
    
    '''The following lines are useful for debugging. They plot paths and 
    '''
    #Paths_1 = []
    #Paths_2 = []
#     fig, axarr = plt.subplots(2,1)
    CMprob = 0.2
    EndingParam = 10 #decides how many parameters to measure when expecting for convergence of the code 
    convergeNumber = 5
    i = 0
    stop = False
    while (i < NumRun+1) & (stop == False): #there is an stop signal that tells the code if desired
        i = i +1 
        randUpdate = np.random.rand(1)
        
        '''Execute update half of the cases'''
        if randUpdate  < exchProb:
            deltaS , Perm =  exchangeUpdate(Perm, p1 , p2)
        elif randUpdate < CMprob + exchProb:
            if np.random.rand(1) < 0.5:
                deltaS = CMupdate( p1
                                  ,p2
                                  ,V_HRL
                                  ,min_array2)
            else:
                 deltaS = CMupdate(p2
                                  ,p1
                                  ,V_HRL
                                  ,min_array2)

        else:

            '''Pic up a T_length-path randomly to sweep'''
            #Select a random time
            rTime = np.random.randint(pathLength)  

            #Roll the array to set rTime to the first position . Roll is used to handle periodic bound. coditions
            rolled_p1 = np.roll(p1,-rTime,(1))
            rolled_p2 = np.roll(p2,-rTime,(1))
            #select the first T_length time positions
            pstart1 = rolled_p1[:, :T_length] 
            pstart2 = rolled_p2[:, :T_length] 
            
            #make sweep only with the fist T components in the array
            if np.random.rand(1) < 0.5:
                pp1 , pp2 , deltaS= gaussianUpdate( pstart1
                                                    ,pstart2 
                                                    ,V_HRL
                                                    ,min_array)
            else:
                pp2 , pp1 , deltaS= gaussianUpdate( pstart2
                                                   ,pstart1 
                                                   ,V_HRL
                                                   ,min_array)

            # pp1 , pp2 , deltaS = sweep( pstart1
            #                           , pstart2 
            #                           , V_HRL
            #                           , min_array)
            
            #update the sweep result
            rolled_p1[:, :T_length] = np.copy(pp1)
            rolled_p2[:, :T_length] = np.copy(pp2)
            #Unroll vector
            p1 = np.roll(rolled_p1,rTime,(1))
            p2 = np.roll(rolled_p2,rTime,(1))

        '''Take measurements every 10000 operations'''
        if i%NumMeasures == 0:
        	#save action values to check convergence
            S_tot = find_Grid_Total_action(p1,p2,Perm,V_HRL, min_array2)

            S_arr.append(S_tot)
            Kin_arr.append(KineticTotalEnergy(m, p1,p2,Perm))
            Pot_arr.append(PotentialTotalEnergy(p1,p2,V_HRL, min_array2))

            #take measurements to decide if it is convergence has been reached 
            if i//NumMeasures > convergeNumber*EndingParam:
                stop = checkConvergence(i,S_arr, EndingParam, convergeNumber)
                    
            '''Plots paths'''
#             axarr[0].plot(range(pathLength) , p1[0,:],'-')
#             axarr[1].plot(range(pathLength) , p2[0,:],'-')
    return p1, p2, np.array(S_arr), Perm, np.array(Kin_arr) , np.array(Pot_arr)

def AntiSymPIMC( NumRun
        , pathLength
        , NumMeasures
        , T_length
        , crossings
        , error
        , exchProb
        ,V_HRL):
    '''Execute path integral montecarlo
     Inputs: NumRun (int) - max num runs
             pathLenght - number of time beads
             numMeasures- number of runs between S-measurements
             T_length - length of the sub-paths used for sweep 
     Return: p1,p2 - Sampled paths
             S_arr - array of measurements

    '''
    
    #Create 2 random paths with definite number of crossings. Perm is the permutation number 
    p1 , p2 , Perm = generate_paths_N_crossings(pathLength, crossings)
    Perm = Perm

    #Create arrays that will be used to measure convergence
    S_arr = []
    Kin_arr = []
    Pot_arr = []


    BigPath = np.concatenate((p1,p2), axis = 1)
    MirrorPath = np.zeros(np.shape(BigPath))
    MirrorPath[:,:pathLength] = BigPath[:,pathLength:]
    MirrorPath[:,pathLength:] = BigPath[:,:pathLength]

    '''The following lines are useful for debugging. They plot paths and 
    '''
    #Paths_1 = []
    #Paths_2 = []
    fig, axarr = plt.subplots(2,1)
    CMprob = 0.2
    EndingParam = 10 #decides how many parameters to measure when expecting for convergence of the code 
    i = 0
    stop = False
    while (i < NumRun+1) & (stop == False): #there is an stop signal that tells the code if desired
        i = i +1 
        randUpdate = np.random.rand(1)
        
        '''Execute update half of the cases'''
        if randUpdate  < exchProb:
            deltaS , Perm = exchangeUpdate(Perm, BigPath[:,:pathLength-1] , BigPath[:,pathLength+1:])
        elif randUpdate < CMprob + exchProb:  
            deltaS = CMupdate( BigPath
                              ,MirrorPath
                              ,V_HRL
                              ,min_array_BIG )
        else:

            '''Pic up a T_length-path randomly to sweep'''
            #Select a random time
            rTime = np.random.randint(2*pathLength)  

            #Roll the array to set rTime to the first position. Roll is used to handle anti-periodic bound. coditions
            rolled_p1 = np.roll(BigPath,-rTime,(1))
            rolled_p2 = np.roll(MirrorPath,-rTime,(1))
            #select the first T_length time positions
            pstart1 = rolled_p1[:, :T_length] 
            pstart2 = rolled_p2[:, :T_length] 
            
            #make sweep only with the fist T components in the array

            # pp1 , pp2 , deltaS = antiSymSweep(pstart1
            #                                  ,pstart2
            #                                  ,V_HRL
            #                                  ,min_array)
            pp1 , pp2 , deltaS= gaussianUpdate( pstart1
                                               ,pstart2 
                                               ,V_HRL
                                               ,min_array)
            #update the sweep result
            rolled_p1[:, :T_length] = np.copy(pp1)
            BigPath = np.roll(rolled_p1,rTime,(1))

        '''Take measurements every 10000 operations'''
        if i%NumMeasures == 0:
            p1 = BigPath[:,:pathLength]
            p2 = BigPath[:,pathLength:]
            #save action values to check convergence
            S_tot = find_Grid_Total_action(p1,p2,Perm,V_HRL, min_array2)

            S_arr.append(S_tot)
            Kin_arr.append(KineticTotalEnergy(m, p1,p2,Perm))
            Pot_arr.append(PotentialTotalEnergy(p1,p2,V_HRL, min_array2))

            #take measurements to decide if it is convergence has been reached 
            if i//NumMeasures > convergeNumber*EndingParam:
                stop = checkConvergence(i,S_arr, EndingParam, convergeNumber)

            '''Plots paths'''

            axarr[0].plot(range(pathLength) , p1[0,:],'-')
            axarr[1].plot(range(pathLength) , p2[0,:],'-')
            # if stop == False:
                # plt.clf()
                # plt.show()
                # plt.pause(0.01)
                
    return BigPath[:,:pathLength], BigPath[:,pathLength:], np.array(S_arr), Perm, np.array(Kin_arr) , np.array(Pot_arr)

def GetAproxValue(s1,EndingParam):
    '''
     Inputs:action
     Return:

    '''
    Std = np.std(s1)
    Mean = np.mean(s1)
    upVal = Mean + Std/(np.sqrt(EndingParam))
    minVal = Mean - Std/(np.sqrt(EndingParam))
    return upVal , minVal

def checkConvergence(i,S_arr, EndingParam, convergeNumber):
    ''' Stops the code if it has reached convergence 
     Inputs:
     Return:

    '''
    s1 = np.array(S_arr[-EndingParam:])
    up = [] ; mini = []
    for j in range(convergeNumber):
        s1 = np.array(S_arr[-(j+1)*EndingParam:-j*EndingParam-1])
        upVal , minVal = GetAproxValue(s1,EndingParam)
        up.append(upVal)
        mini.append(minVal)

    print(i
        ,"Iterations -V Mean Energy: "
        , np.mean(S_arr[-10:])/pathLength
        , '+/-'
        ,np.std(S_arr[-10:])/(pathLength*np.sqrt(10))
        , 'ueV')
    
    stop = False
    # if ((relativeError < error) & (lowMax > upMin)):
    if (np.min(up) > np.max(mini)):
        stop = True
        print("Energy", s1[-1]/pathLength, '--- Stopped at i = ', i)
    return stop



'''------------------Plotting functions-useful for debugging---------------------'''
def PlotConvergence(S_arr, Kin_arr, Pot_arr):
    ''' Plot convergence
     Inputs: S_arr - action array
     Kin_arr - kinetic array
     Pot_array -kinetic array
     Return: '''
    fig, ax = plt.subplots(1,1)
    NumDat = 200
    size_S = len(S_arr[-NumDat:])

    ax.plot(range(size_S),(S_arr[-NumDat:]/pathLength),'r')
    ax.plot(range(size_S),tau*Kin_arr[-NumDat:]/pathLength,'b.')
    ax.plot(range(size_S),tau*Pot_arr[-NumDat:]/pathLength,'k.')

    #Axis Labels
    ax.set_ylabel('$S$')
    ax.set_xlabel('Iterations ($\\times'+ repr(NumMeasures) +'$)')

def PlotPaths(p1,p2):
    ''' Plot final obtained paths
     Inputs: S_arr - action array
     Kin_arr - kinetic array
     Pot_array -kinetic array
     Return:
      '''
    pathLength = len(p1[0,:])
    fig, ax = plt.subplots(1,1) ; p1 = np.roll(p1,-60,(1)) ; p2 = np.roll(p2,-60,(1))

    ax.plot( p1[0,:], np.array(range(pathLength))*tau ,'b.')
    ax.plot( p2[0,:], np.array(range(pathLength))*tau ,'r.')

    ax.set_ylabel('$\\tau*(ins)$')
    ax.set_xlabel('$x (nm)$')


'''-----------------------------------------Execute PIMC-----------------------------------------'''

## -----------------------List of Constants------------------
# """Parameters, units: ueV, nm, ns"""

print('Setting initial constants')
eec = 1.23*(10**5)
print('Electron-electron coupling constant (eec) =', eec, 'ueV')
me = 5.686 * 10 **(-6) # ueV ns^2/nm^2 - electron mass
print('me = ', repr(np.round(me,9)), 'ueV*ns^2/nm^2' )
meff= 0.2 * me #effective electron mass in Silicon
ml = 0.98 * me #effective mass in longitudinal k-direction
mt = 0.19 * me #effective mass in transverse k-direction
mx = mt #effective mass in x-direction
my = mt #effective mass in y-direction
mz = ml #effective mass in z-direction
m = np.array([mx,my,mz])
print('Effective mass vector (m) = ', m/me,'*me' )
pot_step = 3 * 10**6 #ueV - Si/SiO2 interface
print('Step voltage height (pot_step) = ', pot_step,' $ueV$' )

hbar = 0.6582 # ueV * ns
a = 0.543 #nm - lattice constant in silicon

w = 4000/hbar
q = 10**6 # ueV
lamda = 0.5 * hbar**2 / me
r_0 = np.sqrt(hbar / (me*w))

# V_field = V_hrl

print('Lattice constant (a)=', a, 'nm')






'''-----------------------------------Define grid and HRL potential-----------------------------------'''
print('Creating Grid Potentials' )

V_HRL_Grid = np.load('V_HRL_Grid.npz')

XS = V_HRL_Grid['XS']
YS = V_HRL_Grid['YS']
ZS = V_HRL_Grid['ZS']
dXS = XS[1] - XS[0]
dZS = ZS[1] - ZS[0]

Grid = V_HRL_Grid['Grid']
V_HRL= V_HRL_Grid['V_HRL']
Z_max = np.argmax(ZS)

# XS, YS , ZS, Grid =  RefineGrid(xs,ys,zs)
# V_HRL = ComputeGridFunction(V_hrl, Grid, XS, YS , ZS,False)

print('Setting PIMC parameters')

# MaxRun = 5*10**6 #Maximum number of runs --- typically 10**6 for a 1000 path
# pathLength = 5*10**3 #Path Length in time slices
# #Meaurements are executed in order to probe if the code has already converged
# NumMeasures = 10**4 #Number of times between measurements . Two orders of magnitud less than MaxRun
# T_length = 50 # lenght of the time slice moved at each sweep update. 
#               #T_lenght should have 2 orders of magnitude less than pathLenght for better efficiency
# crossings = 10 # approximate number of crossings  of the sampled path
# error = 0.01 #when std/mean < 0.005 in PIMC, the code stops running
# tau = 2*10 ** (-6) #nsdu
# h = np.array([2,1,.1])


Inputs = np.loadtxt('input.dat', delimiter = '\t' )
MaxRun =  int(Inputs[0])
pathLength =  int(Inputs[1])
NumMeasures =  int(Inputs[2])
T_length =  int(Inputs[3])
crossings =  int(Inputs[4])
convergeNumber =  int(Inputs[5])
tau =  Inputs[6] 
tau2 = tau**2
h = np.array([ Inputs[7], Inputs[8], Inputs[9]])
exchProb = Inputs[10]

print('tau = ', tau, '(ns)')
print('Displacement vector (h) =', h , 'nm')

min_array = CreateMinArray(T_length)
min_array2 = CreateMinArray(pathLength)
min_array_BIG = CreateMinArray(2*pathLength)
print('Executing PIMC')
'''Executing PIMC on multiple machines'''
start_time = time.time()

if (crossings)%2 == 1:
    p1,p2,S_arr,Perm, Kin_arr,Pot_arr= AntiSymPIMC(MaxRun
                    						,pathLength
                    						,NumMeasures
                    						,T_length
                    						,crossings 
                    						,convergeNumber
                    						,exchProb
                                            ,V_HRL)	
else:
    p1,p2,S_arr,Perm, Kin_arr,Pot_arr= PIMC(MaxRun
                                        ,pathLength
                                        ,NumMeasures
                                        ,T_length
                                        ,crossings 
                                        ,convergeNumber
                                        ,exchProb
                                        ,V_HRL) 


print("Parity ----", Perm)
print("Value of the mean. ----"
	, np.mean(S_arr[-10:])/pathLength
	, '+/-'
	,np.std(S_arr[-10:])/(pathLength*np.sqrt(10))
	, 'ueV')
print("--- %s seconds ---" % (time.time() - start_time))


np.savez('path' , p1=p1 , p2=p2 , S_arr=S_arr , Perm=Perm )

#Plot Results
PlotConvergence(S_arr,Kin_arr, Pot_arr)
PlotPaths(p1,p2)



from scipy.interpolate import RegularGridInterpolator as rgi
HRL_Potential = 'old_UNSW4_1st withBG TEST ephiQWM UNSW4_1st structure 2nm 800x800 - 10-4-12-4.npz'
# HRL_Potential = 'UNSW4_1st withBG TEST ephiQWM for VX1_1V UNSW4_1st structure 2nm 800x800 - 10-4-12-4.npz'
'''Define grid and HRL potential'''
xs = np.load(HRL_Potential)['xs']
ys = np.load(HRL_Potential)['ys']
zs = np.load(HRL_Potential)['zs']
ephi = 10**6 * np.load(HRL_Potential)['ephi']

interpolate = rgi(points = (xs,ys,zs), values = ephi, bounds_error = False, method = 'linear') #options 'linear' , 'nearest'


# Heterostructure interface potential (z-direction) leading to airy functions as wave function.
def V_hrl(path):
    return interpolate(path.T) + V_step(path[2,:])

fig, ax = plt.subplots(1,1)
T = range(len(p1[2,:]))
# ax.plot(p1[0,:], V_hrl(p1)+ V_hrl(p2),'rv')
ax.plot(p1[0,:],ComputePotential(V_HRL, p1, min_array2) ,'b^')
ax.plot(p2[0,:],ComputePotential(V_HRL, p2, min_array2) ,'rv')
# K_1 = (.5/tau2)*np.matmul(np.transpose(m),(timeDerivative(p1))**2)

# K_2 = (.5/tau2)*np.matmul(np.transpose(m),(timeDerivative(p2))**2)
# ax.plot(T[:-1], K_1 ,'k-')
# ax.plot(T[:-1],K_2, 'y-')
ax.plot(p1[0,:], ee(p1,p2),'go')

ax.set_xlabel('Z(nm)')
ax.set_ylabel('$V_z(\\mu eV)$')


plt.show()




