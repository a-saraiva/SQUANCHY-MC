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
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.stats import linregress
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
import time

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
    plt.rcParams['figure.figsize'] =  6.8, 8.3
# %%
SetPlotParams()


# -------------------------------- Methods for path generation, crossings and staying----------------------------------------

def generate_paths_N_crossings(N, crossings):
    '''Generates random paths of 2 electrons with a defined number of crossings'''
    pathx1 = np.zeros(N) 
    pathx2 = np.zeros(N) 
    
    arr = np.arange(N)
    np.random.shuffle(arr)
    List= np.sort(arr[:crossings])
    CrossingPoints = np.zeros(crossings+2)
    CrossingPoints[0] = 0
    CrossingPoints[-1] = N
    CrossingPoints[1:-1] = List
    CrossingPoints = CrossingPoints.astype(int)

    for i in range(crossings+1):
        if i%2 == 0:
            pathx1[CrossingPoints[i]:CrossingPoints[i+1]] = np.random.uniform(-60,-20,CrossingPoints[i+1]-CrossingPoints[i])
            pathx2[CrossingPoints[i]:CrossingPoints[i+1]] = np.random.uniform(-45,5,CrossingPoints[i+1]-CrossingPoints[i])
        else:
            pathx1[CrossingPoints[i]:CrossingPoints[i+1]] = np.random.uniform(-45,5,CrossingPoints[i+1]-CrossingPoints[i])
            pathx2[CrossingPoints[i]:CrossingPoints[i+1]] = np.random.uniform(-60,-20,CrossingPoints[i+1]-CrossingPoints[i])

        
    pathy1 = np.random.uniform(-12,12,N)
    pathz1 = np.random.uniform(-2,-0,N)
    
    pathy2 = np.random.uniform(-12,12,N)
    pathz2 = np.random.uniform(-2,-0,N)
    
    path1 = np.array([pathx1,pathy1,pathz1])
    path2 = np.array([pathx2,pathy2,pathz2])
    Perm = (-1)**(crossings)
    
    return path1 , path2, Perm




# ##------------------------------------- Vectorized Potentials-------------------------------------------------------------


    '''From now on all potentials are modified to include vectorization'''

def V_step(z_path):
    zero_arr = np.zeros((3,len(z_path)))
    zero_arr[2,:] = z_path
    return(V / (np.exp(-(4/a)*z_path) + 1) + interpolate(zero_arr.T))

def dist(path1,path2):
    minus = path1-path2
    return np.sqrt(minus[0,:]**2 + minus[1,:]**2 + minus[2,:]**2)
#     return np.sqrt(np.matmul(minus.T,minus))

def ee(path1,path2):
    return(eec/dist(path1,path2))

def V_hrl(path):
    return interpolate(path.T) + V_step(path[2,:])


#---------------------------------------------------GRID - Potentials--------------------------------------------------

#Define Grid and minimum array
# pathLenght = 100
# XS, YS , ZS, Grid =  RefineGrid(xs,ys,zs)
# Z_max = np.argmax(ZS)
def CreateMinArray(pathLenght):
    '''
    Inputs:
    Return:
    '''
    mins= np.array([min(XS),min(YS),min(ZS)])
    min_array = np.zeros((3,pathLenght))
    min_array[0,:] +=  mins[0].astype(int)
    min_array[1,:] +=  mins[1].astype(int)
    min_array[2,:] +=  (10*mins[2]).astype(int)
    min_array = min_array.astype(int)
    return min_array

# min_array = CreateMinArray(pathLenght)

def ComputePotential(Grid_Func, path, min_array):
    '''
    Computes the potential interpolating the path over an integer look up table
    Inputs: Grid_Func -- Look-up table of already computed potential over the grid
            path - N-array 
    Return: The Grid 
    '''
    #Get indices of the paths in XS,YS,ZS coordinates
    argsXY = np.round(path[:-1,:]).astype(int) - min_array[:-1,:]
    argZ = np.round(path[-1,:]*10).astype(int) - min_array[-1,:]
    argZ[argZ > Z_max] = Z_max
#     print(argZ)
    return Grid_Func[argsXY[0],argsXY[1],argZ]

def KineticEnergy(MassVec,path):
    '''Computes kinetic energy of path of an electron path
       inputs: MassVec(3*1-array)
               path(3*N-array)
       return: Energy vector(N-array)
    '''
    return (.5/(tau**2))*np.matmul(np.transpose(MassVec),(path[:,1:]-path[:,:-1])**2)

def find_Grid_action(path1,path2,GridFunc, minArray):
    '''Computes the action given two electron paths
       inputs: Path1 , Path2: 3*N-arrays
       outputs: Action(real)
    '''
    #Interaction
    V_interaction =  np.sum(ee(path1,path2))
    
    #Kinetic energy
    s1 = tau*KineticEnergy(m,path1)
    s2 = tau*KineticEnergy(m,path2)
    
    #Potential energy
    path1Potential = ComputePotential(GridFunc, path1,minArray)
    path2Potential = ComputePotential(GridFunc, path2,minArray)
    
    #Action
    s1 = np.sum(s1) + tau * np.sum(path1Potential)# + V_interaction
    s2 = np.sum(s2) + tau * np.sum(path2Potential)# + V_interaction
    
    #sum up
    s = s1 + s2  + tau * V_interaction #+ 2 * np.log(2 * np.pi * tau / meff)
    return s



# def find_action(path1,path2):
#     '''Computes the action given two electron paths
#        inputs: Path1 , Path2: 3*N-arrays
#        outputs: Action(real)
#     '''
#     V_field = V_hrl
#     interaction = 1
#     V_interaction = interaction * np.sum(ee(path1,path2))
    
#     s1 = tau*KineticEnergy(m,path1)
#     s2 = tau*KineticEnergy(m,path2)
    
#     s1 = np.sum(s1) + tau * np.sum(V_field(path1))# + V_interaction
#     s2 = np.sum(s2) + tau * np.sum(V_field(path2))# + V_interaction
    
#     s = s1 + s2 + 2 * np.log(2 * np.pi * tau / meff) + tau * V_interaction
#     return s

    
def KineticTotalEnergy(MassVec,path1,path2, Perm):
    '''Computes kinetic energy of the total periodic path of an electron path
       inputs: MassVec(3*1-array)
               path(3*N-array)
       return: Energy vector(N-array)
    '''
    
    if Perm == 1:
        Rpath1 = np.roll(path1,-1,(1))
        k1 = (.5/(tau**2))*np.matmul(np.transpose(MassVec),(Rpath1-path1)**2)
        
        Rpath2 = np.roll(path2,-1,(1))
        k2 = (.5/(tau**2))*np.matmul(np.transpose(MassVec),(Rpath2-path2)**2)
        return k1 + k2
    
    else:
        BigPath = np.concatenate((path1,path2), axis = 1)
        RBigPath = np.roll(BigPath,-1,(1))
        return (.5/(tau**2))*np.matmul(np.transpose(MassVec),(RBigPath-BigPath)**2)


def find_Grid_Total_action(path1,path2,Perm,GridFunc, minArray):
    '''Computes the action given two electron paths
       inputs: Path1 , Path2: 3*N-arrays
       outputs: Action(real)
    '''
    #Interaction
    V_interaction =  np.sum(ee(path1,path2))
    
    #Kinetic energy
    s1 = tau*KineticTotalEnergy(m,path1,path2, Perm)
    
    #Potential energy
    path1Potential = ComputePotential(GridFunc, path1,minArray)
    path2Potential = ComputePotential(GridFunc, path2,minArray)
    
    #Action
    s1 = np.sum(s1) + tau * np.sum(path1Potential) + tau * np.sum(path2Potential)# + V_interaction+ V_interaction
    
    #sum up
    s = s1  + tau * V_interaction #+ 2 * np.log(2 * np.pi * tau / meff)
    return s
    

'''------------------------------------- Updates --------------------------------'''

"""
One sweep consists of trying N times changing a random element in the array path (size of this change depends on h),
each time accepting/refusing according to Metropolis. 
"""

def sweepUpdate(path1 , path2):
    '''
    Inputs:
    '''
    N = len(path1[0,:])
    if np.random.rand(1) > .5:
         # Creating 3N random numbers between [-h/2,h/2]
        rand_r1 = (np.random.rand(3,N)-.5)
        rand_r1[0,:] = h[0]*rand_r1[0,:]
        rand_r1[1,:] = h[1]*rand_r1[1,:]
        rand_r1[2,:] = h[2]*rand_r1[2,:]
        
        rand_r2 = np.zeros((3,N))
    else:
         # Creating 3N random numbers between [-h/2,h/2]
        rand_r2 = (np.random.rand(3,N)-.5)
        rand_r2[0,:] = h[0]*rand_r2[0,:]
        rand_r2[1,:] = h[1]*rand_r2[1,:]
        rand_r2[2,:] = h[2]*rand_r2[2,:]
        
        rand_r1 = np.zeros((3,N))

    #Adding random numbers to create new path 
    new_path1 = path1 + rand_r1
    new_path2 = path2 + rand_r2
            
    return new_path1, new_path2

def exchangeUpdate(Perm ,path1 , path2):
    '''
    Inputs:
    '''
    N = len(path1[0,:])
    randomTime = np.random.randint(N)
    #Computing previous kinetic components at randomTime
    k1 = (.5/(tau))*np.matmul(np.transpose(m),(path1[:,randomTime]-path1[:,randomTime-1])**2)
    k2 = (.5/(tau))*np.matmul(np.transpose(m),(path2[:,randomTime]-path2[:,randomTime-1])**2)
    
    #Computing new kinetic components after exchange at randomTime
    nk1 = (.5/(tau))*np.matmul(np.transpose(m),(path2[:,randomTime]-path1[:,randomTime-1])**2)
    nk2 = (.5/(tau))*np.matmul(np.transpose(m),(path1[:,randomTime]-path2[:,randomTime-1])**2)
    
    delta_s = nk1 + nk2 - k1 - k2
    
    if delta_s < 0: # always accepting if action lowered
        tempPath = np.copy(path1[:,randomTime:])
        path1[:,randomTime:] = np.copy(path2[:,randomTime:])
        path2[:,randomTime:] = np.copy(tempPath)  
        
#         print('Minor action' , np.exp(-(delta_s/hbar) ))
#         print('Sweep accepted -', delta_s )
        return delta_s ,  (-1) * Perm  #permutation variable changes signs

    elif np.random.rand(1) < np.exp(-(delta_s/hbar)): #otherwise accept with PI probability.
#         print('increased with probability ' , np.exp(-(delta_s/hbar) ))
        tempPath = np.copy(path1[:,randomTime:])
        path1[:,randomTime:] = np.copy(path2[:,randomTime:])
        path2[:,randomTime:] = np.copy(tempPath) 
#         print('Sweep accepted +', delta_s )
        return delta_s , (-1) * Perm
        
    else:
#         print('Stable with probability ' , np.exp(-(delta_s/hbar) ))      
        return 0 , Perm
    


def sweep(Perm ,path1,path2,GridFunc,min_array,circledPath):
    '''Computes the action given two electron paths
       Inputs: Path1 , Path2: 3*N-arrays
               Sold - old action (float)
       outputs: newPath1, newPath2 (3N-arrays)
                delta_s - difference between new action and old action
    '''
    
    #Old ------ Computing the action
#     So = find_action(path1,path2)   #--- not necessary if the action is saved
#     Sn = find_action(new_path1,new_path2)
    
    new_path1, new_path2 = sweepUpdate(path1 , path2)
    
    So =  find_Grid_action(path1,path2,GridFunc,min_array) #--- not necessary if the action is saved
    Sn =  find_Grid_action(new_path1,new_path2,GridFunc,min_array)
    
    
    #In odd permutations if the paths are crossed and we are sweeping at the boundary, we need to change the kinetic action
    if Perm == -1 & circledPath > -1 :
    #         print(circledPath)
            #substracting kinetic action connecting the same path at the boundary
            So = So - (.5/(tau))*np.matmul(np.transpose(m),(path1[:,circledPath+1]-path1[:,circledPath])**2)
            So = So - (.5/(tau))*np.matmul(np.transpose(m),(path2[:,circledPath+1]-path2[:,circledPath])**2)

            Sn = Sn - (.5/(tau))*np.matmul(np.transpose(m),(new_path1[:,circledPath+1]-new_path1[:,circledPath])**2)
            Sn = Sn - (.5/(tau))*np.matmul(np.transpose(m),(new_path2[:,circledPath+1]-new_path2[:,circledPath])**2)
            
            #adding kinetic action connecting crossing paths at the boundary
            So = So + (.5/(tau))*np.matmul(np.transpose(m),(path1[:,circledPath+1]-path2[:,circledPath])**2)
            So = So + (.5/(tau))*np.matmul(np.transpose(m),(path2[:,circledPath+1]-path1[:,circledPath])**2)

            Sn = Sn + (.5/(tau))*np.matmul(np.transpose(m),(new_path1[:,circledPath+1]-new_path2[:,circledPath])**2)
            Sn = Sn + (.5/(tau))*np.matmul(np.transpose(m),(new_path2[:,circledPath+1]-new_path1[:,circledPath])**2)
        

    delta_s = Sn - So 
#     print(So, Sn , delta_s)          
               
    if delta_s < 0: # always accepting if action lowered
#         print('Minor action' , np.exp(-(delta_s/hbar) ))
        return new_path1 , new_path2 , delta_s
    elif np.random.rand(1) < np.exp(-(delta_s/hbar)): #otherwise accept with PI probability.
#         print('increased with probability ' , np.exp(-(delta_s/hbar) ))
        return new_path1 , new_path2 , delta_s
        
    else:
#         print('Stable with probability ' , np.exp(-(delta_s/hbar) ))      
        return path1 , path2, 0


# -------------------------------------------PIMC-------------------------------------------------

def PIMC(NumRun, pathLength, NumMeasures,T_length, crossings, error):
    '''Excute path integral montecarlo
     Inputs: NumRun (int) - max num runs
             pathLenght - number of time beads
             numMeasures- number of runs between S-measurements
             T_length - lenght of the subpaths used for sweep 
     Return: p1,p2 - Sampled paths
             S_arr - array of measurements

    '''
    
    #Create 2 random paths with definite number of crossings. Perm is the permutation number 
    p1 , p2 , Perm = generate_paths_N_crossings(pathLength, crossings)
    S_arr = []
    
    '''The following lines are useful for debugging. They plot paths and 
    '''
    #Paths_1 = []
    #Paths_2 = []
#     fig, axarr = plt.subplots(2,1)

    circledPath = -1 #This variable is used to handle sweeps at the path boundaries with periodic conditions
    exchProb = 0 #This variables controls the probability of doing exchange update. Default 0 if exchanged update is not allowed
    
    i = 0
    stop = False
    while (i < NumRun+1) & (stop == False): #there is an stop signal that tells the code if desired
        i = i +1 
        
        '''Execute update half of the cases'''
        if np.random.rand(1) < exchProb:

            deltaS , Perm = exchangeUpdate(Perm, p1 , p2)

        else:

            '''Pic up a T_length-path randomly to sweep'''
            #Select a random time
            rTime = np.random.randint(pathLength)  
            
            
                
                
            #Roll the array to set rTime to the first position
            rolled_p1 = np.roll(p1,-rTime,(1))
            rolled_p2 = np.roll(p2,-rTime,(1))
            #select the first T_length time positions
            pstart1 = rolled_p1[:, :T_length] 
            pstart2 = rolled_p2[:, :T_length] 
            
            #Need to find the bead to eliminate from last position and break imposed periodic boundary conditions
            if (T_length + rTime - 1) > (pathLength):
                circledPath = pathLength - rTime -1
            
            #make sweep only with the fist T components in the array
            pp1 , pp2 , deltaS = sweep(Perm, pstart1, pstart2 ,V_HRL, min_array, circledPath)
            
            #Restore circledPath at original value
            circledPath = -1
            
            #update the sweep result
            rolled_p1[:, :T_length] = np.copy(pp1)
            rolled_p2[:, :T_length] = np.copy(pp2)
            #Unroll vector
            p1 = np.copy(np.roll(rolled_p1,rTime,(1)))
            p2 = np.copy(np.roll(rolled_p2,rTime,(1)))


#         '''Saved last 10000 paths - debugging'''
#         #if i> (NumRun - SavedPaths):
#         #    if  np.random.rand(1) < 0.2:
#         #        Paths_1.append(p1)
#         #        Paths_2.append(p2)
            
        '''Take measurements every 10000 operations'''
        if i%NumMeasures == 0:
            S_tot = find_Grid_Total_action(p1,p2,Perm,V_HRL, min_array2)
                
#             S_tot = find_Grid_action(p1,p2,V_HRL,min_array2)
            S_arr.append(S_tot)
            
            if i//NumMeasures > 20:
                s = np.array(S_arr[-20:])
                STD = np.std(s)
                MEAN = np.mean(s)
                
#                 if ((exchProb > 0) & (STD/MEAN < 0.05) ) : # If exchange update is allowed, this part cancels it at will at some point
#                     exchProb = 0
#                     print('No more exchange at', i)
                
                if (STD/MEAN < error):
                    stop = True
                    print('Stopped at i = ', i)
                    
            '''Plots paths'''
#             axarr[0].plot(range(pathLength) , p1[0,:],'-')
#             axarr[1].plot(range(pathLength) , p2[0,:],'-')
    return p1, p2, S_arr , Perm


'''------------------Plotting functions-useful for debugging---------------------'''
def PlotConvergence(S_arr):
	''' Plot convergence
	 Inputs: 
	 Return:

	'''
	fig, ax = plt.subplots(1,1)
	ax.semilogy(range(len(S_arr)),S_arr)
	#Axis Labels
	ax.set_ylabel('$S$')
	ax.set_xlabel('Iterations ($\\times'+ repr(NumMeasures) +'$)')



def PlotPaths(p1,p2):
	''' Plot electron paths in x
	 Inputs:
	 Return:

	'''
	pathLength = len(p1[0,:])
	fig, ax = plt.subplots(1,1)

	ax.plot( p1[0,:], np.array(range(pathLength))*tau ,'b.')
	ax.plot( p2[0,:], np.array(range(pathLength))*tau ,'r.')

	ax.set_ylabel('$\\tau*(ins)$')
	ax.set_xlabel('$x (nm)$')


'''-----------------------------------------Execute PIMC-----------------------------------------'''

## -----------------------List of Constants------------------
"""Parameters, units: ueV, nm, ns"""
print('Setting initial constants')

eec = 1.23*10**5
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
pot_step = 3 * 10**6 #ueV - Si/SiO2 interfac
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
error =  Inputs[5]
tau =  Inputs[6] 
h = np.array([ Inputs[7], Inputs[8], Inputs[9]])

print('tau = ', tau, '(ns)')
print('Displacement vector (h) =', h , 'nm')

min_array = CreateMinArray(T_length)
min_array2 = CreateMinArray(pathLength)

print('Executing PIMC')
'''Executing PIMC on multiple machines'''
start_time = time.time()
p1,p2,S_arr,Perm = PIMC(MaxRun
						,pathLength
						,NumMeasures
						,T_length
						,crossings 
						,error)

print("Parity ----", Perm)
print("Value of the mean. ----"
	, np.mean(S_arr[-10:])/pathLength
	, '+/-'
	,np.std(S_arr[-10:])/pathLength
	, 'ueV')
print("--- %s seconds ---" % (time.time() - start_time))

np.savez('path' , p1=p1 , p2=p2 , S_arr=S_arr , Perm=Perm )

#Plot Results
# PlotConvergence(S_arr)
# PlotPaths(p1,p2)

# plt.show()




