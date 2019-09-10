#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:24:58 2019

@author: schlattnerfrederic
"""

"""
IMPORTANT:
    
Many things have to be changed in the code for different scenarios:
    - MC parameters obviously: Temp, Slices, MCsteps
    For following things, you need to add/remove #, or ''' in the code:
    - Potential has to be changed at "Harmonic Oscillator" function.
    - In the PIMC main function, the updating algo has to be choosen:
      either Staging+COM, or Metropolis.
    - Dispersion E(k) has to be changed.
    - Make sure that integration length (h_step parameter) is the right one depending on dispersion
      namely 2pi/a if BZ bounderies for cosine dispersion and longer integration for free propagator (depending on tau)
    - Make sure that if cosine E(k) is used, then updating algo is MetropolisMoveCA,
      which uses an updating scheme based on propagators and not euclidean action.
      
"""


#from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import math
from fast_histogram import histogram1d
from copy import deepcopy
from scipy.special import airy

"""
Parameters of simulation:
"""

hbar=1
k_B=1
w=1
m=1

"""
Parameters of 1D silicon dispersion:
"""

epsilon=7.76856 #eff Hartree
u=3.75 #Ha
v=3.36061 #Ha
a = round(0.543/0.6757,4) #Lattice parameter Si in Bohrrad

"""
Parameters for potential triangle
"""

bar=16.5
echarge=1
slope=0.1

"""
Calculation of Integrals
"""

T = 0.1  # temperature in Kelvin  
lam = hbar**2/(2*m)
   

numParticles = 1    
numTimeSlices = 20
numMCSteps = 100000
tau = 1.0/(T*k_B*numTimeSlices)
binSize = 100


lat=np.sqrt(hbar/(w*m))/200
#lat=a/2
#lat=0.5

pos=lat*np.linspace(0,3000,3001,endpoint=True)
int_val=np.zeros(len(pos))
int_val_E=np.zeros(len(pos))

chopping=500
#h_step=((2*math.pi/a)/chopping)
h_step=(60/chopping)

"""Two different dispersions"""
#def E(k):
#    return(epsilon-2*u*math.cos(k*a/4)+2*v*math.cos(k*a/2))

def E(k):
    return((hbar*k)**2/(2*m))

def weight(pos,k):
    return(math.cos(pos*k)*math.exp(-tau*E(k)/hbar))

"""Use plot_A function to determine integration bounderies if free propagator is calculated"""

def plot_A():
    l=np.linspace(0,50,200)
    E1= [weight(0.32,i) for i in l]
    print('tau=', tau)
    plt.figure
    plt.plot(l,E1)
    plt.xlabel('position')
    plt.ylabel('Airy')
    plt.show
# %%
    
def weight_E(pos,k):
    return(E(k)*math.cos(pos*k)*math.exp(-tau*E(k)/hbar))

for i in range(0,len(pos)):
    integral=0
    integral_E=0
    for j in range(0,chopping):
        integral+=h_step*0.5*(weight(pos[i],(j+1)*h_step)+weight(pos[i],j*h_step))
        integral_E+=h_step*0.5*(weight_E(pos[i],(j+1)*h_step)+weight_E(pos[i],j*h_step))
    int_val[i]=2*integral/(2*math.pi)
    int_val_E[i]=2*integral_E/(2*math.pi)
    #int_val[i]=integral
    #int_val_E[i]=integral_E

# ------------------------------------------------------------------------------------------- 
def SHOEnergyExact(T):
    '''The exact SHO energy when \hbar \omega/ k_B = 1.''' 
    return 0.5*hbar*w/np.tanh(0.5*hbar*w/(k_B*T))

# %%------------------------------------------------------------------------------------------- 
def HarmonicOscillator(R):
    '''Simple harmonic oscillator potential with m = 1 and \omega = 1.'''
    #return 0.5*m*w**2*np.dot(R,R);
    return(bar/(math.exp(10*R)+1)+slope*R)
    #if R<0:
    #   return(bar)
    #else: return(slope*R)
    
def Airy(R):
    E1=(hbar**2/(2*m))**(1/3)*(9*math.pi*echarge*slope/8)**(2/3)
    return airy((2*m/(hbar*echarge*slope)**2)**(1/3)*(echarge*slope*R-(E1)))[0]
  

def plot_Airy():
    
    l=np.linspace(0,15,50)
    E1= [Airy(i) for i in l]
    #print(slope)
    plt.figure
    plt.plot(l,E1)
    plt.xlabel('position')
    plt.ylabel('Airy')
    plt.show

def plot_CV():
    
    l=np.linspace(0,5,50)
    E1= [SHOEnergyExact(i) for i in l]
    #print(slope)
    plt.figure
    plt.plot(l,E1)
    plt.xlabel('temp')
    plt.ylabel('EnergyHO')
    plt.show


# %%------------------------------------------------------------------------------------------- 
class Paths:
    '''The set of worldlines, action and estimators.'''
    def __init__(self,beads,tau,lam):
        self.tau = tau
        self.lam = lam
        self.beads = np.copy(beads)
        self.numTimeSlices = len(beads)
        self.numParticles = len(beads[0])

    def SetPotential(self,externalPotentialFunction):
        '''The potential function. '''
        self.VextHelper = externalPotentialFunction

    def Vext(self,R):
        '''The external potential energy.'''
        return self.VextHelper(R)

    def PotentialAction(self,tslice):
        '''The potential action.'''
        pot = 0.0
        for ptcl in range(self.numParticles):
            pot += self.Vext(self.beads[tslice,ptcl]) 
        return self.tau*pot
    
    def KineticAction(self,tslice):
        kin=0.0
        norm=0.5*m/self.tau
        
        tslicep1 = (tslice + 1) % self.numTimeSlices
        for ptcl in range(self.numParticles):
            delR = self.beads[tslicep1,ptcl] - self.beads[tslice,ptcl]
            kin = kin + np.dot(delR,delR)
        return norm*kin
    
    def KineticActionCA(self,tslice):
        kin=1
        tslicep1 = (tslice + 1) % self.numTimeSlices
        for ptcl in range(self.numParticles):
            delR = self.beads[tslicep1,ptcl] - self.beads[tslice,ptcl]
            kin=kin*int_val[abs(int(round(delR/lat)))]
        return kin
        
    
    

    def KineticEnergy(self):
        '''The thermodynamic kinetic energy estimator.'''
        tot = 0.0
        norm = 1.0/(4.0*self.lam*self.tau*self.tau)
        for tslice in range(self.numTimeSlices):
            tslicep1 = (tslice + 1) % self.numTimeSlices
            for ptcl in range(self.numParticles):
                delR = self.beads[tslicep1,ptcl] - self.beads[tslice,ptcl]
                tot = tot - norm*np.dot(delR,delR)

        KE = 0.5*self.numParticles/(self.tau) + tot*hbar/(self.numTimeSlices)
        
        return KE
    
    def KineticEnergyCA(self):
        '''The thermodynamic kinetic energy estimator.'''
        
        tot = 0.0
        for tslice in range(self.numTimeSlices):
            tslicep1 = (tslice + 1) % self.numTimeSlices
            for ptcl in range(self.numParticles):
                delR = self.beads[tslicep1,ptcl] - self.beads[tslice,ptcl]
                
                
                tot = tot+(int_val_E[abs(int(round(delR/lat)))]/int_val[abs(int(round(delR/lat)))])
               
        
        KE =tot*hbar/(self.numTimeSlices)
        return KE
    

    def PotentialEnergy(self):
        '''The operator potential energy estimator.'''
        PE = 0.0
        for tslice in range(self.numTimeSlices):
            for ptcl in range(self.numParticles):
                R = self.beads[tslice,ptcl]
                PE = PE + self.Vext(R)
        return PE/(self.numTimeSlices*hbar)
    
    def PotentialEnergy1(self):
        '''The operator potential energy estimator.'''
        PE = 0.0
        for tslice in range(self.numTimeSlices):
            for ptcl in range(self.numParticles):
                R = self.beads[tslice,ptcl]
                PE = PE + self.Vext(R) + 0.5*R*R
        return PE/(self.numTimeSlices*hbar)
    
    

    def Energy(self):
        '''The total energy.'''
        
        return (self.PotentialEnergy() + self.KineticEnergy(),self.PotentialEnergy() + self.KineticEnergyCA())
        #return self.PotentialEnergy() + self.KineticEnergyCA()

# ------------------------------------------------------------------------------------------- 
def PIMC(numSteps,Path):
    '''Perform a path integral Monte Carlo simulation of length numSteps.'''
    observableSkip = 5
    equilSkip = 1000
    numAccept = {'CenterOfMass':0,'Staging':0,'Metropolis':0}
    EnergyTrace = []
    EnergyTraceCA = []
    
    path_list = [[]] * int((numSteps-equilSkip)/observableSkip)
    
   

    for steps in range(0,numSteps): 
        
        
        """
        # for each particle try a center-of-mass move
        for ptcl in np.random.randint(0,Path.numParticles,Path.numParticles):
                numAccept['CenterOfMass'] += CenterOfMassMove(Path,ptcl)
        
        # for each particle try a staging move
        for ptcl in np.random.randint(0,Path.numParticles,Path.numParticles): 
            numAccept['Staging'] += StagingMove(Path,ptcl)
        """
        
        # for each particle try Metropolis move
        for ptcl in np.random.randint(0,Path.numParticles,Path.numParticles): 
            numAccept['Metropolis'] += MetropolisMoveCA(Path,ptcl)
            #numAccept['Metropolis'] += MetropolisMove(Path,ptcl)
        
            

        # measure the energy
        if steps % observableSkip == 0 and steps > equilSkip:
       
            EnergyTrace.append(Path.Energy()[0])
            EnergyTraceCA.append(Path.Energy()[1])
            path_list[int((steps-equilSkip)/observableSkip)]=deepcopy(Path.beads[:,0])
            
            
            
        #if steps % 5000 == 0:
            #print(steps)
            
    
                     
    print('Acceptance Ratios:')
    print('Center of Mass: %4.3f' %
          ((1.0*numAccept['CenterOfMass'])/(numSteps*Path.numParticles)))
    print('Staging:        %1.3f\n' %
          ((1.0*numAccept['Staging'])/(numSteps*Path.numParticles)))
    print('Metropolis:     %1.3f\n' %
          ((1.0*numAccept['Metropolis'])/(numSteps*Path.numParticles)))
    return (np.array(EnergyTrace),path_list,np.array(EnergyTraceCA))

# ------------------------------------------------------------------------------------------- 
def CenterOfMassMove(Path,ptcl):
    '''Attempts a center of mass update, displacing an entire particle
    worldline.'''
    #N=40
   
    #shift = np.random.choice([1,-1])*np.random.randint(1,N)*lat
    
    #Not lattice movement
    delta = 0.5
    shift = delta*(-1.0 + 2.0*np.random.random())
    

    # Store the positions on the worldline
    oldbeads = np.copy(Path.beads[:,ptcl])

    # Calculate the potential action
    oldAction = 0.0
    for tslice in range(Path.numTimeSlices):
        oldAction += Path.PotentialAction(tslice)

    # Displace the worldline
    for tslice in range(Path.numTimeSlices):
        Path.beads[tslice,ptcl] = oldbeads[tslice] + shift

    # Compute the new action
    newAction = 0.0
    for tslice in range(Path.numTimeSlices):
        newAction += Path.PotentialAction(tslice)

    # Accept the move, or reject and restore the bead positions
    if np.random.random() < np.exp(-(newAction - oldAction)/hbar):
        return True
    else:
        Path.beads[:,ptcl] = np.copy(oldbeads)
        return False

        
def MetropolisMove(Path, ptcl):
    
    #Not lattice movement
    h=1.0
    shift = h*(-1.0 + 2.0*np.random.random(Path.numTimeSlices))
    
    #N=3
    
    accrate=0
        
    index=0
    
    #rand=np.random.uniform(0,1,2*Path.numTimeSlices)
    
    
    #random sweeping through path
    #for tslice in np.random.randint(0,Path.numTimeSlices,Path.numTimeSlices):
        
    
    for tslice in range(Path.numTimeSlices):
    
        #tslicep1 = (tslice + 1) % Path.numTimeSlices
        tslicem1 = (tslice - 1) % Path.numTimeSlices
        #newbead=np.copy(Path.beads[tslice,ptcl])
        #newbead = newbead + h*(rand[index]-0.5)
        
        
        
        # Store the positions on the worldline
        
        oldbeads = np.copy(Path.beads[tslice,ptcl])
        
    
        # Calculate the potential action
        
        oldAction = 0.0
        oldAction= Path.PotentialAction(tslice) + Path.KineticAction(tslice) + Path.KineticAction(tslicem1)
        
        # Displace bead
    
        #Traditional
        Path.beads[tslice,ptcl] = oldbeads + shift[index]
        index+=1
        
        #Lattice
        #Path.beads[tslice,ptcl] = oldbeads + np.random.choice([1,-1])*np.random.randint(1,N)*lat


        # Compute the new action
        newAction = 0.0
        newAction= Path.PotentialAction(tslice) + Path.KineticAction(tslice) + Path.KineticAction(tslicem1)
        
        # Accept the move, or reject and restore the bead positions
        if np.random.random() < np.exp(-(newAction - oldAction)/hbar):
            accrate+=(1/Path.numTimeSlices)
            
            
        else:
            Path.beads[tslice,ptcl] = np.copy(oldbeads)
            
            
    
    return accrate
        
    
def MetropolisMoveCA(Path, ptcl):
    
    #Not lattice movement
    #h=0.75/6
    #shift = h*(-1.0 + 2.0*np.random.random(Path.numTimeSlices))
    
    N=220
    a=lat
    
    accrate=0
    
    index=0
    
    
    #rand=np.random.uniform(0,1,2*Path.numTimeSlices)
    #for tslice in np.random.randint(0,Path.numTimeSlices,Path.numTimeSlices):
        
    for tslice in range(Path.numTimeSlices):
    
        #tslicep1 = (tslice + 1) % Path.numTimeSlices
        tslicem1 = (tslice - 1) % Path.numTimeSlices
        #newbead=np.copy(Path.beads[tslice,ptcl])
        #newbead = newbead + h*(rand[index]-0.5)

        # Store the positions on the worldline
        
        oldbeads = np.copy(Path.beads[tslice,ptcl])
        
    
        # Calculate the potential action
        
        oldAction = 0.0
        oldAction= math.exp(-Path.PotentialAction(tslice))*Path.KineticActionCA(tslice)*Path.KineticActionCA(tslicem1)
        
        # Displace bead
    
        #Traditional
        #Path.beads[tslice,ptcl] = oldbeads + shift[index]
        #index+=1
        
        #Lattice
        Path.beads[tslice,ptcl] = oldbeads + np.random.choice([1,-1])*np.random.randint(1,N)*a
 

        # Compute the new action
        newAction = 0.0
        newAction= math.exp(-Path.PotentialAction(tslice))*Path.KineticActionCA(tslice)*Path.KineticActionCA(tslicem1)
       
        # Accept the move, or reject and restore the bead positions
        
        acc=min(1,newAction/oldAction)
        
        if np.random.random() < acc:
            #print(acc)
            #print("yes")
            accrate+=(1/Path.numTimeSlices)
            
            
            
        else:
            #pri©nt(np.exp(-(newAction - oldAction)/hbar))
            #print("no")
            Path.beads[tslice,ptcl] = np.copy(oldbeads)
            
            
    
    return accrate
        
 

# ------------------------------------------------------------------------------------------- 
def StagingMove(Path,ptcl):
    '''Attempts a staging move, which exactly samples the free particle
    propagator between two positions.

    See: http://link.aps.org/doi/10.1103/PhysRevB.31.4234
    
    Note: does not work for periodic boundary conditions.
    '''
    
    # the length of the stage, must be less than numTimeSlices
    m1 = 8

    # Choose the start and end of the stage
    alpha_start = np.random.randint(0,Path.numTimeSlices)
    alpha_end = (alpha_start + m1) % Path.numTimeSlices

    # Record the positions of the beads to be updated and store the action
    oldbeads = np.zeros(m1-1)
    oldAction = 0.0
    for a in range(1,m1):
        tslice = (alpha_start + a) % Path.numTimeSlices
        oldbeads[a-1] = Path.beads[tslice,ptcl]
        oldAction += Path.PotentialAction(tslice)

    # Generate new positions and accumulate the new action
    newAction = 0.0;
    for a in range(1,m1):
        tslice = (alpha_start + a) % Path.numTimeSlices
        tslicem1 = (tslice - 1) % Path.numTimeSlices
        tau1 = (m1-a)*Path.tau
        avex = (tau1*Path.beads[tslicem1,ptcl] +
                Path.tau*Path.beads[alpha_end,ptcl]) / (Path.tau + tau1)
        sigma2 = 2.0*Path.lam / (1.0 / Path.tau + 1.0 / tau1)
        Path.beads[tslice,ptcl] = avex + np.sqrt(sigma2)*np.random.randn()
        #Path.beads[tslice,ptcl] = round((avex + np.sqrt(sigma2)*np.random.randn())/lat)*lat
        
        newAction += Path.PotentialAction(tslice)

    # Perform the Metropolis step, if we rejct, revert the worldline
    if np.random.random() < np.exp(-(newAction - oldAction)/hbar):
        return True
    else:
        for a in range(1,m1):
            tslice = (alpha_start + a) % Path.numTimeSlices
            Path.beads[tslice,ptcl] = oldbeads[a-1]
        return False

# ------------------------------------------------------------------------------------------- 
def main():

    print('Simulation Parameters:')
    #print('N      = %d' % numParticles)
    print('N      = %d' % numTimeSlices)
    print('tau    = %6.4f' % tau)
    #print('lambda = %6.4f' % lam)
    print('T      = %4.2f\n' % T)
    print('Lat    =' ,lat)

    # fix the random seed
    np.random.seed(1173)

    # initialize main data structure
    beads = np.zeros([numTimeSlices,numParticles])
    
    
    """ FOR INITIAL PATH
    # random initial positions (classical state) 
    for tslice in range(numTimeSlices):
        for ptcl in range(numParticles):
            #beads[tslice,ptcl] = 0.5*(-1.0 + 2.0*np.random.random())
            beads[tslice,ptcl] = 10*np.random.random()
    """


    # setup the paths
    Path = Paths(beads,tau,lam)
    Path.SetPotential(HarmonicOscillator)

    # compute the energy via path-integral Monte Carlo
    Energy,Pathss,EnergyCA, = PIMC(numMCSteps,Path)
   
    
    MC_step=np.linspace(0,len(Energy)-1,len(Energy))
    
    
    
   
    
    
    # Do some simple binning statistics
    numBins = int(1.0*len(Energy)/binSize)
    slices = np.linspace(0, len(Energy),numBins+1,dtype=int)
    binnedEnergy = np.add.reduceat(Energy, slices[:-1]) / np.diff(slices)
    
    binnedEnergyCA = np.add.reduceat(EnergyCA, slices[:-1]) / np.diff(slices)
    
    Mean=np.ones(len(MC_step))
    Mean*=np.mean(binnedEnergy)
    
    plt.figure
    plt.plot(MC_step,Energy,".")
    plt.plot(MC_step,Mean,"-",label='Energy   = %8.4f +/- %6.4f' % (np.mean(binnedEnergy),
                                        (np.std(binnedEnergy)/np.sqrt(numBins-1))))
    plt.xlabel("Monte Carlo Iteration")
    plt.ylabel("Energy")
    plt.legend()
    plt.show
    #plt.savefig('Energy_triangle_long_metro.pdf')

    # output the final result
    print('Energy   = %8.4f +/- %6.4f' % (np.mean(binnedEnergy),
                                        (np.std(binnedEnergy)/np.sqrt(numBins-1))))
    print('EnergyCA = %8.4f +/- %6.4f' % (np.mean(binnedEnergyCA),
                                        (np.std(binnedEnergyCA)/np.sqrt(numBins-1))))
    print('Eexact   = %8.4f' % SHOEnergyExact(T))
    
    
    
    return(Energy,EnergyCA,Pathss)
    
path_list_E,path_list_E_CA,path_list=main()

# ----------------------------------------------------------------------


# %%

def histo(path):
    
   x_min=np.amin(path[:])
   x_max=np.amax(path[:])
   number_bins=41
   
   


   histo = histogram1d(path,range=[x_min,x_max],bins=number_bins)
   histo=histo/((len(path_list)-1)*numTimeSlices)
   #histo=histo/(numTimeSlices)
   h=np.linspace(x_min,x_max,number_bins)
   plt.plot(h,histo,".")
   
   plt.xlabel("Position")
   plt.ylabel("Expectation value <x>")
   #plt.savefig('prob_dist_triangle_metro.pdf')
   
# %%
   
    
def mean():
    x_exp=np.zeros(len(path_list))
    x=np.linspace(0,len(path_list)-1,len(path_list))
    x_mean=0
    for i in range(1,len(path_list)):
        for j in range(0,numTimeSlices):
            x_exp[i]+=path_list[i][j]
        x_exp[i]=x_exp[i]/numTimeSlices
    for i in range(1,len(path_list)):
        x_mean+=x_exp[i]
    x_mean=x_mean/len(path_list)
    print(x_mean)
    const=np.ones(len(path_list))*x_mean
    plt.figure()
    plt.plot(x[1:],x_exp[1:],".")
    plt.plot(x[1:],const[1:])
    plt.xlabel("sweep")
    plt.ylabel("mean")
    plt.show    

# %%
    
def G2(path,del_tau):
    corr=0
    for i in range(0,len(path)):
        k=(i+del_tau)%len(path)
        corr+=path[i]*path[k]
    return(corr/len(path))
    
# %%
    
def ES(lop):
    tauu=30
    del_tau=np.linspace(1,tauu,tauu,dtype=int)
    G2=np.zeros(tauu)
    
    for k in del_tau:
        stock=0
        
        for j in range(1,len(lop)):
            corr=0
            for i in range(0,numTimeSlices):
                i_del=(i+k)%numTimeSlices
                corr+=lop[j][i]*lop[j][i_del]
                #corr+=lop[j][i]
            stock+=(corr/numTimeSlices)
            #print('hallo',stock)
            
        stock=stock/(len(lop)-1)
        #G2[k-1]=math.log(stock)
        G2[k-1]=stock
        
    plt.figure()
    plt.plot(del_tau,np.log(G2),".")
    plt.xlabel("tau")
    plt.ylabel("log(G2)")
    plt.show    
    
    #print(-(np.log(G2[7])-np.log(G2[6])))
    print(-(np.log(G2[7])-np.log(G2[6]))/tau)
        
    
    
# %%

def ES_gen(lop):
    tauu=30
    del_tau=np.linspace(1,tauu,tauu,dtype=int)
    G2=np.zeros(tauu)
    
    for k in del_tau:
        stock=0
        stock_mean=0
        stock_mean_dec=0
        
        for j in range(1,len(lop)):
            corr=0
            mean=0
            mean_dec=0
            for i in range(0,numTimeSlices):
                i_del=(i+k)%numTimeSlices
                corr+=lop[j][i]*lop[j][i_del]
                mean+=lop[j][i]
                #mean+=0
                
                mean_dec+=lop[j][i_del]
                
                #corr+=lop[j][i]
            stock+=(corr/numTimeSlices)
            stock_mean+=(mean/numTimeSlices)
            stock_mean_dec=(mean_dec/numTimeSlices)
            
            #print('hallo',stock)
            
        stock=stock/(len(lop)-1)
        stock_mean=stock_mean/(len(lop)-1)
        stock_mean_dec=stock_mean_dec/(len(lop)-1)
        
        #G2[k-1]=stock_mean
        G2[k-1]=stock-stock_mean*stock_mean_dec
        
    plt.figure()
    plt.plot(del_tau,np.log(G2),".")
    plt.xlabel("tau")
    plt.ylabel("log(G2)")
    plt.show    
    
    #print(-(np.log(G2[7])-np.log(G2[6])))
    print(-(np.log(G2[2])-np.log(G2[1]))/tau)
    

