import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.stats import linregress


"""Parameters, units: ueV, nm, ns"""
print('Setting initial constants')

interaction = 1 # Turn on/off electron-electron interaction (1 or 0)
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

print('Effective mass vector (m) = ', m,'*me' )
pot_step = 3 * 10**6 #ueV - Si/SiO2 interfac
print('Step voltage height (pot_step) = ', pot_step,' $ueV$' )
        
        
# %%
hbar = 0.6582 # ueV * ns
a = 0.543 #nm - lattice constant in silicon
w = 4000/hbar
q = 10**6 # ueV
lamda = 0.5 * hbar**2 / me

V = pot_step

# V_field = V_hrl

print('Lattice constant (a)=', a, 'nm')

# HRL_Potential = 'old_UNSW4_1st withBG TEST ephiQWM UNSW4_1st structure 2nm 800x800 - 10-4-12-4.npz'
HRL_Potential = 'UNSW4_1st withBG TEST ephiQWM for VX1_1V UNSW4_1st structure 2nm 800x800 - 10-4-12-4.npz'
print('Setting HRL potential VX1 =1V' )



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

# Heterostructure interface potential (z-direction) leading to airy functions as wave function.
def V_hrl(path):
    return interpolate(path.T) + V_step(path[2,:])


print('HRL')


def RefineGrid(xs,ys,zs):
    '''Refine the Grid of HRL
        Inputs: HRL axes: xs, ys , zs
        Return: New axes XS,YS,ZS , 
                NewMeshgrid: Grid (xs,ys,zs)-array
    '''
    XS = np.linspace(min(xs),max(xs),2*(len(xs))-1)
    YS = np.linspace(min(ys),max(ys),2*(len(ys))-1)
#     ZS = np.linspace(min(zs)+.5,max(zs)-.5,len(zs)-1)
    ZS = np.linspace(min(zs),max(zs),int(10*(max(zs)-min(zs)))+1) #--- Stronger refinement
    return XS , YS , ZS , np.array(np.meshgrid(XS, YS, ZS))

def ComputeGridFunction(Func , Grid, xs,ys,zs, transpose):
        '''
        Compute the desired space function over the Grid
        Inputs: Func: Function from 3*N-Array to N-Array
                Grid: 3D Grid
                xs,ys,zs: grid axes 
        Return: Func(Grid) - (xs,ys,zs)-array
    '''
        GridLenght = len(xs)*len(ys)*len(zs)
        GridArr = np.reshape(Grid,[3,GridLenght])
        
        
        if transpose:
            Grid_Func = Func(GridArr.T)
            Grid_Func = np.reshape(Grid_Func.T,[len(ys),len(xs),len(zs)], order= 'C') #Order is inportant in reshape
        else:
            Grid_Func = Func(GridArr)
            Grid_Func = np.reshape(Grid_Func,[len(ys),len(xs),len(zs)], order= 'C') #Order is inportant in reshape
        Grid_Func = np.moveaxis(Grid_Func, 1, 0) #After reshape the axes are inverted, need to swap axis
        
        return Grid_Func
    
# A = np.array([XS, YS , ZS, Grid, V_HRL])

'''Define grid and HRL potential'''
xs = np.load(HRL_Potential)['xs']
ys = np.load(HRL_Potential)['ys']
zs = np.load(HRL_Potential)['zs']
ephi = 10**6 * np.load(HRL_Potential)['ephi']

interpolate = rgi(points = (xs,ys,zs), values = ephi, bounds_error = False, method = 'linear') #options 'linear' , 'nearest'

XS, YS , ZS, Grid =  RefineGrid(xs,ys,zs)
V_HRL = ComputeGridFunction(V_hrl, Grid, XS, YS , ZS,False)

print('Saved in V_HRL_Grid')
np.savez('V_HRL_Grid' , XS=XS, YS=YS , ZS = ZS, Grid = Grid, V_HRL = V_HRL  )