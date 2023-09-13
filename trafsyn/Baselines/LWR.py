import os, pkg_resources, pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from math import sqrt
from trafsyn.utils import Qdata
from typing import Any, Tuple, Optional, List

################################ EXPLICIT SOLUTIONS ################################ 
def func1(x,t)->float:
    """ Explicit LWR solution with complex shock and rarefaction waves 
    computed by Alex.B """

    if t <= 1:
        if x < 1/2*t:
            value = 1
        elif x > 1/2*t and x <= 1:
            value = 1 + (2*x - t)/(2 - t)
        elif x > 1 and x <= 10:
            value = 2
        elif x > 10 and x <= 11 - t:
            value = 2 + 2*(x - 10)/(1 - t)
        elif x > 11 - t and x <= 20 - t:
            value = 4
        elif x > 20 - t and x < 1/2*t + 20:
            value = 2 - 2*(x - 20)/t
        else:
            value = 1
    
    elif t > 1 and t <= 2:
        if x <= 1/2*t:
            value = 1
        elif x > 1/2*t and x <= 1:
            value = 1 + (2*x - t)/(2 - t)
        elif x > 1 and t <= -2*x + 21:
            value = 2
        elif t > -2*x + 21 and x <= 20 - t:
            value = 4
        elif x > 20 - t and x <= 1/2*t + 20:
            value = 2 - 2*(x - 20)/t
        else:
            value = 1
    elif t > 2 and t <= 40/3:
        if t > 4*x - 2:
            value = 1
        elif t <= -2*x + 21 and t <= 4*x - 2:
            value = 2
        elif t <= 20 - x and t > -2*x + 21:
            value = 4
        elif x > 20 - t and x < 1/2*t + 20:
            value = 2 - 2*(x - 20)/t
        else:
            value = 1
    elif t > 40/3 and t <= 154/9:
        if t < -4*x + 86/3:
            value = 1
        elif t > -4*x + 86/3 and t < 20 - x:
            value = 4
        elif x > 20 - t and x < 1/2*t + 20:
            value = 2 - 2*(x - 20)/t
        else:
            value = 1
    else:
        if x < (t/2 - sqrt(154*t)/2 + 20):
            value = 1
        elif x <= t/2 + 20 and x >= (t/2 - sqrt(154*t)/2 + 20):
            value = 2 - 2*(x - 20)/t
        else:
            value = 1
            
    return value

def f(r, v=1 , rmax = 4)->float:
    """Greenshield flow function"""
    return v * r * (1 - (r/rmax))

def Greenshield_RiemannSolution(x,t,c1  = 1,
                        c2  = 2,
                        v   = 1,
                        rmax= 4,
                        ) -> float:
    """Riemann soluton for LWR equation using Greenschield Fundamental Diagram
        with initial condition c1 if x<0 else c2
    Args:
        x,t (floats): point of evaluation
        c1 (float): left (x<0) initial density
        c2 (float): right (x>0) initial density
        v (float): free flow velocity (greenshield)
        rmax (float): maximum density  (greenshield)
    """

    # Greenshield_flux 
    f = lambda r: v * r * (1- r/rmax)
    # derivative
    fp = lambda r: v * (1 - (2*r)/rmax)


    #  c1 < c2 : Simple shockwave
    if c1 <= c2:
        slope = (f(c2) - f(c1))/(c2-c1)

        if x <= slope * t:
            return c1
        else:
            return c2
    else:
        s1 = fp(c1)
        s2 = fp(c2)

        if x <= s1*t:
            return c1
        elif x >=  s2*t:
            return c2
        else:
            return 0.5 * rmax * ( 1 - x /(t * v) )

def Triangular_RiemannSolution(x,t,c1  = 1,
                        c2  = 2,
                        r_c   = 1,
                        v_c   = 1,
                        rmax= 2,
                        ) -> float:
    """Riemann soluton for LWR equation using a triangular Fundamental Diagram
        with initial condition c1 if x<0 else c2
    Args:
        x,t (floats): point of evaluation
        c1 (float): left (x<0) initial density
        c2 (float): right (x>0) initial density
        r_c (float): critical density 
        v_c (float): critical (maximum) speed at r_c
        rmax (float): maximum density 
    """
    
    # triangular fundamental diagram function
    def f_tr(r,r_c=1,v_c=1,rmax=2):
        return (v_c * r) * (r < r_c)  +  ( v_c * r_c * (rmax - r)/(rmax - r_c) ) * (r >= r_c)

    # derivative
    def f_tr_p(r,r_c=1,v_c=1,rmax=2):
        return  v_c * (r < r_c)  -  ( v_c * r_c / (rmax - r_c) ) * (r >= r_c)

    f = lambda r: f_tr(r,r_c=r_c,v_c=v_c,rmax=rmax)
    fp = lambda r: f_tr_p(r,r_c=r_c,v_c=v_c,rmax=rmax)

    #  c1 < c2 : Simple shockwave
    if c1 <= c2:
        slope = (f(c2) - f(c1))/(c2-c1)

        if x <= slope * t:
            return c1
        else:
            return c2
    else:
        s1 = fp(c1)
        s2 = fp(c2)

        if x <= s1*t:
            return c1
        elif x >=  s2*t:
            return c2
        else:
            return r_c

################################ EXPLICIT DISCRETIZERS ################################ 

def GenerateQdata1(dx,dt,Nts) -> Qdata:
    """Generate a Qdata object containing the func1 solution discretized over a given grid
    Args:
        dx (float): cell x lenght
        dt (float): duration of timestep
        Nts (int) : number of timesteps to compute
    """
    L = 50
    Nx = int(L/dx)

    X = np.linspace(-10, Nx *dx-10, Nx + 1)
    T = np.linspace(  0, (Nts-1)*dt   , Nts)

    q_lwr = np.zeros((len(X)-1,len(T)))
    print("grid shape:", q_lwr.shape)
    
    N_disc = 10

    for t_i in tqdm(range(len(T))):
        for x_i in range(len(X)-1):

            X_i = np.linspace(X[x_i],X[x_i+1],N_disc)
            Y = [func1(x,T[t_i]) for x in X_i]

            #compute the integral in order to get a conservative discretization
            q_lwr[x_i,t_i] = 1/dx * np.trapz(Y,X_i)

    return Qdata(q_lwr, dt=dt, dx= dx)

def Generate_Gr_Riemann_Qdata(dx,dt,
                            c1: float,
                            c2: float,
                            Nts:int = 100,
                            L:float=50,
                            x0:float = 25,
                            v:float = 1.,
                            rmax:float = 4.,
                            verbose = False) -> Qdata:
    """ Generate a Qdata object containing the solution of the Riemann problem of LWR-Greenshield 
        associated with the initial condition:
     rho(x,t=0) = c1 if x<L/2 else c2       for x \in [0,L] 

     Args: 
        dx (float): cell x length
        dt (float): duration of timestep
        Nts (int) : number of timesteps to compute
        c1 (float): left (x<0) initial density
        c2 (float): right (x>0) initial density
        L (float): Length of the x space to discretize: x \in [0,L]
        x0 (float): location of initial density jump
        v (float): free flow velocity (greenshield)
        rmax (float): maximum density  (greenshield)
    Returns:
        Qdata: a Qdata objects containing the discretized Riemann solution.
    """
    
    CFL = v * dt / dx

    N_disc = 5
    Nx = int(L/dx) + 1
    Nx_raw = N_disc * Nx
    dxr = dx/N_disc
    X = np.linspace( dxr/2, (Nx_raw-.5) * dxr,Nx_raw)
    T = np.linspace( 0, (Nts-1) * dt, Nts)

    q_raw = np.zeros((Nx_raw,Nts))
    q_lwr = np.zeros((Nx,Nts))
    
    if verbose:
        print("CFL=",CFL)
        print("grid shape:", q_lwr.shape)
    
    #Sample exact values of the solution on a finer grid
    for t_i in range(len(T)):
            q_raw[:,t_i] = [Greenshield_RiemannSolution(
                            x - x0,
                            T[t_i],
                            c1=c1,
                            c2=c2,
                            v=v,
                            rmax=rmax,
                            ) for x in X]

    #Aggregate data on x to have the real FVM cell values
    for i in range(Nx):
                q_lwr[i,:] = np.sum(q_raw[N_disc*i:N_disc*(i+1), :],axis=0)
    #normalize and return
    return Qdata(q_lwr/N_disc, dx= dx, dt=dt)

def Generate_Tri_Riemann_Qdata(dx,dt,
                            c1: float,
                            c2: float,
                            Nts:int = 100,
                            L:float=50,
                            x0:float = 25,
                            r_c:float = 1.,
                            v_c:float = 1.,
                            rmax:float = 2.,
                            verbose = False) -> Qdata:
    """ Generate a Qdata object containing the solution of the Riemann problem of LWR-Triangular FD 
        associated with the initial condition:
     rho(x,t=0) = c1 if x<L/2 else c2       for x \in [0,L] 

     Args: 
        dx (float): cell x length
        dt (float): duration of timestep
        Nts (int) : number of timesteps to compute
        c1 (float): left (x<0) initial density
        c2 (float): right (x>0) initial density
        L (float): Length of the x space to discretize: x \in [0,L]
        x0 (float): location of initial density jump
        r_c (float): critical density of triangular FD
        v_c (float): critical (maximum) speed at r_c for triangular FD
        rmax (float): maximum density of triangular FD
    Returns:
        Qdata: a Qdata objects containing the discretized Riemann solution.
    """
    
    CFL = v_c * dt / dx

    N_disc = 5
    Nx = int(L/dx)
    Nx_raw = N_disc * Nx
    dxr = dx/N_disc
    X = np.linspace( dxr/2, (Nx_raw-.5) * dxr,Nx_raw)
    T = np.linspace( 0, (Nts-1) * dt, Nts)

    q_raw = np.zeros((Nx_raw,Nts))
    q_lwr = np.zeros((Nx,Nts))
    
    if verbose:
        print("CFL=",CFL)
        print("grid shape:", q_lwr.shape)
    
    #Sample exact values of the solution on a finer grid
    for t_i in range(len(T)):
            q_raw[:,t_i] = [Triangular_RiemannSolution(
                            x - x0,
                            T[t_i],
                            c1=c1,
                            c2=c2,
                            r_c = r_c,
                            v_c = v_c,
                            rmax = rmax,
                            ) for x in X]

    #Aggregate data on x to have the real FVM cell values
    for i in range(Nx):
                q_lwr[i,:] = np.sum(q_raw[N_disc*i:N_disc*(i+1), :],axis=0)
    #normalize and return
    return Qdata(q_lwr/N_disc, dx= dx, dt=dt)

################ Generate Riemann Waves Training Set  ################################

def GenerateGrRiemannTrainingSet(
    N:int,
    Nts:int,
    dx:float,
    dt:float,
    plot:bool = False,
    v: float = 1,
    rmax: float = 4,
    verbose:bool=True,
    save=True,
    )-> List[Qdata]:    
    """ Generates a set of discretized Riemann waves
    TODO proper description
    """

    #folder to save generated data for further training
    dirPath = os.path.join(pkg_resources.resource_filename('trafsyn', 'data/'),'trainSet')
    if not os.path.exists(dirPath): os.makedirs(dirPath)

    # File in which we save or load the dataset 
    fileName = f"{N*(N-1)}Riemann_Green_Waves_dx{dx}_dt{dt}Nts{Nts}.pkl"
    filePath = os.path.join(dirPath,fileName)
    
    # Load dataset if exists, generate it otherwise
    if save and os.path.exists(filePath):
        # Load the list from the file
        print("Loading existing data")
        with open(filePath, 'rb') as file:
            RiemannDataQs = pickle.load(file)
        Nts = RiemannDataQs[0].shape[1]
        if verbose: print(f"loaded {len(RiemannDataQs)} Riemann waves with shape (Nx,Nt)={RiemannDataQs[0].shape}\n \
            \tdx={RiemannDataQs[0].dx},\n \
            \tdt={RiemannDataQs[0].dt}")
    else:
        # Ensure that each solution has free flow BC i.e the waves don't hit the border
        L = 2 * v * (Nts + 1) * dt
        x0 = L/2 
        RiemannDataQs = []
        X = np.linspace(0,rmax,N)
        points = np.array([[c1,c2] for c1 in X for c2 in X if c1 != c2])
      
        if verbose: print(f"generated {len(points)} points in [0,{rmax}]²: generating associated Riemann solutions...")
        if plot:
            plt.figure(figsize=(2,2))
            plt.scatter(points[:,0],points[:,1],s = 3)
            plt.show()
            
        for pt in tqdm(points):
            c1,c2 = pt
            if c1 != c2:
                RiemannDataQs.append(Generate_Gr_Riemann_Qdata( dx=dx, dt=dt,
                                c1= c1,
                                c2= c2,
                                L=L,
                                Nts=Nts,
                                x0 = x0,
                                ))
        #save training dataset
        if save:
            with open(filePath, 'wb') as file:
                pickle.dump(RiemannDataQs, file)
                if verbose: print("save dataset File as ",fileName)
    #returns the Train Set !    
    return RiemannDataQs 


def GenerateTriRiemannTrainingSet(
    N:int,
    Nts:int,
    dx:float,
    dt:float,
    plot:bool = False,
    r_c:float = 1.,
    v_c:float = 1.,
    rmax:float = 2.,
    save=True,
    verbose:bool=True,
    )-> List[Qdata]:    
    """ Generates a set of discretized Riemann waves
    TODO proper description
    """

    #folder to save generated data for further training
    dirPath = os.path.join(pkg_resources.resource_filename('trafsyn', 'data/'),'trainSet')
    if not os.path.exists(dirPath): os.makedirs(dirPath)

    # File in which we save or load the dataset 
    fileName = f"{N*(N-1)}Riemann_Tri_Waves_dx{dx}_dt{dt}Nts{Nts}.pkl"
    filePath = os.path.join(dirPath,fileName)
    
    # Load dataset if exists, generate it otherwise
    if save and os.path.exists(filePath):
        # Load the list from the file
        print("Loading existing data")
        with open(filePath, 'rb') as file:
            RiemannDataQs = pickle.load(file)
        Nts = RiemannDataQs[0].shape[1]
        if verbose: print(f"loaded {len(RiemannDataQs)} Riemann waves with shape (Nx,Nt)={RiemannDataQs[0].shape}\n \
            \tdx={RiemannDataQs[0].dx},\n \
            \tdt={RiemannDataQs[0].dt}")
    else:
        # Create New dataset !
        # Ensure that each solution has free flow BC i.e the waves don't hit the border
        v_back = np.abs(v_c * r_c / (rmax - r_c))
        L1 = v_c * (Nts + 2) * dt
        L2 = v_back * (Nts + 2) * dt
        L = L1 + L2
        x0 = L1 

        # Discretization
        RiemannDataQs = []
        X = np.linspace(0,rmax,N)
        points = np.array([[c1,c2] for c1 in X for c2 in X if c1 != c2])
      
        if verbose: print(f"generated {len(points)} points in [0,{rmax}]²: generating associated Riemann solutions...")
        if plot:
            #Show points sampling in [0,rmax]²
            plt.figure(figsize=(2,2))
            plt.scatter(points[:,0],points[:,1],s = 3)
            plt.show()
            
        for pt in tqdm(points):
            c1,c2 = pt
            if c1 != c2:
                RiemannDataQs.append(Generate_Tri_Riemann_Qdata( dx=dx, dt=dt,
                                c1 = c1,
                                c2 = c2,
                                L=L,
                                Nts=Nts,
                                x0 = x0,
                                r_c=r_c,
                                v_c=v_c,
                                rmax=rmax,
                                ))
        #save training dataset
        if save:
            with open(filePath, 'wb') as file:
                pickle.dump(RiemannDataQs, file)
                if verbose: print("save dataset File as ",fileName)
    #returns the Train Set !    
    return RiemannDataQs 

################################ LAX HOPF SOLVER ################################ 
def Lax_Hopf_solver(
            I0:np.array = None,
            dx:float = .1,
            dt:float = .1,
            Nt:int = 140,
            verbose: bool = True
            ) -> Qdata:
    """ Lax-Hopf solver for LWR-greenshield 
    Args:   
        I0 (np.array): inital datum
        dx (float): length of cells
        dt (float): duration of timesteps
        Nt (int) : number of timesteps to compute
    Returns:
        Qdata : the Qdata object containing the grid as well as the discretization parameters
    """

    #example piecewise constant solution
    if I0 is None:
        x = np.linspace(0,30,1000)
        I0 = 3*((x > 3)*(x < 7)) + 1*((x > 7)*(x < 10)) + 4*((x > 10)*(x < 16))
    
    #Greenshield flux params
    rmax = 4
    v = 1
    #legendre transform of Greenshield flux f(x) = v*x *( 1 - x/rmax )
    g = lambda z: - (rmax/(4*v)) * (v - z)**2

    #grid params
    Nx = len(I0) + 1
    L = dx*(Nx-1)
    T = Nt * dt 
    t = np.linspace(0, T, Nt)
    x = np.linspace(0, L, Nx)
    y = x
     
    if verbose: print(f"Computing solution on {Nx}x{Nt} grid...")
    #antiderivative of initial condition
    v_0 = np.zeros(Nx)
    for i in range(Nx):
       v_0[i] = dx * np.trapz(I0[:i+1])
    
    #computing the values
    w = np.zeros((Nt, Nx, Nx + 1))
    for i in tqdm(range(1,Nt)):
        for k in range(Nx):
            for l in range(Nx):
                w[i, k, l] = t[i] * g((x[k] - y[l]) / t[i]) + v_0[l]
            i_g_max = np.argmin(np.abs(x - (x[k]-v*t[i]) ))
            w[i, k, Nx] = v_0[i_g_max]
        
    #get the maximum
    w = np.amax(w, axis=2)
    #derive the obtained solution to get the density solution
    wx = 1 / dx * (w[:,1:] - w[:,:-1])
    #replace the first row by the initial datum
    wx[0, :] = I0

    return Qdata(wx.T,dx=dx,dt=dt)