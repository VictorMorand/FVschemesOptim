# This file contains code that implements a baseline Godunov solver for LWR-Greenshield traffic model  
# using initial code from P.Goatin, June 2017

import numpy as np
import matplotlib.pyplot as plt
from trafsyn.utils import Model, Qdata


def f_greenshield(Q,v=1,qmax=4):
    return Q * v * (1 - Q/qmax)

def f_tr(Q,q_c=1,f_c=1,qmax=2):
    return (f_c * Q/q_c) * (Q < q_c)  +  ( f_c * (qmax - Q)/(qmax - q_c) ) * (Q >= q_c)


################################################################################################
####################################   GODUNOV SCHEME    #######################################
################################################################################################

# Godunov Numerical Flux
def F_godunov(Q,f,q_c):
    """Godunov numerical flux for a CONCAVE flux f with max value at q = q_c
    Args:
        Q (np.array) array of densities Q 

    Returns:
        (np.array) : (F_{i+1/2}) for i in range(len(Q))
    """
    #  demand
    D = f(Q) * ( Q < q_c) + f(q_c) * ( Q >= q_c)
    #  supply
    S = f(Q) * ( Q > q_c) + f(q_c) * ( Q <= q_c)
    S = np.append(S[1:], S[len(S) - 1])
    # Godunov flux
    F = np.minimum(D, S)
    return F

# embedding of the Godunov scheme in Model class
class GodunovModel(Model):
    """ Model that implements the Godunov scheme for concave Fluxes
    Implementing trafsyn Model interface
    """
    def __init__(self, f, q_c):
        """Model that implements the Godunov scheme 
        using a concave flux function given as a parameter
        Args:
            f (function): CONCAVE flux function with max value at q = q_c
            q_c (float): the critical density, i.e the argmax of the flux f 
        """
        super(GodunovModel, self).__init__()
        
        # solver parameters
        self.F = lambda Q : F_godunov(Q,f,q_c)

    def _call_once(self, qx: Qdata):
        """Make one prediction with Godunov scheme from initial density data qx
        prediction over several Timesteps is implemented in Model interface
        Args:
            qx (Qdata): Vector of initial cell densities Containing grid parameters.
        
        Returns:
            np.array of shape len(qx) the array containing the predictions of the model after one timestep
        """
        F = self.F(qx)
        Fm = np.insert(F[0:len(F) - 1], 0, F[0])
        qx = qx - (qx.dt / qx.dx) * (F - Fm)
        
        return qx 

    def count_parameters(self)->int:
        """Returns the number of parameters of the model"""
        return 2


################################################################################################
####################################   ENGQUIST-OSHER SCHEME  ##################################
################################################################################################

#  Engquist-Osher Numerical Flux
def F_EO(Q,f,q_c):
    """Engquist-Osher numerical flux for CONVEX flux"""
    f_1 = f(Q)
    f_2 = np.append(f_1[1:], f_1[-1])
    f_c = f(q_c) * np.ones_like(Q)
    Q2 = np.append(Q[1:], Q[-1])

    return f_1 * (Q <= q_c)*(Q2 <= q_c) + f_2 * (Q > q_c)*(Q2 > q_c) + f_c * (Q > q_c)*(Q2 <= q_c) + (f_1 + f_2 - f_c) * (Q2 > q_c)*(Q <= q_c)

# embedding of the Engquist-Osher scheme in Model class
class EngquistOsherModel(Model):
    """Model that implements the Engquist-Osher scheme for concave Fluxes 
    Implementing trafsyn Model interface
    """
    def __init__(self, f, q_c):
        """Model that implements the Engquist-Osher scheme 
        using a concave flux function given as a parameter
        Args:
            f (function): CONCAVE flux function with max value at q = q_c
            q_c (float): the critical density, i.e the argmax of the flux f
        """
        super(EngquistOsherModel, self).__init__()
        
        # solver parameters
        self.F = lambda Q : F_EO(Q,f,q_c)

    def _call_once(self, qx: Qdata):
        """Make one prediction with Godunov scheme from initial density data qx
        prediction over several Timesteps is implemented in Model interface
        Args:
            qx (Qdata): Vector of initial cell densities Containing grid parameters.
        
        Returns:
            np.array of shape len(qx) the array containing the predictions of the model after one timestep
        """
        F = self.F(qx)
        Fm = np.insert(F[0:len(F) - 1], 0, F[0])
        qx = qx - (qx.dt / qx.dx) * (F - Fm)
        
        return qx 

    def count_parameters(self)->int:
        """Returns the number of parameters of the model"""
        return 2


################################################################################################
####################################   LAX-FRIEDRICHS SCHEME   #################################
################################################################################################

class LaxFriedrichs(Model):
    """Model that implements the Lax-Fiedrichs scheme
    Implementing trafsyn Model interface
    """
    def __init__(self, f):
        """Model that implements the LxF scheme 
        using the flux function given as input parameter
        Args:
            f (function): flux function 
        """
        super(LaxFriedrichs, self).__init__()
        
        # flux function
        self.f = f

    def _call_once(self, qx: Qdata):
        """Make one prediction with LxF scheme from initial density data qx
        prediction over several Timesteps is implemented in Model interface
        Args:
            qx (Qdata): Vector of initial cell densities Containing grid parameters.
        
        Returns:
            np.array of shape len(qx) the array containing the predictions of the model after one timestep
        """
        #padding
        Q = np.concatenate(([qx[0]],qx,[qx[-1]]))
        #Apply LxF scheme
        F = self.f(Q)
        Q = 1/2 * ( Q[:-2] + Q[2:]) - ((qx.dx)/(2*qx.dt)) * ( F[2:] - F[:-2] )  
        return Qdata(Q,qx.dx,qx.dt)
    
    
    def count_parameters(self)->int:
        """Returns the number of parameters of the model"""
        return 2

################################################################################################
####################################  PLACEHOLDER FLUX SCHEME   ################################
################################################################################################

class FluxModel(Model):
    """
        Generic Scheme using any given (0,1)-stencil Numerical Flux 
        Implementing Model interface from trafsyn 
    """
    def __init__(self, flux):
        """Create a scheme from the given numerical flux function
        Args:
            flux: (function: rho_i,rho_{i+1} -> F_{i+1/2}) (0,1)-stencil Numerical Flux to use in the scheme
        """
        super(FluxModel, self).__init__()
        
        # solver parameters
        self.flux = flux

    def _call_once(self, qx: Qdata):
        """Make one prediction with the conservative scheme from initial density data qx
        prediction over several Timesteps is implemented in Model interface
        Args:
            qx (Qdata): Vector of initial cell densities Containing grid parameters.
        
        Returns:
            np.array of shape len(qx) the array containing the predictions of the model after one timestep
        """
        F = self.flux(qx,np.append(qx[1:], qx[-1]))
        Fm = np.insert(F[0:len(F) - 1], 0, F[0])
        qx = qx - (qx.dt / qx.dx) * (F - Fm)
        return qx 

    def count_parameters(self)->int:
        """Returns the number of parameters of the model"""
        return None


if __name__ == "__main__":
    # quick usage of the code in this file
    from trafsyn.utils import loadData, evaluate

    #load data
    print("Loading LWR exact data...")
    data = loadData("LWRexact.npy")

    #Create Godunov model
    print("creating Godunov model...")
    Gmodel = GodunovModel(f_greenshield,q_c=2)

    #quick evaluation of the model
    evaluate(Gmodel, data)
