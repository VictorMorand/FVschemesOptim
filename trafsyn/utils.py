"""A module containing different utilities."""
import os, scipy
import numpy as np
import pkg_resources
from typing import Callable, Any, Tuple, Optional

# Data class
class Qdata(np.ndarray):
    """A subclass of np.ndarray with extra dx and dt features
    """
    def __new__(cls, input_array, dx:float = 1., dt:float = 1.):
        """Method called at the cration of the object
        Usage new_qtata = Qdata(array, dx, dt)
        Args:
            input_array (np.array): Array containing the data
            dx (float): the cell space length
            dt (float): the cell time span  
        """
        obj = np.asarray(input_array).view(cls)
        obj.dx = dx
        obj.dt = dt
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.dx = getattr(obj, 'dx', None)
        self.dt = getattr(obj, 'dt', None)
    
    def __reduce__(self):
        """needed override for pickle dumping and loading"""
        # Get the parent's __reduce__ tuple
        pickled_state = super(Qdata, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.dx,self.dt,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        """needed override for pickle dumping and loading"""
        self.dx = state[-2]  
        self.dt = state[-1]  
        # Call the parent's __setstate__ with the other tuple elements.
        super(Qdata, self).__setstate__(state[:-2])

# Generic model class
class Model(object):
    """
         - Generic model callable interface for trafsyn -
        in which we can embed all the models under study and easily compare or evaluate them
        
    """
    # Model parameters class attribute
    p = None
    
    def __init__(self):
        pass

    def __call__(self,
        qx0 : Qdata,
        m: int,
        qr: Optional[np.array] = None,
        ql: Optional[np.array] = None,
        ) -> Qdata:
        """Evaluate arbitrary steps of the flow model on an initial state.
        Make a prediction with the model from initial density data qx0 on m timesteps
        
        Args:
            qx0 (Qdata): Vector of initial cell densities containing grid parameters.
            m (int)       : The number of time steps to predict with the model
            qr (Optional: np.array with len m) : Vector containing the values of the right boundary condition, if None: free flow BC
            qr (Optional: np.array with len m) : Vector containing the values of the left boundary condition, if None: free flow BC
        
        Returns:
            Qdata(np.array) of shape (len(qx0), m) the array containing the predictions of the model as well as the grid parameters.
        """
        if qr is not None and ql is not None:
            if len(qr) != m or len(ql) != m: raise ValueError("Boundary values must be of len m!")
        else:
            qr = None
            ql = None

        Qtot = np.zeros((len(qx0),m))
        Q = qx0
        for ti in range(m):
            if qr is not None:
                #Optionally Hardly replace extremities by the given boundary condition
                Q[0] = ql[ti]  
                Q[-1] = qr[ti]  
            Q = self._call_once(Q)
            Qtot[:,ti] = Q

        return Qdata(np.array(Qtot),Q.dx,Q.dt)
    
    def _call_once(self, qx: Qdata):
        """Make one prediction with the model from initial density data qx
        Args:
            qx (Qdata): Vector of initial cell densities containing grid parameters.
        
        Returns:
            Qdata of shape len(qx) the array containing the predictions of the model after one timestep
        """
        pass


######################## DATA UTILS ########################

def loadData(file: str) -> Qdata:
    """Returns an traffic dataset from the data stored in a npy file.
    The data files are stored in the trafsyn/data folder

    Args:
        file (str): the name of the npy file to load

    Returns:
        np.ndarray:
        A two dimensional array containing density values over space and time.
        The spacial cells are the first dimension of the array, the time is the
        second.
    """
    DATA_PATH = os.path.join(pkg_resources.resource_filename('trafsyn', 'data/'),file)
    data = np.load(DATA_PATH,allow_pickle=True)
    data_dict = data.item()
    return Qdata(data_dict["grid"], dx = data_dict["dx"], dt = data_dict["dt"])

def saveData(data:Qdata, fileName: str):
    """save the data into a npy file"""
    filename = fileName.split('.')[0] + ".npy"
    data_dict = {"grid": np.array(data[:]),
            "dx": data.dx,
            "dt": data.dt}
    np.save(filename,data_dict)

######################## MODEL UTILS ########################
from trafsyn.plot import PlotData, ComparePlots, animateQdata

def evaluate(
    model: Model,
    data: Qdata,
    loss: str = 'mse',
    verbose: bool = True,
    plot: bool = True,
    hardBC: bool = False,
    animate:bool = False,
    xlim: list = None,) -> None:
        """
        Evaluate model on data: default behavior is to computes mse, can optionally show prediction VS ground truth  plot or gif animation;
        Args:
            model (Model): the model to evaluate
            data (QData): the data to evaluate the model on
            loss (str): which metric to compute: can be on of {'mse','mae','l1',l2'}
            verbose (bool): Print additionnal information about the model.
            plot (bool) : optionally plots the prediction vs ground truth and absolute error
            hardBC (bool): Optionally use hard BC inference when predicting solutions with tyhe model
            xlim Optionnal(list): if plotting, limits the visualization of the solution/prediction to given x range 
        
        Returns:
                (float) the error metric computed as specified by loss.
        """
        if verbose: print("MODEL EVALUATION...")
        
        if hardBC:
                qr = data[-1,1:]
                ql = data[0,1:]
        else:
                qr = None
                ql = None

        #make prediction over whole data
        q_pred = model.__call__(
            data[:,0], 
            data.shape[1] - 1,
            ql = ql,
            qr = qr,
            )

        # TODO remove hard BC from evaluation !
        
        #Print model parameters
        if verbose: print("number of parameters: " + str(model.count_parameters()))

        #Compute given error metric on data
        if loss == 'mae':
                err = np.mean( np.abs(q_pred - data[:,1:])).item()
        elif loss == "l2":
                err = np.sqrt( data.dx * data.dt * np.sum((q_pred - data[:,1:])**2) ).item() 
        elif loss == "l1":
                err =  data.dx * data.dt * np.sum(np.abs(q_pred - data[:,1:])).item() 
        else: #default mse
                err = np.mean( (q_pred - data[:,1:])**2 ).item() 
        if verbose: print(f"Computed {loss} on data: {err}")
        
        #Optionally plot the data
        if plot:
            ComparePlots(
                data[:,1:],
                q_pred,
                dx = data.dx,
                dt = data.dt,
                title1 = "Ground truth",
                title2 = "Model prediction",
                xlim=xlim,
                )

            # Plots the Absolute Error
            PlotData( np.abs(q_pred - data[:,1:]),
                       xlim=xlim,
                       title="absolute error",
                       cmap='afmhot',
                       show=True)
        if animate:
            if verbose: print("Drawing animation of prediction vs Ground truth...")
            animateQdata([data[:,1:],q_pred],
                         styles=['c-','r--'],
                         legends=["Ground Truth","Model Prediction"],
                         filename="Prediction.gif")

        return err


def generate_state_transition_samples(qx):
    """Generate state transition samples from recorded traffic data.

    Args:
        qx (np.ndarray): Matrix of cell densities.
    Returns:
        np.ndarray:
        An array of shape `(qx.shape[0], 2, qx.shape[1]-1)`
        initial and successor density samples. The first
        array dimension corresponds to the spacial index,
        the second to the time index, and the third to the
        sample index.
    """
    ind = np.arange(qx.shape[1]-1) + np.array([0,1])[:,None]
    return qx[:,ind]

def generate_stencil_samples(qx, a, b, const_extrp=True):
    """Generate generate stencils from recorded traffic data.
    
    Stencils are the standard abstraction to compute the flow
    between cells from. In particular, the flow is needed
    to compute successor states.
    Stencils are padded by zeros on the boundary.

    Args:
        qx (np.ndarray): Matrix of cell densities.
        a (int): Extension of the stencil to the left.
        b (int): Extension of the stencil to the right.
        const_extrp (bool): Whether to do constant extrapolation
        at the boundary. If set to `False`, the boundary is
        zero-padded.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
        Two arrays of shape `(a+b+1, qx.shape[0], qx.shape[1])`.
        In the first array, the index `:,x,t` contains the values
        `qx[x-a:x+b+1,t]`. In the second array, the index `:,x,t`
        contains the values `qx[x-a+1:x+b+2,t]`. Non-existing values
        are replaced by 0.
    """
    # generate zero padded version of qx[:, :-1]
    qx_extended = np.zeros((qx.shape[0]+a+b+1, qx.shape[1]))
    
    if const_extrp:
        qx_extended[:a,:] = qx[0,:]
        qx_extended[a+qx.shape[0]:,:] = qx[-1,:]

    qx_extended[a:a+qx.shape[0],:] = qx[:,:]
    # generate indices
    ind = np.arange(0,a+b+1)[:,None]+np.arange(0,qx.shape[0])
    # sample stencils
    Q_hat_stencils = qx_extended[ind,:]
    ind += 1
    Q_hat_stencils_plus_1 = qx_extended[ind,:]

    return Q_hat_stencils, Q_hat_stencils_plus_1

def compute_flux(qx, f):
    """Compute the fluxes associated with a dataset.
     
    Args:
        qx (np.ndarray): Matrix of cell densities.
        f (np.ndarray): The flux function that governs
        the PDE of which the data is a discretized solution.
    Returns:
        np.ndarray:
        The exact flux that describes the dataset.
        The shape of the matrix is `(qx.shape[0], qx.shape[1]-1)`.
    """
    L = qx.shape[0]
    T = qx.shape[1]
    B = np.array((qx.dx/qx.dt)*(qx[:,1:]-qx[:,:-1]))
    A = np.zeros((L,L))
    A[np.arange(0,L),np.arange(0,L)] = 1.
    A[np.arange(0,L-1),np.arange(1,L)] = -1.
    B[-1,:] += f(qx[-1,:-1])
    return scipy.linalg.solve_triangular(A, B)

def generate_flux_stencil_dataset(qxs, f, a, b):
    """Generate a dataset with all possible stencils and
    the associated fluxes.
     
    Args:
        qxs (List[np.ndarray]): List of matrices of
        cell densities with spacial cells corresponding
        to rows and time steps corresponding to columns.
        f (np.ndarray): The flux function that governs
        the PDE of which the data is a discretized solution.
        a (int): Non-negative integer representing the left-side
        extension of the stencil.
        b (int): Non-negative integer representing the right-side
        extension of the stencil.
    
    Returns:
        Tuple[nd.array, nd.array]:
        A matrix of shape
        `(a+b+1, np.sum([qx.shape[0]*(qx.shape[1]-1) for qx in qxs]))`
        representing the independent variable of the dataset, i.e.,
        the stencils, and a vector of length
        `np.sum([qx.shape[0]*(qx.shape[1]-1) for qx in qxs])`
        representing the dependent variable of the dataset, i.e.,
        the associated fluxes.
    """
    Q_stencils_all = []
    F_all = []

    for qx in qxs:
        F = compute_flux(qx, f)
        # last time stencils are not needed
        qxab = generate_stencil_samples(qx, a, b)[0][:,:,:-1]

        # reshape flatten space and time
        Fflat = np.reshape(F, newshape=(F.shape[0]*F.shape[1],), order='F')
        qxabflat = np.reshape(qxab, newshape=(qxab.shape[0], qxab.shape[1]*qxab.shape[2]), order='F')

        F_all.append(Fflat)
        Q_stencils_all.append(qxabflat)

    # concatenate over datasets
    Q_stencils_all = np.concatenate(Q_stencils_all, axis=1)
    F_all = np.concatenate(F_all)
    return Q_stencils_all, F_all