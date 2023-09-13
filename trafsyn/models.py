"""
Implementation of the core functionality of the package.

A general class of flow models that are described discrete conservation
laws governed by non-linear flow models of stencils is defined by
``FlowModel``.
"""
import numpy as np
from pyfunctionbases import RecursiveExpansion
from .utils import generate_state_transition_samples, generate_stencil_samples


class FlowModel(object):
    """A flow model for discrete conservation laws based on a
    basis of non-linear functions on stencils.
    
    See the section on models for details.

    Args:
        a (int): Non-negative integer representing the left-side
        extension of the stencil.
        b (int): Non-negative integer representing the right-side
        extension of the stencil.
        basis (str or tuple): A string of either 'standard_poly',
        'legendre_poly', 'legendre_rational','chebychev_poly'.
        For custom functions see the documentation of
        ``pyfunctionbases.RecursiveExpansion``.
        scaling (float): Scaling of the data before it is evaluated.
        Helps fit the model to the scale of the data.
    """
    def __init__(self, a, b, order, basis="legendre_poly", scaling=None):
        """Initialize a flow model."""
        self.a = a
        self.b = b
        self.order = order
        self.scaling = scaling
        # expansion object for evaluating basis functions
        self.expn = RecursiveExpansion(order, recf=basis)
        
    def __call__(self, p, qx0, m):
        """Evaluate arbitrary steps of the flow model on an initial state.
        
        Args:
            p (np.ndarray): A vector of length `(order+1)**(a+b+1)`
            representing the coefficients of a model encoded in the
            specified basis (full tensor product).
            qx0 (np.ndarray): Vector of initial cell densities.
            m (int or np.ndarray): Non-negative integer
            or vector of unique non-negative integers. Indicates
            which time steps of the flow model starting from
            the supplied initial density are returned.

        Returns:
            np.ndarray:
            If `m` is an integer, then a vector of length `len(qx0)` is
            returned that represents the mth succesor state to `qx0` in
            the specified model.
            If `m` is an integer vector, an array of shape `(len(qx0), len(m))`
            containing the density representing all mth-successor states to
            `qx0` in the specified model.
        """
        k  = 0
        qxnext = qx0.copy()
        # compute the dynamical system at time m
        if isinstance(m, int):
            while k < m:
                qxnext[:] = self._call_once(p, qxnext)
                k += 1
            return qxnext
        # compute the dynamical sytem at times m
        else:
            msorted = np.sort(m)
            qxtraj = np.empty((len(qx0), len(m)))
            j = 0
            while k < msorted[-1]:
                if msorted[j] == k:
                    qxtraj[:,j] = qxnext
                    j += 1
                qxnext[:] = self._call_once(p, qxnext)
                k += 1
            qxtraj[:,j] = qxnext
            return qxtraj
            

    def _call_once(self, p, qx0):
        """Evaluate one step of the flow model on an initial state.
        
        Args:
            p (np.ndarray): A vector of length `(order+1)**(a+b+1)`
            representing the coefficients of a model encoded in the
            specified basis (full tensor product).
            qx0 (np.ndarray): Vector of initial cell densities.

        Returns:
            np.ndarray:
            A vector of length `len(qx0)` containing the density
            representing the successor state to `qx0` in the
            specified model.
        """
        qx0 = np.reshape(qx0, (len(qx0), 1))
        # independent variable
        basis_diff = self._basis_evaluation(qx0)
        # dependent variable
        pred = basis_diff@p
        # add flow and previous value
        pred += qx0[:,0]
        # clip to stop error propagation
        #np.maximum(0.,pred,out=pred)
        return pred

    def fit(self, qx):
        """Fit the flow model to supplied data.
        
        This function returns parameters that L2-optimally
        describe the supplied data within the specified
        model class.
        
        Args:
            qx (np.ndarray): Matrix of cell densities with
            spacial cells corresponding to rows and time steps
            corresponding to columns.

        Returns:
            np.ndarray:
            A vector of length `(order+1)**(a+b+1)` representing
            the coefficients of the optimal model encoded in the
            specified basis (full tensor product).
        """
        n_samples = qx.shape[0]*(qx.shape[1]-1)

        # independent variable;
        # drop the last timestep as there is
        # no data for the dependent variable
        basis_diff = self._basis_evaluation(qx[:,:-1])
        
        # dependent variable
        Q_hat = generate_state_transition_samples(qx)
        flow_diff = np.reshape(Q_hat[:,1,:].T, n_samples)-np.reshape(Q_hat[:,0,:].T, n_samples)
        
        # find the coefficients of the least squares fit
        p_star = np.linalg.lstsq(basis_diff, flow_diff, rcond=.0175)
        return p_star[0]

    def _basis_evaluation(self, qx):
        """Compute the difference of evaluations of the specified
        basis on all possible stencils and their shift in `qx`.
        
        This function generates the independent variable of a
        least squares formulation of non-linear flow-model fitting
        to observed data.
        
        Args:
            qx (np.ndarray): Matrix of cell densities with
            spacial cells corresponding to rows and time steps
            corresponding to columns.

        Returns:
            np.ndarray:
            An array of shape `(qx.shape[0]*qx.shape[1], (order+1)**(a+b+1)`
            containing the difference in value of each function of
            the selected basis evaluated on all available stencils and the
            shifted stencils. The rows correspond to samples and the
            columns correspond to basis functions (full tensor product).
        """
        qx = qx.copy()
        a, b, order = self.a, self.b, self.order
        if self.scaling is not None:
            qx /= self.scaling
    
        n_samples = qx.shape[0]*qx.shape[1]
            
        # generate stencils for inflow and stencils plus 1 for outflow calculation
        Q_hat_stencils, Q_hat_stencils_plus_1 \
            = generate_stencil_samples(qx, a, b)
            
        # then make a dataset out of all stencils
        Q_hat_stencils_dataset = np.reshape(Q_hat_stencils.T,
            (n_samples, a+b+1))
        Q_hat_stencils_dataset_plus_1 = np.reshape(Q_hat_stencils_plus_1.T,
            (n_samples, a+b+1))

        # compute the basis functions
        basis_eval = self.expn.execute(Q_hat_stencils_dataset, prec=None)
        basis_eval_plus_1 = self.expn.execute(Q_hat_stencils_dataset_plus_1, prec=None)

        # map multidim array from the tensor product to 1-d array
        basis_eval = np.reshape(basis_eval, (n_samples, (order+1)**(a+b+1)))
        basis_eval_plus_1 = np.reshape(basis_eval_plus_1, (n_samples, (order+1)**(a+b+1)))
        
        basis_eval -= basis_eval_plus_1
        return basis_eval

