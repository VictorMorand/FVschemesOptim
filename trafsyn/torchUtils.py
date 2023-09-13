# Framework needed to train and test torch model for trasyn
import numpy as np
import pandas as pd
import time, gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from trafsyn.utils import Qdata, Model
from trafsyn.plot import ComparePlots, pcolormesh
from typing import Any, Tuple, Optional, List

######################## DATA UTILS ########################

# Custom dataset class that wraps NumPy arrays
class QDataset(Dataset):
    """Wraps np arrays dataset into a pytorch compatible dataset"""

    def __init__(self, 
                 Q0: List[np.array],
                 Qnexts:List[np.array],
                 dtype=None
                 ):
        """
        Create a torch compatible dataset of traffic density states
        associated with their follwing state(s).

        Args:
        Q0 List(np.ndarray): list of 1 dimensionnal arrays of initial densities,
                with shape (len(x))
        Qnexts List(np.ndarray): Matrix containing the next(s) state(s) cell densities, 
                with `shape == (len(x), len(t))`..
                all arrays must have same shape
        """
        Nts = Qnexts[0].shape[1]
        
        if not dtype:
            #use the same precision as numpy
            dtype = torch.float64 if Q0[0].dtype == np.float64 else torch.float32

        #sanity checks
        if len(Q0) != len(Qnexts): raise ValueError(f"Got {len(Q0)} initial states and {len(Qnexts)} following ! Must have same length")
        for i, q_nxts in enumerate(Qnexts):
            len_x, len_t = q_nxts.shape
            if len(Q0[i]) != len_x:
                raise ValueError("Inital condition and prediction X shapes don't match")
            if len_t != Nts:
                raise ValueError("All predictions should be over the same amount of timesteps")
        
        #store values as tensors
        self.Q_t = [torch.tensor(qx0.T,dtype=dtype).unsqueeze(0) for qx0 in Q0]
        self.Q_nexts = [torch.tensor(qx.T,dtype=dtype) for qx in Qnexts]
        self.len = len(Q0)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.Q_t[index], self.Q_nexts[index]

# Custom dataset associating 1 state to the N following
class Nts_QDataset(Dataset):
    """Wraps np arrays into a pytorch dataset"""

    def __init__(self, 
                 Q: np.array,
                 sample_size:int = 1
                 ):
        """
        Create a torch compatible dataset of traffic density states associated with their follwing state after a timestep
        from a numpy array containing the densities of the cells over time and space.

        Args:
        Q (np.ndarray): Matrix of cell densities,with `shape == (len(x), len(t))`..
        
        """
        Q = Q.T
        self.Q_t = Q[:-sample_size]
        self.Q_nexts = []

        for i in range(1,1+len(self.Q_t)):
            self.Q_nexts.append(np.copy(Q[i:i+sample_size, : ]))

        self.Q_nexts = np.array(self.Q_nexts)
        
    def __len__(self):
        return len(self.Q_t)
    
    def __getitem__(self, index):
        # translate into a torch compatible tensor
        x = torch.from_numpy(self.Q_t[index]).unsqueeze(0)
        y = torch.from_numpy(self.Q_nexts[index])
        return x, y

######################## MODELS UTILS ########################

#custom model that allow multiple timestep prediction for better training
class multistep_model(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 Niter: int = 1):
        """
            Multistep predictor from a oneStepper model
        """
        super(multistep_model, self).__init__()
        self.model = model
        self.Niter = Niter
    
    def forward(self, x):
        
        res = torch.empty((x.shape[0],self.Niter,x.shape[2]),dtype=x.dtype).to(x.device)

        for i in range(self.Niter):
            x = self.model.forward(x)
            res[:,i,:] = x.squeeze()
        
        return res

# Wrapper class for trafsyn
class torchModel(Model,nn.Module):
    """New interface that implements the Model interface from trafsyn to wrap any PyTorch model"""
    
    def __init__(self,dx,dt):
        nn.Module.__init__(self)
        Model.__init__(self)
        self.dx = dx
        self.dt = dt

    def forward(self,x):
        """ torch forward fx: ALWAYS ASSUMES Free Flow CONDITIONS"""
        pass

    # Implement Model interface
    def _call_once(self,qx:Qdata):
        """Perform a one step prediction from the state qx woth the pytorch model. Doesn't compute gradients ->  Use Only for evalutation
        Args:
            qx (Qdata): of shape (len_X) containing the initial density state
        
        Returns:
            Qdata of same shape (len_X) as qx containing the predicted density after one timestep    
        """
        self.cpu()
        self.dx = qx.dx
        self.dt = qx.dt
        tqx  = torch.tensor(qx,dtype=self.dtype).reshape((1,1,-1))
        with torch.no_grad():
            return Qdata(self.forward(tqx).reshape(-1).numpy(),
                        dx = self.dx,
                        dt = self.dt ) 
    
    #override interface implementation for faster inference with CUDA acceleration.
    def __call__(self,
        qx0 : Qdata,
        m: int,
        qr: Optional[np.array] = None,
        ql: Optional[np.array] = None,
        ) -> Qdata:
        """Evaluate arbitrary steps of the flow model on an initial state.
        Make a prediction with the model from initial density data qx0 on m timesteps
        Will use CUDA acceleration if available
        
        Args:
            qx0 (Qdata): Vector of initial cell densities containing grid parameters.
            m (int)       : The number of time steps to predict with the model
            qr (Optional: np.array with len m) : Vector containing the values of the right boundary condition, if None: free flow BC
            qr (Optional: np.array with len m) : Vector containing the values of the left boundary condition, if None: free flow BC
        
        Returns:
            Qdata(np.array) of shape (len(qx0), m) the array containing the predictions of the model as well as the grid parameters.
        """
        self.dx = qx0.dx
        self.dt = qx0.dt
        
        if qr is not None and ql is not None:
            if len(qr) != m or len(ql) != m: raise ValueError("Boundary values must be of len m!")
        else:
            qr = None
            ql = None

        dtype = next(self.parameters()).dtype
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        Qtot = torch.zeros((len(qx0),m)).to(device)
        Q = torch.tensor(qx0,dtype=dtype).reshape((1,1,-1)).to(device)

        with torch.no_grad():
            for ti in range(m):
                if qr is not None:
                    #Optionally Hardly replace extremities by the given boundary condition
                    Q[:,:,0] = ql[ti]  
                    Q[:,:,-1] = qr[ti]  
                Q = self.forward(Q)
                Qtot[:,ti] = Q.reshape(-1)
        del Q
        self.cpu()
        return Qdata(Qtot.detach().cpu().numpy(),qx0.dx,qx0.dt)

    def fit(
        self,
        Qdatas: List[Qdata],
        timesteps: int = 0,
        epochs: int = 10,
        batch_size: int = 0,
        learning_rate: float = 1e-5 ,
        k1: int = 100,
        decay: float = 1.,
        patience: int = 20,
        sliceData: bool = False,
        loss: str = "mse",
        dev: str = None 
        )-> List[dict]:
        """ Trains the model on several timsteps predictions using decayed-Truncated Backpropagation through time. 
        The Qdata provided will be split in the right amount of time slices for training 
        
        Args:
            Qdatas: Qdata or List[Qdata] the data to train on.
            timesteps (int): the number of timesteps to train on.
            learning_rate (float): the initial LR to use
            loss (str): 'mae' by dfault or 'mse' : the name of the loss to use
        
        Returns:
            The history of training as a List of dicts
        """
        #sanity checks
        if type(Qdatas) is not list: Qdatas = [ Qdatas ]
        
        if timesteps > Qdatas[0].shape[1]:
            raise ValueError(f"Cannot build dataset of {timesteps} timesteps with grid timesteps of {Qdatas[0].shape[1]} !")
        if not timesteps: timesteps = Qdatas[0].shape[1]

        #GPU cleaning before tryping to allocate:
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

        #build numpy dataset
        if sliceData:
            # separate training data in slices of len(x) x timestemps
            N_slices = Qdatas[0].shape[1] // timesteps
            Qbuff =[]
            for qData in Qdatas:
                for i in range(N_slices):
                    Qbuff.append(qData[:,timesteps*i : timesteps*(i+1)])
            Qdatas = Qbuff
        else:
            # only truncate the first timesteps of training data solutions 
            Qdatas= [qData[:,:timesteps] for qData in Qdatas]

        
        #perform normal training on the module and dataset
        hist = TBPTT(
            self,
            Qdatas,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            k1=k1,
            decay=decay,
            patience = patience,
            lossName = loss,
            dev = dev
            )
        
        return hist

    def __str__(self)->str:
        return ("\tTorch module implementing Trafsyn Model interface \n"
        + super(nn.Module, self).__str__() 
        + f"\n parameters: {self.count_parameters()}"
        + f"\n grid dx:{self.dx} dt:{self.dt}"
        )
    
    def __getattribute__(self,attr):
        if attr=='p':
            return self.state_dict()
        elif attr=='dtype':
            return next(self.parameters()).dtype
        else:
            return object.__getattribute__(self, attr)
    
    def count_parameters(self)->int:
        """Prints the number of parameters of the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

######################## TRAINING FUNCTIONS ########################

# training functions
def trainModel(
    model: nn.Module,
    dataset: QDataset,
    batch_size: int = 1,
    epochs: int = 10,
    learning_rate: float = 1e-5,
    loss_memory: int = None,
    patience: int = 50,
    dev: str = None,
    lossName: str = "mse",
    ):
    """Trains the model on the dataset using Adam optimizer
    
    Args:
        model (nn.Module)    : the model to train
        dataset (QDataset)   : dataset to train on
        timesteps (int)      : Number of timesteps
        batch_size (int)     : batch size for parallelization,
        epochs (int)         : number of passes on the training data,
        learning_rate (float): number of passes on the training data
        dev (string)         : 'cuda' or 'cpu device to use (default cuda if available)

    """
    # Device management
    if dev:
        device = torch.device(dev)
        print('using ' + dev + ' device')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("using GPU :",torch.cuda.get_device_name(0))
        else: 
            device = torch.device('cpu')
            print('using cpu device')
    #clean GPU before allocating anything
    gc.collect()
    torch.cuda.empty_cache()

    #loss function to use
    if lossName == "mse":
        loss_fx = F.mse_loss
    elif lossName == "mae":
        loss_fx = F.l1_loss
    else :
        print("loss must be either 'mse' or 'mae': default to mse")
        loss_fx = F.mse_loss

    if not loss_memory:
        loss_memory = dataset[0][1].shape[0]

    print(f"Train model on {lossName} with batch_size {str(batch_size)} and memory {loss_memory}")
    
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []

    #LR scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode = 'min',
        patience = patience,
        cooldown = 2*patience,
        min_lr=1e-8,
        verbose = True)

    x_len = dataset[0][1].shape[1]
    t_len = dataset[0][1].shape[0]

    avgPred = 0
    avgOpt = 0
    cons_loss = 0
    neg_cost = 0

    tbeg = time.time()
    # Start training
    for epoch in range(epochs):
        for i, (Q_t, Q_nxt) in enumerate(dataloader):

            # Forward pass
            Q_t = Q_t.to(device)
            Q_nxt = Q_nxt.to(device)
            Q_pred = model.forward(Q_t)

            # compute loss
            loss = loss_fx(Q_pred[:,-loss_memory:],Q_nxt[:,-loss_memory:])

            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step(loss)
        history.append(
            {   
                'epoch': epoch,
                'loss': loss.item(),
                'LR': optimizer.param_groups[0]['lr'],
            }
        )
        if optimizer.param_groups[0]['lr'] < 1e-7:
            print ("Epoch[{}/{}], Step [{}/{}], {} Loss: {:.6f},LR: {:.3e}" 
                    .format(epoch+1, epochs, i+1, len(dataloader),lossName, loss, optimizer.param_groups[0]['lr']))
            print("Stopping ...")
            break
        
        if epoch % int(epochs/10+1) == 0:
            print ("Epoch[{}/{}], Step [{}/{}], {} Loss: {:.6f},LR: {:.3e}" 
                    .format(epoch+1, epochs, i+1, len(dataloader),lossName, loss, optimizer.param_groups[0]['lr']))
    
    elapsed = time.time() - tbeg
    print(f"Performed {epoch} epochs in {elapsed :.2}s: {epoch/elapsed:.3} epochs/s")

    #clean GPU
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return history

# TBPTT IMPLEMENTATION
def TBPTT(
    model: torchModel,
    Qdatas: List[Qdata],
    k1 :int = 100,
    timesteps: int = 0,
    batch_size: int = None,
    epochs: int = 10,
    optimizer: torch.optim.Optimizer = torch.optim.Adam,
    learning_rate: float = 1e-5,
    decay: float = 1.,
    clip = None,
    patience: int = 50,
    dev: str = None,
    lossName: str = "mse",
    ):
    """Trains the model on the dataset using Adam optimizer
    
    Args:
        model (nn.Module)    : the model to train
        dataset (QDataset)   : dataset to train on
        timesteps (int)      : Number of timesteps
        batch_size (int)     : batch size for parallelization,
        epochs (int)         : number of passes on the training data,
        learning_rate (float): number of passes on the training data
        dev (string)         : 'cuda' or 'cpu device to use (default cuda if available)

    """
    #Sanity checks 
    if type(Qdatas) is not list: Qdatas = [ Qdatas ]
    if not timesteps: timesteps = Qdatas[0].shape[1]
    else:
        if timesteps > Qdatas[0].shape[1]:
            raise ValueError(f"can't train on {timesteps} timesteps with {Qdatas[0].shape[1]} timesteps in data !")
    if not batch_size: batch_size = len(Qdatas)
    if lossName == "mse":
        loss_fx = F.mse_loss
    elif lossName == "mae":
        loss_fx = F.l1_loss
    else :
        print("loss must be either 'mse' or 'mae': default to mse")
        loss_fx = F.mse_loss


    # Device management
    if dev:
        device = torch.device(dev)
        print('using ' + dev + ' device')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("using GPU :",torch.cuda.get_device_name(0))
        else: 
            device = torch.device('cpu')
            print('using cpu device')
    
    #build dataset
    Q0 = [qData[:,0] for qData in Qdatas]
    Qnexts = [qData[:,1:timesteps] for qData in Qdatas]
    
    #create torch dataset from lists
    dataset = QDataset(
                    Q0 = Q0,
                    Qnexts = Qnexts,
                    dtype=next(model.parameters()).dtype #cast dataset as params dtype
                    ) 

    #clean GPU before allocating anything
    gc.collect()
    torch.cuda.empty_cache()
    
    # Training variables
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    history = []

    #LR scheduler
    min_lr = 5e-9
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode = 'min',
        patience = patience,
        factor=0.5,
        cooldown = 2*patience,
        eps = 0.1*min_lr,
        min_lr=min_lr,
        verbose = True)

    tbeg = time.time()
    decays = [decay**(timesteps-2-j) for j in range(timesteps-1)]
    print(f"Train model with TBPTT: Nts {timesteps} | loss {lossName} | b_size {str(batch_size)} | k1 {k1} | decay {decay} {'| clip ' + str(clip) if clip else ''}")
    # Start training
    for epoch in range(epochs):
        for i, (Q_t, Q_nxt) in enumerate(dataloader):
        # Forward pass
            Q_t = Q_t.to(device)
            Q_nxt = Q_nxt.to(device)
            loss = 0

            for j in range(timesteps-1):
                Q_t = model.forward(Q_t)
                loss +=  loss_fx(Q_t,Q_nxt[:,j:j+1,:]) * decays[j]
                #backward every k1 timesteps an release CUDA memory for longer inference
                if (timesteps - 2 - j )%k1 == 0:
                    optimizer.zero_grad()
                    loss.backward()
                    if clip: nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                    Q_t.detach_()
                    loss.detach_()

        scheduler.step(loss)
        history.append(
            {   
                'epoch': epoch,
                'loss': loss.item(),
                'LR': optimizer.param_groups[0]['lr'],
            }
        )
        if optimizer.param_groups[0]['lr'] <= min_lr:
            print ("Epoch[{}/{}], Step [{}/{}], {} Loss: {:.6f},LR: {:.3e}" 
                    .format(epoch+1, epochs, i+1, len(dataloader),lossName, loss, optimizer.param_groups[0]['lr']))
            print("Stopping ...")
            break
        
        if epoch % int(epochs/10+1) == 0:
            print ("Epoch[{}/{}], Step [{}/{}], {} Loss: {:.6f},LR: {:.3e}" 
                    .format(epoch+1, epochs, i+1, len(dataloader),lossName, loss, optimizer.param_groups[0]['lr']))
    
    elapsed = time.time() - tbeg
    print(f"Performed {epoch+1} epochs in {elapsed :.2}s: {(epoch+1)/elapsed:.3} epochs/s")

    #clean GPU
    model.cpu()
    del dataloader
    del optimizer
    del scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return history

# BPTT IMPLEMENTATION
def BPTT(
    model: torchModel,
    Qdatas: List[Qdata],
    timesteps: int = 0,
    batch_size: int = None,
    epochs: int = 10,
    optimizer: torch.optim.Optimizer = torch.optim.SGD,
    learning_rate: float = 1e-5,
    patience: int = 100,
    clip = None,
    dev: str = None,
    lossName: str = "mse",
    ):
    """Trains the model on the dataset using Adam optimizer
    
    Args:
        model (nn.Module)    : the model to train
        dataset (QDataset)   : dataset to train on
        timesteps (int)      : Number of timesteps
        batch_size (int)     : batch size for parallelization,
        epochs (int)         : number of passes on the training data,
        learning_rate (float): number of passes on the training data
        dev (string)         : 'cuda' or 'cpu device to use (default cuda if available)

    """
    # Qdata Sanity check 
    if type(Qdatas) is not list: Qdatas = [ Qdatas ]

    # Device management
    if dev:
        device = torch.device(dev)
        print('using ' + dev + ' device')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("using GPU :",torch.cuda.get_device_name(0))
        else: 
            device = torch.device('cpu')
            print('using cpu device')
    
    #build dataset
    Q0 = [qData[:,0] for qData in Qdatas]
    Qnexts = [qData[:,1:timesteps] for qData in Qdatas]
    
    #create torch dataset from lists
    dataset = QDataset(
                    Q0 = Q0,
                    Qnexts = Qnexts,
                    dtype=next(model.parameters()).dtype #cast dataset as params dtype
                    ) 
    
    # if no Batch size given, maximize parallelization
    if not batch_size:
        batch_size = len(dataset)
    
    #loss function to use
    if lossName == "mse":
        loss_fx = F.mse_loss
    elif lossName == "mae":
        loss_fx = F.l1_loss
    else :
        print("loss must be either 'mse' or 'mae': default to mse")
        loss_fx = F.mse_loss


    print(f"Train model with BPTT: Nts {timesteps} | loss {lossName} | b_size {str(batch_size)}")
    
    #clean GPU before allocating anything
    gc.collect()
    torch.cuda.empty_cache()
    
    # Training variables
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    history = []

    #LR scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode = 'min',
        patience = patience,
        cooldown = 2*patience,
        min_lr=1e-8,
        verbose = True)

    tbeg = time.time()
    # Start training
    for epoch in range(epochs):
        for i, (Q_t, Q_nxt) in enumerate(dataloader):
        # Forward pass
            Q_pred = torch.empty_like(Q_nxt,dtype=Q_nxt.dtype).to(device)
            Q_t = Q_t.to(device)
            Q_nxt = Q_nxt.to(device)

            for n in range(timesteps-1):
                Q_t = model.forward(Q_t)
                Q_pred[:,n:n+1,:] = Q_t
                
            #compute loss 
            loss = loss_fx(Q_pred,Q_nxt)
            
            #optimize
            optimizer.zero_grad()
            if clip: nn.utils.clip_grad_norm_(model.parameters(), clip)
            loss.backward()
            optimizer.step()

        scheduler.step(loss)
        history.append(
            {   
                'epoch': epoch,
                'loss': loss.item(),
                'LR': optimizer.param_groups[0]['lr'],
            }
        )
        if optimizer.param_groups[0]['lr'] < 1e-7:
            print ("Epoch[{}/{}], Step [{}/{}], {} Loss: {:.6f},LR: {:.3e}" 
                    .format(epoch+1, epochs, i+1, len(dataloader),lossName, loss, optimizer.param_groups[0]['lr']))
            print("Stopping ...")
            break
        
        if epoch % int(epochs/10+1) == 0:
            print ("Epoch[{}/{}], Step [{}/{}], {} Loss: {:.6f},LR: {:.3e}" 
                    .format(epoch+1, epochs, i+1, len(dataloader),lossName, loss, optimizer.param_groups[0]['lr']))
    
    elapsed = time.time() - tbeg
    print(f"Performed {epoch+1} epochs in {elapsed :.2}s: {(epoch+1)/elapsed:.3} epochs/s")

    #clean GPU
    model.cpu()
    del dataloader
    del optimizer
    del scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return history

#################################### MISCELLANEOUS ####################################

# Random initialization of weights
def weights_init_uniform(m):
    if isinstance(m, nn.Conv1d):
        # nn.init.xavier_uniform_(m.weight.data, gain=1)
        # nn.init.orthogonal_(m.weight.data, gain=1)
        # nn.init.constant_(m.weight.data, 0)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        # nn.init.xavier_uniform_(m.weight.data, gain=1)
        # nn.init.orthogonal_(m.weight.data, gain=1)
        # nn.init.constant_(m.weight.data, 0)
        nn.init.constant_(m.bias.data, 0)

#prediction using the torch module
def predict(model: nn.Module,
            Q_init: np.array,
            N_steps: int,
            qr: Optional[np.array] = None,
            ql: Optional[np.array] = None,
            ) -> np.array:
    """ Make a prediction of the traffic with initial state Q_init over N_steps timesteps
    Args:
        Q_init (np.array): the initial density data
        N_steps (int)       : The number of timesteps to predict
        qr (Optional: np.array with len m) : The right boundary condition, if None: free flow BC
        qr (Optional: np.array with len m) : The left boundary condition, if None: free flow BC

    Returns:
        np.array : The array of the density with shape ( len(Q_init),   N_steps ) 
    """
    #BC checkup
    if qr is not None and ql is not None:
        BC = True
        qr = torch.tensor(qr)
        ql = torch.tensor(ql)
        if len(qr) != N_steps or len(ql) != N_steps: 
            raise ValueError("Boundary values must be of len N_steps!")
    else:
        qr = None
        ql = None
        BC = False

    #translate to torch tensor
    Q = torch.from_numpy(Q_init).unsqueeze(0).unsqueeze(0).cpu()
    model = model.cpu()
    #predict sequentially
    Q_pred = np.zeros((Q_init.shape[0], N_steps))
    with torch.no_grad():
        for i in range(N_steps):
            #if Hard BC, store value in ghost cell
            if BC: 
                Q = torch.cat(
                    (ql[i].reshape((1,1,1)), Q, qr[i].reshape((1,1,1))), 
                    dim = -1
                )
            #make prediction
            Q = model(Q)
            #store solution and remove ghost cell if hard BC
            if BC:
                Q_pred[:,i] = (Q.squeeze(0).squeeze(0).detach().cpu().numpy())[1:-1]
                Q = Q[:,:,1:-1]
            else:
                Q_pred[:,i] = (Q.squeeze(0).squeeze(0).detach().cpu().numpy())

    return np.array(Q_pred)

def evaluateModule(model: nn.Module,
                  data: Qdata,
                  plot: bool = True,
                  printModel: bool = False,
                  xlim: list = None,
                ) -> None:
    """displays information about the module"""
    
    print("torch Module Eval ...")
    if printModel:
        print(model,'\n')
        
    print("Number of parameters:",count_parameters(model),'\n')

    q_pred = predict(model, data[:,0], data.shape[1] - 1)
    MAE = np.mean( np.abs(q_pred - data[:,1:]))
    print("Mean Absolute error of model on data: ", MAE)

    if plot:
        ComparePlots(
            data[:,1:],
            q_pred,
            dx = data.dx,
            dt = data.dt,
            title1 = "Ground truth",
            title2 = "model prediction",
            xlim=xlim,
            )
        
        t = np.linspace(0, 1, data.shape[1]-1)
        x = np.linspace(0, 2, data.shape[0]+1)

        # plot the data
        pcolormesh(t, x, np.abs(q_pred - data[:,1:]),
                   xlim=xlim,
                   title="absolute error",
                   show=True)
    return

# counting parameters
def count_parameters(module: nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def GetFingerPrint(t_flowFx, rmax:float, N:int=100)->Qdata:
    """Draw (0,1) fingerPrint of any torch (a,b)-function
    Args:
        t_flowfx    : A torch module that outputs numerical fluxes from a torch tensor containing densities
        rmax (float): the maximum density will compute flows for (r1,r2) in [0,rmax].
        N (int) will sample N² points in [0,rmax]²
    Returns:
        fingerPrint (np.array) the resulting flow as a function of (Q_i,Q_i+1)
    """
    Nx = N
    X = np.linspace(0,rmax,Nx)
    dx = rmax/(Nx-1)
    pts = np.array([[x1,x2] for x1 in X for x2 in X])
    coords =  np.array([[Nx - 1 - i2,i1] for i1 in range(Nx) for i2 in range(Nx)])
    
    t_pts = torch.tensor(pts).unsqueeze(1)
    flows = t_flowFx(t_pts).detach().squeeze(1).numpy()[:,0]
    fingerPrint = np.zeros((Nx,Nx))
    for i,coord in enumerate(coords):
        i1,i2 = coord
        fingerPrint[i1,i2] = flows[i]
    return Qdata(fingerPrint,dx,dx)

#################################### PLOTTING FUNCTIONS ####################################
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

# fancy plot of training history 
def PlotHistory(history):
    
    df = pd.DataFrame(history)
    fig, ax1 = plt.subplots()

    ax1.plot(df['loss'], label='loss',color='red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('loss', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.plot(df['LR'], 'b--',linewidth=.5, label='learning rate')
    ax2.set_yscale('log')  # Set logarithmic scale for the second y-axis
    ax2.set_ylabel('LR', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Adding legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='best')

    fig.tight_layout()
    plt.title("Training history")
    plt.show()

#Plot fingerprints
def DrawFingerPrint(
                    fingerPrints:List[np.array],
                    rmax:float = 4,
                    titles: List[str]=None,
                    cmap:str = 'plasma',
                    *args,**kwargs):
    """draw the fingerprint of a given set of numerical (0,1)stencil flux functions
    Args: 
        fingerPrints: a list of np.arrays
    """
    # plt.rcParams.update({'font.family': 'sans-serif'})  # Set font family
    plt.rcParams.update({"mathtext.default": 'it'})  # Set font family

    N = len(fingerPrints)
    x_scale = [0,rmax]

    
    if N >= 4:
        num_rows = 2
        num_cols = int(N/2)
    else:
        num_rows = 1
        num_cols = N
    #plotting
    fig, axs = plt.subplots(num_rows, num_cols + 1, figsize=(num_cols * 4 + 1, num_rows * 4), gridspec_kw={"width_ratios": [1] * num_cols + [0.08]})
    
    if N >= 4:
        gs = axs[0,-1].get_gridspec()
        cax = fig.add_subplot(gs[:, -1])
        axs[-1,-1].remove()
        axs[0,-1].remove()
        axs = axs[:,:-1].flatten()
    else:
        cax = axs[-1]

    vmax = max([np.max(fp) for fp in fingerPrints])
    vmin = min([np.min(fp) for fp in fingerPrints])

    for i, img in enumerate(fingerPrints):
        im = axs[i].imshow(img,
                    vmin=vmin,
                    vmax=vmax,
                    extent=[x_scale[0], x_scale[1], x_scale[0], x_scale[1]],
                    cmap=cmap,
                    *args, **kwargs)
        axs[i].set_xlabel(r'$\rho_i$', fontdict={'fontsize': 15})
        axs[i].set_ylabel(r'$\rho_{i+1}$', fontdict={'fontsize': 15})
        if titles:
            axs[i].set_title(titles[i])

    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(r'$\mathcal{F}(\rho_{i},\rho_{i+1})$',fontdict={'fontsize': 15})
    
    plt.tight_layout()