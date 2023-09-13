import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Tuple, Optional, List

from trafsyn.torchUtils import *
from trafsyn.utils import Model, Qdata

########################### 'Sub Modules' used as base bricks ###########################

# Base ab-stencil CNN for cnn density modules
class ab_stencil_CNN(nn.Module):
    def __init__(self,
                 stencil: Tuple[int,int],
                 hidden: int = 20,
                 depth: int = 1,
                 activation = nn.ReLU,
                 dtype: torch.dtype = torch.float64):
        """CNN with a (a,b) assymmetric kernel size"""
        super(ab_stencil_CNN, self).__init__()

        try:
            a, b = stencil
        except:
            raise ValueError("The stencil must be (a:int, b:int)")
        
        #store stencil as buffer
        self.register_buffer('stencil',torch.tensor(stencil))
        self.register_buffer(type(activation()).__name__,torch.zeros((1)))

        #add convolution layers in sequential module
        self.conv = nn.Sequential()
        self.conv.add_module( "Conv1",
            nn.Conv1d(  in_channels=1,
                        out_channels=hidden,
                        kernel_size=(a+b+1),
                        padding = 0,
                        dtype=dtype)  )
        self.conv.add_module("act1", activation() )
        for i in range(depth-1):
            #the hidden FC layers
            self.conv.add_module( "fc_"+str(i),
                nn.Conv1d(  in_channels=hidden,
                        out_channels=hidden,
                        kernel_size=1,
                        padding = 0,
                        dtype=dtype)
                        )
            self.conv.add_module("relu"+str(i), activation() )
        #final layers
        self.conv.add_module( "fc_end",
            nn.Conv1d(  in_channels=hidden,
                        out_channels=1,
                        kernel_size=1,
                        padding = 0,
                        dtype=dtype) )
        self.conv.add_module("end_relu", nn.ReLU() )
        
    def forward(self, x):
        #Pads the input tensor before convolution: 
        # Here we impose Free Flow condition (dF/dx = 0) at both L/R boundaries
        a,b = self.stencil
        x = torch.cat((
                            x[:,:,0].unsqueeze(-1).repeat(1,1,a),
                            x,
                            x[:,:,-1].unsqueeze(-1).repeat(1,1,b),
                            ) , dim=2
                        ,
                        ).to(x.device)

        # x = F.pad( x, pad=(self.a,self.b) )
        return self.conv(x)
    
# Base Conv module that convolve any module on 1D input
class ConvModule(nn.Module):
    def __init__(self,
                module: nn.Module,
                stencil: Tuple[int,int]):
        super(ConvModule,self).__init__()
        try:
            a, b = stencil
        except:
            raise ValueError("The stencil must be (a:int, b:int)")  
        self.a = a
        self.b = b
        self.module = module
        dtype = next(module.parameters()).dtype
        self.register_buffer('stencil',torch.tensor(stencil))
        self.register_buffer('CNNIdWeight',torch.eye(a+b+1,dtype=dtype).unsqueeze(1))
    
    def forward(self, x):
        #Pads the input tensor before convolution: 
        # Here we impose Free Flow condition (dF/dx = 0) at both L/R boundaries
        x = torch.cat((
                            x[:,:,0].unsqueeze(-1).repeat(1,1,self.a),
                            x,
                            x[:,:,-1].unsqueeze(-1).repeat(1,1,self.b),
                            ) , dim=2
                        ).to(x.device)
        #Convolve solution
        #we use transpose to switch the channels and the last dimension
        x =  nn.functional.conv1d(x,self.CNNIdWeight,bias=None).transpose(1,2).to(x.device)
        
        # print("convolution:",x,x.size())
        # Apply the module on each
        x = self.module(x)
        return x.reshape(len(x),1,-1)

########################### Torch Modules ###########################

# Dummy ab-stencil CNN model
class abModel(torchModel):
    """
        Dummy module that uses a (a,b)-stencil dense neural network to predict the next state's density 
         - this module is discretization-specific
    """
    def __init__(self, 
                stencil: Tuple[int,int],
                depth: int = 2,
                hidden: int = 10,
                dx: float = 1.,
                dt: float = 1.,
                ):
        #register dx and dt at parent class level, 
        super(abModel, self).__init__(dx,dt)
        self.ABmodule = ab_stencil_CNN(stencil,hidden,depth)
    
    def forward(self,x):
        """Directly predicts the next state density without constaints"""        
        return self.ABmodule.forward(x)

# Speed FC module convolved with convModule
class nnSpeedModel(torchModel):
    """
        Module that uses a (a,b)-stencil module to predict the vehicles speed in the cells
        Computes the flow using flow = density * speed 
        and finally predicts the next state's density 
    """

    def __init__(self, 
                stencil: Tuple[int,int],
                depth: int = 2,
                hidden: int = 10,
                dx: float = 1.,
                dt: float = 1.,
                dtype=torch.float64,
                ):
        #register dx and dt at parent class level
        super(nnSpeedModel, self).__init__(dx,dt)
        a,b = stencil
        
        #register stencil as non trainable param of speedModel
        self.stencil = stencil
        act_fx = nn.ReLU() 

        ######### Create FC NN as a Speed model #########
        fcModule = nn.Sequential()
        # First FC layer
        fcModule.add_module( "fc1",
            nn.Linear(  in_features=(a+b+1),
                        out_features=hidden, dtype=dtype) )
        fcModule.add_module("act1", act_fx)        
        # Hidden FC layers
        for i in range(depth-1):
            fcModule.add_module( "fc_"+str(i),
                nn.Linear(  in_features=hidden,
                        out_features=hidden, dtype=dtype))
            fcModule.add_module("relu"+str(i), act_fx)
        # Final layer
        fcModule.add_module( "fc_end",
            nn.Linear(  in_features=hidden,
                        out_features=1, dtype=dtype) )
        fcModule.add_module("end_relu", nn.ReLU() )
        fcModule.apply(weights_init_uniform)

        #Pass the fc module to get a convolution of it with given stencil
        self.speedModel = ConvModule(module= fcModule, stencil= self.stencil)
    
    def forward(self,x):
        #Predict right flow
        flowR = torch.mul( x, self.speedModel(x))
        flowR =  torch.cat((
                            flowR[:,:,0].unsqueeze(-1),
                            flowR,
                            ) , dim=2
                        ).to(x.device)
        #add flows to cells
        return x - (self.dt/self.dx) * (flowR[:,:,1:] - flowR[:,:,:-1])

# Same but Faster model using the ab_stencil_cnn
class cnnSpeedModel(torchModel):
    """
        Module that uses a (a,b)-stencil cnn module to predict the vehicles speed in the cells
        Computes the flow using flow = density * speed 
        and finally predicts the next state's density 
    """

    def __init__(self, 
                stencil: Tuple[int,int],
                depth: int = 2,
                hidden: int = 10,
                dx: float = 1.,
                dt: float = 1.,
                activation = nn.ReLU,
                dtype=torch.float64,
                ):
        #register dx and dt at parent class level
        super(cnnSpeedModel, self).__init__(dx,dt)

        #create a convolution module with given parameters
        self.speedModel = ab_stencil_CNN(stencil, hidden, depth, activation=activation, dtype=dtype)
     
    def forward(self,x):
        #Predict right speed and compute flow
        flowR = torch.mul( x, self.speedModel(x))
        flowR =  torch.cat((
                            flowR[:,:,0].unsqueeze(-1),
                            flowR,
                            ) , dim=2
                        ).to(x.device)
        #add flows to cells
        return x - (self.dt/self.dx) * (flowR[:,:,1:] - flowR[:,:,:-1])

# Same but Faster model using the ab_stencil_cnn
class cnnFlowModel(torchModel):
    """
        Module that uses a (a,b)-stencil cnn module to predict the vehicles Flow Between cells
        and then predicts the next state's density using a Finite Volume method scheme
    """

    def __init__(self, 
                stencil: Tuple[int,int],
                depth: int = 2,
                hidden: int = 10,
                dx: float = 1.,
                dt: float = 1.,
                activation = nn.ReLU,
                dtype=torch.float64,
                ):
        #register dx and dt at parent class level
        super(cnnFlowModel, self).__init__(dx,dt)

        #Pass the fc module to get a convolution of it with given stencil
        self.flowModel = ab_stencil_CNN(stencil, hidden, depth, activation=activation, dtype=dtype)
    
    def forward(self,x):
        #Predict right flow
        flowR = self.flowModel(x)
        # BOUNDARY CONDITION
        # Define left entering flow:  we use here the free flow condition
        # i.e the input flow of the first cell is equal to the output flow. 
        # Then the first cell stays constant in density 
        flowR =  torch.cat((
                            flowR[:,:,0].unsqueeze(-1),
                            flowR,
                            ) , dim=2
                        ).to(x.device)

        #add flows to cells
        return x - (self.dt/self.dx) * (flowR[:,:,1:] - flowR[:,:,:-1])

def loadTorchModel(filename, dtype=torch.float64, verbose = False):
    """Helper function that quickly loads a model from a state_dict checkpoint
    WARNING sets the model dx and dt default to 1 (CFL = 1) please update them for proper inference
    
    Args:
        filename (str): the saved checkpoint .pth file
        
    Returns:
        model (TorchModel): the model with loaded parameters
    """
    state_dict = torch.load(filename, map_location=torch.device('cpu'))
    activations = dir(torch.nn.modules.activation)
    keys = list(state_dict.keys())
    modelName = keys[0].split('.')[0]
    if verbose: print('model Name:',modelName) 
    
    #get activation fx
    activation = 'ReLU' #default ReLU
    for key in keys:
        name = key.split('.')[-1]
        if name in activations:
            activation = name
            if verbose: print(f"found activation fx : {name}")

    # depending on the type of model
    if modelName == 'speedModel':
        #it is a nnSeedModel OR cnnSpeedModel 
        stencil_key = 'speedModel.stencil'
        if len([key for key in keys if 'stencil' in key]):
            stencil = state_dict[stencil_key].numpy()
            if verbose: print(f"Found stencil of {stencil}")
        else: 
            a , b = filename.split('(')[1][:3].split(',')
            stencil = (int(a),int(b))
            state_dict[stencil_key] = torch.tensor(stencil)
            if verbose: print(f"Guessed stencil: {stencil}")
        weights_keys = [key for key in keys if 'weight' in key]
        hidden = state_dict[weights_keys[0]].shape[0]
        depth = len(weights_keys) -1
        if verbose: print(f"found module with hidden dim {hidden} and depth {depth}")

        args = {
            'stencil'   : stencil,
            'hidden'    : hidden,
            'depth'     : depth,
            'dx'        : 1,
            'dt'        : 1,
            'dtype'     : dtype,
        }
        if len([key for key in keys if 'CNNIdWeight' in key]):
            #This is a nnSpeedModel   
            model = nnSpeedModel(**args)
            if verbose: print("Created nnSpeedModel")  
        else:
            args["activation"] = getattr(torch.nn.modules.activation,activation)
            model = cnnSpeedModel(**args)
            if verbose: print("Created cnnSpeedModel")  
        
        #load params in model
        model.load_state_dict(state_dict)
        return model
    
    elif modelName == 'flowModel':
        #it is a nnSeedModel OR floweedModel 
        stencil_key = 'flowModel.stencil'
        if len([key for key in keys if 'stencil' in key]):
            stencil = state_dict[stencil_key].numpy()
            if verbose: print(f"Found stencil of {stencil}")
        else: 
            a , b = filename.split('(')[1][:3].split(',')
            stencil = (int(a),int(b))
            state_dict[stencil_key] = torch.tensor(stencil)
            if verbose: print(f"Guessed stencil: {stencil}")
        weights_keys = [key for key in keys if 'weight' in key]
        hidden = state_dict[weights_keys[0]].shape[0]
        depth = len(weights_keys) -1
        if verbose: print(f"found module with hidden dim {hidden} and depth {depth}")

        args = {
            'stencil'   : stencil,
            'hidden'    : hidden,
            'depth'     : depth,
            'dx'        : 1,
            'dt'        : 1,
            'dtype'     : dtype,
        }
        args["activation"] = getattr(torch.nn.modules.activation,activation)
        model = cnnFlowModel(**args)
        if verbose: print("Created cnnFlowModel")  
        
        #load params in model
        model.load_state_dict(state_dict)
        return model

    elif modelName == "ABmodule":
        #abModel
        #it is a nnSeedModel OR cnnSpeedModel 
        stencil_key = 'ABmodule.stencil'
        if len([key for key in keys if 'stencil' in key]):
            stencil = state_dict[stencil_key].cpu().numpy()
            if verbose: print(f"Found stencil of {stencil}")
        else: 
            a , b = filename.split('(')[1][:3].split(',')
            stencil = (int(a),int(b))
            state_dict[stencil_key] = torch.tensor(stencil)
            if verbose: print(f"Guessed stencil: {stencil}")
        weights_keys = [key for key in keys if 'weight' in key]
        hidden = state_dict[weights_keys[0]].shape[0]
        depth = len(weights_keys) -1
        if verbose: print(f"found module with hidden dim {hidden} and depth {depth}")
        args = {
            'stencil'   : stencil,
            'hidden'    : hidden,
            'depth'     : depth,
            'dx'        : 1,
            'dt'        : 1,
            'dtype'     : dtype,
        }
        model = abModel(**args)
        model.load_state_dict(state_dict)
        return model
        
    else:
        print("unknown model")
        return None   

## Deprecated
# Speed FC module convolved with convModule
class nnSpeedModule(nn.Module):
    """
        Module that uses a (a,b)-stencil module to predict the vehicles speed in the cells
        Computes the flow using flow = density * speed 
        and finally predicts the next state's density 
    """

    def __init__(self, 
                stencil: Tuple[int,int],
                depth: int = 2,
                hidden: int = 10,
                dx: float = 1.,
                dt: float = 1.,
                ):

        super(nnSpeedModule, self).__init__()
        a,b = stencil
        self.stencil = stencil
        self.dx = dx
        self.dt = dt
        act_fx = nn.ReLU() #nn.LeakyReLU(0.3)

        ######### Create FC NN as a Speed model #########
        fcModule = nn.Sequential()
        # First FC layer
        fcModule.add_module( "fc1",
            nn.Linear(  in_features=(a+b+1),
                        out_features=hidden).double() )
        fcModule.add_module("act1", act_fx)        
        # Hidden FC layers
        for i in range(depth-1):
            fcModule.add_module( "fc_"+str(i),
                nn.Linear(  in_features=hidden,
                        out_features=hidden).double())
            fcModule.add_module("relu"+str(i), act_fx)
        # Final layer
        fcModule.add_module( "fc_end",
            nn.Linear(  in_features=hidden,
                        out_features=1).double() )
        fcModule.add_module("end_relu", nn.ReLU() )
        fcModule.apply(weights_init_uniform)

        #Pass the fc module to get a convolution of it with given stencil
        self.speedModel = ConvModule(module= fcModule, stencil= self.stencil)

    def forward(self,x):
        
        #Predict right flow
        flowR = torch.mul( x, self.speedModel(x))

        # BOUNDARY CONDITION
        # Define left entering flow:  we use here the free flow condition
        #   i.e the input flow of the first cell is equal to the output flow. 
        #   Then the first cell stays constant in density 
        #   TODO: could implement other flow conditions (0,circling condition,etc...)
        flowR =  torch.cat((
                            flowR[:,:,0].unsqueeze(-1),
                            flowR,
                            ) , dim=2
                        ).to(x.device)

        #add flows to cells
        return x - (self.dt/self.dx) * (flowR[:,:,1:] - flowR[:,:,:-1])

# Speed FC module convolved with native Conv1D torch layers
class cnnFlowModule(nn.Module):
    """Models that embeds a (a,b)-stencil flow model"""
    def __init__(self, 
                stencil: Tuple[int,int],
                depth: int = 2,
                hidden: int = 10,
                dx: float = 1.,
                dt: float = 1.,
                ):
        super(cnnFlowModule, self).__init__()

        #the flow model computes the flow from i to i+1
        self.flowModel = ab_stencil_CNN(stencil,hidden=hidden,depth=depth)
        self.r = dt / dx

    def forward(self,x):

        #Predict right flow
        flowR =  self.flowModel(x)

        # BOUNDARY CONDITION
        #Define left entering flow:  we use here the free flow condition
        # i.e the input flow of the first cell is equal to the output flow. 
        # Then the first cell stays constant in density 
        # TODO: could implement other flow conditions (0,circling condition,etc...)
        flowR =  torch.cat((
                            flowR[:,:,0].unsqueeze(-1),
                            flowR,
                            ), dim=2
                        ).to(x.device)

        #add flows to cells
        return x - self.r * (flowR[:,:,1:] - flowR[:,:,:-1])

# Flow FC module convolved with native Conv1D torch layers
class cnnSpeedModule(nn.Module):
    """
        Model that uses a (a,b)-stencil module to predict the vehicles speed in the cells
        it then computes the flow using flow = density * speed 
        and finally predicts the next state's density 
    """

    def __init__(self, 
                stencil: Tuple[int,int],
                depth: int = 2,
                hidden: int = 10,
                dx: float = 1.,
                dt: float = 1.,
                ):
        super(cnnSpeedModule, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #the speed model computes the speed in cell i
        self.speedModel = ab_stencil_CNN(stencil,hidden=hidden,depth=depth)
        self.r = dt / dx


    def forward(self,x):
        
        #Predict right flow
        flowR = torch.mul( x, self.speedModel(x))

        # BOUNDARY CONDITION
        #Define left entering flow:  we use here the free flow condition
        # i.e the input flow of the first cell is equal to the output flow. 
        # Then the first cell stays constant in density 
        # TODO: could implement other flow conditions (0,circling condition,etc...)
        flowR =  torch.cat((
                            flowR[:,:,0].unsqueeze(-1),
                            flowR,
                            ) , dim=2
                        ).to(x.device)

        #add flows to cells
        return x - self.r * (flowR[:,:,1:] - flowR[:,:,:-1])

