import torch, gc, os, pickle
import numpy as np
import trafsyn
from trafsyn.utils import *
from trafsyn.plot import *
from trafsyn.torchUtils import *
from trafsyn.torchModels import *
from trafsyn.Baselines.LWR import *
from trafsyn.Baselines.NumSchemes import *

#load all datasets
dataTR = loadData("Triangular.npy")

######################## Training Dataset ######################## 
args = {
    #Greenshield variables
    "v_c" : 1,
    "r_c" : 1,
    "rmax" : 2,
    #grid params of Training Set
    "dx" : .01,
    "dt" : .01,
    "N"  : 15,   #generate N(N-1) Riemann waves 
    "Nts": 400,   #number of ts
}

#generate or load Training Set
TrainSet = GenerateTriRiemannTrainingSet(**args,plot=True,save=0)

#Filter only rarefaction 
Rarefactions = []
for data in TrainSet:
    if data[0,0] > data[-1,0]:
        Rarefactions.append(data)

def ReduceTS(RiemannData,Nts):
    """Assume CFL=1, and vmax = 1"""
    Reduced = []
    Nx = RiemannData[0].shape[0]
    border = max(int((Nx-2*Nts)/2) - 1 , 0)
    for data in RiemannData:
        Reduced.append(data[border:-(border+1),:Nts])
    return Reduced

######################## CREATE MODEL ######################## 
dtype= torch.float32
init = None

torch.manual_seed(1)
if init: RModel = loadTorchModel(init,dtype=dtype)
else:
    # reinitialize model parameters if we get 0 in output of flow model
    RModel = cnnSpeedModel((0,1),
                                depth=6,
                                hidden=15,
                                dx = TrainSet[0].dx,
                                dt = TrainSet[0].dt,
                                activation=torch.nn.ELU,
                                dtype=dtype)

print("Training model:", RModel)

######################## TRAINING ######################## 

h = []
#First fit using float precision for faster training
for nts in [50,100]:
    h += TBPTT(RModel,
        ReduceTS(TrainSet,nts),
        epochs=30000,
        k1=50,
        decay=0.99,
        patience=100,
        learning_rate=1e-4,
        optimizer = torch.optim.Adam,
    )
torch.save(RModel.state_dict(),f"cnnSPEED_TR_100.pth")

#focus on rarefactions that are more difficult to get right
for nts in [100,200,300]:
    h += TBPTT(
        RModel,
        ReduceTS(Rarefactions,nts),
        k1=50,
        decay=0.99,
        epochs=10000,
        patience=50,
        learning_rate=1e-5,
        optimizer = torch.optim.Adam,
    )
    torch.save(RModel.state_dict(),f"cnnSPEED_TR_r{nts}.pth")

#further training on larger horizon
for nts in [100,200,300]:
    h += TBPTT(
        RModel,
        ReduceTS(TrainSet,nts),
        k1=50,
        decay=0.99,
        epochs=10000,
        patience=50,
        learning_rate=1e-5,
        optimizer = torch.optim.Adam,
    )
    torch.save(RModel.state_dict(),f"cnnSPEED_TR_{nts}.pth")

#finetunning with higher precision and gradient clipping
Rmodel = RModel.double()
for nts in [400,400]:
    h += TBPTT(
        RModel,
        ReduceTS(TrainSet,nts),
        k1=50,
        decay=0.99,
        epochs=3000,
        patience=30,
        clip=1,
        learning_rate=1e-7,
        optimizer = torch.optim.Adam,
    )
    
# save Model
torch.save(RModel.state_dict(),f"cnnSPEED_TR_OUT.pth")

# save history
with open("history.pkl", 'wb') as file:
    pickle.dump(h, file)
