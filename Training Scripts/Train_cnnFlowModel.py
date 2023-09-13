import pickle
import trafsyn
from trafsyn.utils import *
from trafsyn.torchUtils import *
from trafsyn.torchModels import *
from trafsyn.Baselines.LWR import GenerateGrRiemannTrainingSet

########################################################################
################ Generate Training Set : Riemann Waves ################
########################################################################

args = {
    #Greenshield variables
    "v" : 1,
    "rmax" : 4,
    #grid params of Training Set
    "dx" : .01,
    "dt" : .01,
    "N"  : 15,    # Will generate N(N-1) waves
    "Nts": 400,   # time horizon of generated solutions
}
#generate or load Training Set
TrainSet = GenerateGrRiemannTrainingSet(**args,plot=False,save=1)

CFL = args['v'] * args['dt'] / args['dx']
print(f"Using training set containing {len(TrainSet)} solutions of shape {TrainSet[0].shape}, dx={args['dx']}, dt={args['dt']}, CFL={CFL}")

#Filter only rarefaction 
Rarefactions = []
for data in TrainSet:
    if data[0,0] > data[-1,0]:
        Rarefactions.append(data)

#utils to reduce the time horizon of the training set
def ReduceTS(RiemannData,Nts):
    """Assume CFL=1, and vmax = 1"""
    Reduced = []
    Nx = RiemannData[0].shape[0]
    border = max(int((Nx-2*Nts)/2) - 1 , 0)
    for data in RiemannData:
        Reduced.append(data[border:-(border+1),:Nts])
    return Reduced


########################################################################
###############################  Create model ##########################
########################################################################

dtype= torch.float32
init = None
#init = "checkpoint.pth"

torch.manual_seed(1)
if init: RModel = loadTorchModel(init,dtype=dtype)
else:
    RModel = cnnFlowModel((0,1),
                                depth=6,
                                hidden=15,
                                dx = TrainSet[0].dx,
                                dt = TrainSet[0].dt,
                                activation=torch.nn.ELU,
                                dtype=dtype)
 
print("Training model:", RModel)

########################################################################
###############################   TRAINING    ##########################
########################################################################

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
torch.save(RModel.state_dict(),f"cnnFLOW_100.pth")

#focus on rarefactions that are more difficult to get right
for nts in [100,200,300]:
    h += TBPTT(
        RModel,
        ReduceTS(Rarefactions,nts),
        k1=100,
        decay=0.99,
        epochs=10000,
        patience=50,
        learning_rate=1e-5,
        optimizer = torch.optim.Adam,
    )
    torch.save(RModel.state_dict(),f"cnnFLOW_r{nts}.pth")

#further training on larger horizon
for nts in [100,200,300]:
    h += TBPTT(
        RModel,
        ReduceTS(TrainSet,nts),
        k1=100,
        decay=0.99,
        epochs=10000,
        patience=50,
        learning_rate=1e-5,
        optimizer = torch.optim.Adam,
    )
    torch.save(RModel.state_dict(),f"cnnFLOW_{nts}.pth")

#finetunning with higher precision and gradient clipping
Rmodel = RModel.double()
for nts in [400,400]:
    h += TBPTT(
        RModel,
        ReduceTS(TrainSet,nts),
        k1=100,
        decay=0.99,
        epochs=3000,
        patience=30,
        clip=1,
        learning_rate=1e-7,
        optimizer = torch.optim.Adam,
    )

# save Model
torch.save(RModel.state_dict(),f"cnnFLOW_OUT.pth")

# save history
with open("history.pkl", 'wb') as file:
    pickle.dump(h, file)

