import numpy as np
import mdtraj as md


#---------------- load trajectories: for example from simulation at higher temperature
import glob

def loadData(dataName, top):
    numpy_vars = []
    #for np_name in glob.glob('/Users/zofia/github/DFM/Data/Std/Traj/T'+repr(simulatedTemperature)+ '/*.h5'):
    for np_name in glob.glob(dataName):
        numpy_vars.append(md.load(np_name))

    traj = numpy_vars
    print(len(traj))

    numberOfIterations=len(traj)

    Xmdtraj=list()
    for i in range(numberOfIterations):
        #print(traj[i][0].xyz)
        Xmdtraj.append(md.Trajectory(traj[i].xyz, top))#mdl.testsystem.topology) )

    print(Xmdtraj[0].xyz.shape[2])
    L=int(Xmdtraj[0].xyz.shape[0]*len(traj))
    nrP=int(Xmdtraj[0].xyz.shape[1])
    print(nrP)
    D=int(Xmdtraj[0].xyz.shape[2])
    X=np.zeros((L, nrP, D))
    for i in range(0,len(Xmdtraj)):
            X[i*len(Xmdtraj[i].xyz):(i+1)*len(Xmdtraj[i].xyz),:,:]=Xmdtraj[i].xyz

    Xref=X

    Xalign = md.Trajectory(X, top)
    Xalign=Xalign.center_coordinates()
    Xalign=Xalign.superpose(Xalign[0])
    X_FT=Xalign.xyz
    return X_FT


def compute_radius(X):
    return np.linalg.norm(X[:,0,:]-X[:,1,:], 2, axis=1)

#
# numberOfIterations=10
# dataNameEnergies = '/Traj/Energies/'
# nameDataEnergies = folderName+methodName+dataNameEnergies
# E = loadData(nameDataEnergies)
# print(X_FT.shape)


# def loadEnergy(X_FTnameData):
#     E=[]
#     Eval=np.zeros(len(X_FT))
#     Evalues=np.zeros(len(X_FT))
#
#     for i in range(0,numberOfIterations):
#
#         Etmp=np.load(nameData)
#
#         E.append(Etmp)
#
#     E=np.hstack(E)
#     plt.hist(E, 100)
#     plt.title('Saved from openmm')
#     plt.show()
#
#     return E

def computeTargetMeasure(X_FT, smpl):

    qTargetDistribution=np.zeros(len(X_FT))
    Erecompute=np.zeros(len(X_FT))
    from simtk import unit
    for i in range(0,len(X_FT)):
                Erecompute[i]=smpl.model.energy(X_FT[i,:,:]*smpl.model.x_unit).value_in_unit(smpl.model.energy_unit)
                tmp = Erecompute[i]
                betatimesH_unitless =tmp / smpl.kT.value_in_unit(smpl.model.energy_unit) #* smpl.model.temperature_unit
                qTargetDistribution[i]=np.exp(-(betatimesH_unitless))
    print('Done')

    return qTargetDistribution, Erecompute




def get_levelset_cv(X, width, cv):

    levelsets=list()
    for i in range(len(X)):
        if(cv[i] ==0):
            tmp=( np.where(np.abs(cv - cv[i]) < width))
        else:
            tmp=( np.where((np.abs(cv - cv[i])/np.abs(cv[i])) < width))

        levelsets.append( tmp[0])

    # m = float(cv.size)
    # delta =  100/m*(max(cv)-min(cv))
    # levelsets=list()
    # levels = np.linspace(min(cv),max(cv), num=width)
    # print(levels)
    # for k in range(width-1):
    #     tmp=np.where(np.abs(cv - levels[k]) < delta)
    #
    #     levelsets.append( tmp[0])


    return np.asarray(levelsets)



####################################

def compute_free_energy(X_FT, radius, width=0.05, weights=None, kBT=1):

    if (weights is None):

        weights =np.ones(len(radius))

    levelset=get_levelset_cv(X_FT, width, radius)


    free_energy=np.zeros(radius.shape)
    totallentgh=0
    for i in range(len(free_energy)):
        tmp= np.sum(weights[levelset[i] ])
        #print(tmp)
        free_energy[i] =tmp #* weight[i]
        totallentgh+=tmp


    free_energy=free_energy/np.sum(totallentgh)
    free_energy+=0.0001
    free_energy= - kBT*np.log(free_energy)

    return free_energy
