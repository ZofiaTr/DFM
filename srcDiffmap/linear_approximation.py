import numpy as np
from numpy import linalg as LA
import rmsd
import model
import mdtraj as md

from model import dummyModel, dummyTopology



def linApproxPsi(point, data_landmarks, V_landmarks, v):
    """
    add description:

    data_landmarks is mdTraj
    """

    # point is projected on the closest landmark (norm) - this should be done in a better way ..?

    K=len(data_landmarks)
    nr=np.zeros(K)

    #dlm=reshapeDataBack(data_landmarks)
    #X0 = point-  rmsd.centroid(point)
    #for i in range(0,len(X)):

    #    X[i] = X[i] -  rmsd.centroid(X[i])
    #    X[i] = rmsd.kabsch_rotate(X[i], X0)

    #point=point.reshape([point.shape[0]*point.shape[1]])
    #1) first find landmark points with minimal rmsd and then rotate the point wrt that landmark...
    #find k such that x is between l_k and l_k+1



    xyzPoint=point/dummyModel.x_unit #np.array([point.value_in_unit(modelObj.x_unit)])


    #print( xyzPoint.shape[0])
    #print( xyzPoint.shape[1])
    X=md.Trajectory(xyzPoint, dummyTopology)
    #LM0=md.Trajectory(data_landmarks[-1], dummyTopology)
    #X.superpose(LM0)

    for k in range(0,K):

        tmp= data_landmarks[k].reshape((xyzPoint.shape[0],xyzPoint.shape[1]))

        dlm =md.Trajectory(tmp, dummyTopology)
        #print dlm
        nr[k]=md.rmsd(dlm, X)

        #d1=xyzPoint-data_landmarks[k]
        #nr[k]=np.tensordot(d1, d1)
        #nr[k]=LA.norm(point-data_landmarks[k,:])
        #nr[k]=rmsd.kabsch_rmsd(point, dlm)
        #nr[k]=rmsd.calculate_rmsd point dlm
        #print nr[k]

    lkidx=np.argmin(nr)


    lk=data_landmarks[lkidx]
    #print v[lkidx].shape
    #print np.dot((point - lk).T, v[lkidx])

    xyzPointResh=xyzPoint#.reshape(lk.shape)
    #print(xyzPointResh - lk)

    #print lk
    psiX= np.tensordot(xyzPointResh - lk, v[lkidx]) + V_landmarks[lkidx]

    vPoint=v[lkidx]

    psiX=np.array(psiX)
    #vPoint=np.array(vPoint)

    vPoint=vPoint.reshape((xyzPoint.shape[0],xyzPoint.shape[1]))
    #print (vPoint.shape)
    #vPoint =md.Trajectory(vPoint, modelObj.testsystem.topology)
    #psiX=psiX*modelObj.x_unit

    return psiX, vPoint

def diff_lin(x1,x2, f1,f2):

    X1=md.Trajectory(x1, dummyTopology)
    X2=md.Trajectory(x2, dummyTopology)

    v=np.shape(x1)

    # dif=(x2-x1)
    #
    # nlk=LA.norm(dif)**2
    # if(nlk == 0):
    #     nlk=1
    #
    # v= dif*(f2-f1)/float(nlk)

    dif=(x2-x1)

    #nlk=md.rmsd(X1,X2)
    nlk=np.tensordot(dif, dif)
    if(nlk == 0):
        nlk=1

    v= dif*(f2-f1)/float(nlk)
    

    return v

def compute_all_derivatives(data_landmarks, V_landmarks):

    K=data_landmarks.shape[0]

    #dim=data_landmarks.shape[-1]
    #print(dim)
    v=np.zeros(data_landmarks.shape)

    for lkidx in range(0,K-1):

            x1=data_landmarks[lkidx+1]
            x0=data_landmarks[lkidx]

            f1=V_landmarks[lkidx+1]
            f0=V_landmarks[lkidx]

            v[lkidx]=diff_lin(x0,x1, f0,f1)

    if data_landmarks.ndim==1:
        v[K-1]=v[K-2]

    else:
        v[K-1,:]=v[K-2,:]



    return v
