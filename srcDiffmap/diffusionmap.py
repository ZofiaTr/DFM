"""Diffusion map"""

# Author: Zofia
# License:


import numpy as np

import math
from scipy.sparse import csr_matrix

import scipy.sparse.linalg as SLA

import sklearn.neighbors as neigh_search
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sps

import mdtraj as md

from numpy import linalg as LA

import rmsd

import model


dummyModel=model.dummyModel#model.Model()
#

#epsilon = 0.1;
#r=math.sqrt(2*epsilon);

def reshapeData(X):

    m, nrP, dim=X.shape
    Xreshaped=np.zeros((m,nrP*dim))
    for i in range(0,m):
        for j in range(0, nrP):
            Xreshaped[i,j*dim:(j+1)*dim]=X[i,j,:]
    return Xreshaped


# compute sparse matrices from a given trajectory


def compute_kernel(X, epsilon, myMetric=None, adaptive_epsilon=None):
    """
    Compute kernel of trajectory: using RMSD distances
    parameters: X is matrix of size number of steps times nDOF
    """
    #
    # #kernel=kernel.Kernel(type='gaussian', distance = 'euclidean', epsilon = 1.0, k=64)
    #
    #
    # Y = X
    # compute_self = True
    # k=64
    # typei='gaussian'
    # neigh = NearestNeighbors(metric=myRMSDmetric)
    # k0 = min(k, np.shape(Y)[0])
    # A = neigh.fit(Y).kneighbors_graph(X,n_neighbors=k0, mode='distance')
    # # retrieve all nonzero elements and apply kernel function to it
    # v = A.data
    # if (typei=='gaussian'):
    #     A.data = np.exp(-v**2/epsilon)
    # else:
    #     raise("Error: Kernel type not understood.")
    # # symmetrize
    # if (compute_self == True):
    #     A = 0.5*(A+A.transpose())
    #
    # return A
    # ##################################
    m = np.shape(X)[0];

    cutoff = np.sqrt(2*epsilon);

    if myMetric == None:
        metric=myRMSDmetric
    else:
        metric=myMetric

    #calling nearest neighbor search class: returning a (sparse) distance matrix
    #albero = neigh_search.radius_neighbors_graph(X, radius = cutoff, mode='distance', p=2, include_self=None)
    print('constructing neighbors graph with metric '+repr(metric))
    albero = neigh_search.radius_neighbors_graph(X, radius=cutoff, mode='distance', metric = metric, include_self=None)#mode='pyfunc',, metric_params={'myRMSDmetric':myRMSDmetric}, include_self=None)
    print('neighbors graph done')
    #albero = neigh_search.radius_neighbors_graph(X.xyz, radius=cutoff, mode='pyfunc', metric_params={'func' : md.rmsd}, include_self=None)

    #adaptive epsilon
    x=np.array(albero.data)

    if adaptive_epsilon == None:
        adaptiveEpsilon=epsilon
    else:
        adaptiveEpsilon=0.5*np.mean(x)


    diffusion_kernel = np.exp(-(x**2)/(adaptiveEpsilon))
    #print("Adaptive epsilon in compute_kernel is "+repr(adaptiveEpsilon))

    # adaptive epsilon should be smaller as the epsilon, since it is half of maximal distance which is bounded by cutoff parameter
    assert( adaptiveEpsilon <= epsilon )

    # computing the diffusion kernel value at the non zero matrix entries
    #diffusion_kernel = np.exp(-(np.array(albero.data)**2)/(epsilon))

    # build sparse matrix for diffusion kernel
    kernel = sps.csr_matrix((diffusion_kernel, albero.indices, albero.indptr), dtype = float, shape=(m,m))
    kernel = kernel + sps.identity(m)  # accounting for diagonal elements

    return kernel;

def compute_kernel_mdtraj(traj, epsilon):
    """UNDER CONSTRUCTION: DOES NOT WORK"""

    #check the format of traj - if there are more particles
    print('You are in compute_kernel_mdtraj: under construction, does not work yet..')
    # Number of frames in trajectory
    m = len(traj)

    # Compute cutoff for neighbor search
    cutoff = np.sqrt(2*epsilon);

    # Calling nearest neighbor search class: returning a (sparse) distance matrix

    tmpTraj = traj.xyz
    reshapedTraj = tmpTraj.reshape(tmpTraj.shape[0], tmpTraj.shape[1]*tmpTraj.shape[2] )
    albero = neigh_search.radius_neighbors_graph(reshapedTraj, radius=cutoff, mode='distance', metric = myRMSDmetric, include_self=None)#mode='pyfunc',, metric_params={'myRMSDmetric':myRMSDmetric}, include_self=None)

    #md.rmsd(X[i], X[j])

    # computing the diffusion kernel value at the non zero matrix entries
    diffusion_kernel = np.exp(-(np.array(albero.data)**2)/(epsilon))

    # build sparse matrix for diffusion kernel
    kernel = sps.csr_matrix((diffusion_kernel, albero.indices, albero.indptr), dtype = float, shape=(m,m))
    kernel = kernel + sps.identity(m)  # accounting for diagonal elements

    return kernel;

def myEuclideanMetric(arr1, arr2):
    """
    Under assumption that the trajectory is aligned w.r.t minimal rmsd w.r.t. first frame
    This is built under the assumption that the space dimension is 3!!!
    Requirement from sklearn radius_neighbors_graph: The callable should take two arrays as input and return one value indicating the distance between them.
     Input: One row from reshaped xyz trajectory as number of steps times nDOF
     Inside: Reshape back to xyz (NrPart, dim) and apply norm
     Output: r
    """


    nParticles = len(arr1) / 3;
    assert (nParticles == int(nParticles))

    arr1 = arr1.reshape(int(nParticles), 3 )
    arr2 = arr2.reshape(int(nParticles), 3 )

    s=0
    for i in range(int(nParticles)):
        stmp = np.linalg.norm(arr1[i,:]-arr2[i,:])
        s+=stmp*stmp
    d=np.sqrt( s / nParticles)


    return d


def myRMSDmetric(arr1, arr2):
    """
    This is built under the assumption that the space dimension is 3!!!
    Requirement from sklearn radius_neighbors_graph: The callable should take two arrays as input and return one value indicating the distance between them.
     Input: One row from reshaped xyz trajectory as number of steps times nDOF
     Inside: Reshape back to md.Trajectory and apply md.rmsd as r=md.rmsd(X[i], X[j])
     Output: r
    """


    nParticles = len(arr1) / 3;
    assert (nParticles == int(nParticles))

    arr1 = arr1.reshape(int(nParticles), 3 )
    arr2 = arr2.reshape(int(nParticles), 3 )


    p1MD=md.Trajectory(arr1, dummyModel.testsystem.topology)
    p2MD=md.Trajectory(arr2, dummyModel.testsystem.topology)

    d=md.rmsd(p1MD, p2MD)#, precentered=True)

    return d

def myRMSDmetricPrecentered(arr1, arr2):
    """
    This is built under the assumption that the space dimension is 3!!!
    Requirement from sklearn radius_neighbors_graph: The callable should take two arrays as input and return one value indicating the distance between them.
     Input: One row from reshaped xyz trajectory as number of steps times nDOF
     Inside: Reshape back to md.Trajectory and apply md.rmsd as r=md.rmsd(X[i], X[j])
     Output: r
    """


    nParticles = len(arr1) / 3;
    assert (nParticles == int(nParticles))

    arr1 = arr1.reshape(int(nParticles), 3 )
    arr2 = arr2.reshape(int(nParticles), 3 )


    p1MD=md.Trajectory(arr1, dummyModel.testsystem.topology)
    p2MD=md.Trajectory(arr2, dummyModel.testsystem.topology)

    d=md.rmsd(p1MD, p2MD, precentered=True)

    return d



def min_rmsd(arr1, arr2):

    nParticles = len(arr1) / 3;
    assert (nParticles == int(nParticles))

    X1 = arr1.reshape(int(nParticles), 3 )
    X2 = arr2.reshape(int(nParticles), 3 )

    X1 = X1 -  rmsd.centroid(X1)
    X2 = X2 -  rmsd.centroid(X2)

    x = rmsd.kabsch_rmsd(X1, X2)

    return x




def compute_P(kernel, X):

    alpha = 0.5;
    m = np.shape(X)[0];
    D = sps.csr_matrix.sum(kernel, axis=1).transpose();
    Dalpha = sps.spdiags(np.power(D,-alpha), 0, m, m)
    kernel = Dalpha * kernel * Dalpha;
    D = sps.csr_matrix.sum(kernel, axis=1).transpose();
    Dalpha = sps.spdiags(np.power(D,-1), 0, m, m)
    kernel = Dalpha * kernel;

    return kernel

def compute_unweighted_P( X, epsilon, sampler, target_distribution, kernel=None):

    #print('Unweighting according to temperature '+repr(sampler.T))
    m = len(X) #np.shape(X)[0];
    if kernel==None:
        if isinstance(X, md.Trajectory):
            kernel=compute_kernel_mdtraj(X, epsilon)
        else:

            kernel = compute_kernel(X, epsilon)

    qEmp=kernel.sum(axis=1)

    weights = np.zeros(m)

    for i in range(0,len(X)):

        #target_distribution[i] = np.exp( -  sampler.model.potential(X[i]) / sampler.T)
        weights[i] = np.sqrt(target_distribution[i]) /  qEmp[i]

    D = sps.spdiags(weights, 0, m, m)
    Ktilde =  kernel * D

    #Dalpha = sps.csr_matrix.sum(Ktilde, axis=0);
    Dalpha = sps.csr_matrix.sum(Ktilde, axis=1).transpose();
    Dtilde = sps.spdiags(np.power(Dalpha,-1), 0, m, m)

    L = Dtilde * Ktilde

    return L, qEmp, kernel

def pdist2(x,y):
    v=np.sqrt(((x-y)**2).sum())
    return v

def get_eigenvectors(data, nrEV, **kwargs):

    if ('DiffMapMatrix' in kwargs):
         P= kwargs['DiffMapMatrix']
    else:
         P,q = compute_P_and_q(data)
    lambdas, V = SLA.eigs(P, k=nrEV)#, which='LM' )
    ix = lambdas.argsort()[::-1]
    return V[:,ix], lambdas[ix]




def get_landmarks(data, K, q, V1, potEn, getLevelSets=None):

    m = float(q.size)
    #q=np.array(q)

    delta = 100/m*(max(V1)-min(V1))
    deltaMax=2*delta
    levels = np.linspace(min(V1),max(V1),num=K)

    lb = 1

    landmarks=np.zeros(K)
    emptyLevelSet=0

    if(getLevelSets):
        levelsets=list()

    # compute potential energy on the data points (usually available in the codes..)
    # E=np.zeros(len(data))
    # for n in range(0,len(data)):
    #     E[n] = potEn(data[n,:])
    E=potEn

    for k in range(K-1, -1, -1):


            levelsetLength=0

            # we want to identify idices in V1 which are delta close to the levels
            #---o----o----o----o----o---
            #  *o** *o*  *o*  *o*   o
            # if there are no indeces in the delta distance, increase the delta distance

            while levelsetLength==0:

                levelset = np.where(np.abs(V1 - levels[k]) < delta)
                levelset=levelset[0]

                if(getLevelSets):
                    levelsets.append(levelset)

                levelsetLength=len(levelset)
                #print levelsetLength

                delta=delta*1.001

                if delta>deltaMax:
                    levelset=range(0,len(V1))
                    #print("In get_landmarks: Levelset chosen as V1")

            data_level = data[levelset,:]

            if k==K-1:
                #idx = np.argmax(q[levelset]/m )
                idx = np.argmin(E[levelset])#/m )
                landmarks[k]= levelset[idx]

            else:
                #idx = np.argmax(q[levelset]/m )
                #qtmp=q[levelset]/m
                idx = np.argmin(E[levelset])#/m )
                qtmp=E[levelset]#/m

                # compute the distance to the last landmark
                dist_to_last=np.zeros(data_level.shape[0])
                for i in range(data_level.shape[0]):
                    dist_to_last[i] = pdist2(data[int(landmarks[k+1]),:], data_level[i])

                #dtmp=np.array(dist_to_last.reshape(qtmp.shape))
                dtmp=dist_to_last
                #print(dtmp.shape)
                v=qtmp  + lb*dtmp

                idx = np.argmin(v);

                landmarks[k]= levelset[idx]

    if(getLevelSets):
        return landmarks.astype(int), levelsets, levels
    else:
        return landmarks.astype(int)

def get_levelsets(data, K, q, V1):

    m = float(q.size)
    #q=np.array(q)

    delta = 100.0/m*(max(V1)-min(V1))
    deltaMax=2*delta
    levels = np.linspace(min(V1),max(V1),num=K)

    lb = 1

    landmarks=np.zeros(K)
    emptyLevelSet=0

    levelsets=list()

    for k in range(K-1, -1, -1):

            levelsetLength=0

            # we want to identify idices in V1 which are delta close to the levels
            #---o----o----o----o----o---
            #  *o** *o*  *o*  *o*   o
            # if there are no indeces in the delta distance, increase the delta distance

            while levelsetLength==0:

                levelset_k = np.where(np.abs(V1 - levels[k]) < delta)
                levelsetLength=len(levelset_k)
                #print levelsetLength

                delta=delta*1.001

                if delta>deltaMax:
                    levelset_k=range(0,len(V1))
                    #print("In get_landmarks: Levelset chosen as V1")

                levelsets.append(levelset_k[0])

    return levelsets, levels

def get_levelset_onePoint(idx, width, V1):


    #q=np.array(q)


    deltaMax=2*width

    #print(np.abs(V1 - V1[idx]))



    if(V1[idx] ==0):
        tmp=( np.where(np.abs(V1 - V1[idx]) < width))
    else:
        tmp=( np.where(np.abs(V1 - V1[idx])/np.abs(V1[idx]) < width))

    levelset= tmp[0]


    return np.asarray(levelset)


def sort_landmarks(data, landmarks):

    X=data[landmarks,:]

    #if len(X.shape)>2:
    #    X=reshapeData(X)


    nbrs = NearestNeighbors(n_neighbors=len(landmarks), algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    return landmarks[indices[0]]

def sort_landmarkedPoints(gridCV):

    dim=gridCV.shape[1]
    X=gridCV[:,0:dim+1]

    if len(X.shape)>2:
        X=reshapeData(X)


    nbrs = NearestNeighbors(n_neighbors=len(X), algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    return gridCV[indices[0], :]

def compute_eigenvectors(laplacian):

    #random_state = check_random_state(random_state)

    laplacian *= -1
    v0 = np.random.uniform(-1, 1, laplacian.shape[0])
    lambdas, diffusion_map = eig(laplacian, k=n_components,
                                         sigma=1.0, which='LM',
                                         tol=eigen_tol, v0=v0)
    return diffusion_map.T[n_components::-1] # * dd
