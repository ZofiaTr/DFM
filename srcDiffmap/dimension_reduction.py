import numpy as np
from pydiffmap import diffusion_map as pydm
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import mdtraj as md



#--------------------- Functions --------------------------------

###########################################################


def dominantEigenvectorDiffusionMap(tr, epsilon, sampler, T, method, nrOfFirstEigenVectors=2, numberOfNeighborsPotential=1, energy = None,  metric='euclidean'):
        """
        Dimension reduction - compute diffusion map from the trajectory
        Parameters
        -------------
        tr : ndarray of shape n x DOF where n is number of time steps / frames and DOF is number of particles times space dimension
        nrOfFirstEigenVectors : number of first eigenvectors, without the first one. Carefull, pydiffmap takes nrOfFirstEigenVectors and resurns matrix with nrOfFirstEigenVectors+1 columns

        returns
        ------------
        v1 : ndarray of shape (len(tr), nrOfFirstEigenVectors), eigenvectors from diffusion maps
        """

        nrNeigh=5000
        if len(tr)< nrNeigh:
            nrNeigh = len(tr)-1

        print("Computing diffusion maps. ")

        chosenmetric = metric
        print('Chosen metric for diffusionmap is '+str(chosenmetric))

        X_FT =tr.reshape(tr.shape[0],sampler.model.testsystem.positions.shape[0],sampler.model.testsystem.positions.shape[1] )
        tr = align_with_mdanalysis(X_FT, sampler);
        tr = tr.reshape(X_FT.shape[0], X_FT.shape[1]*X_FT.shape[2])

        E = computeEnergy(tr, sampler, modelShape=False)

        qTargetDistribution=[]
        if method == 'TMDiffmap':
            qTargetDistribution= computeTargetMeasure(E, sampler)



        # if method=='PCA':
        #
        #     X_pca = decomposition.TruncatedSVD(n_components=1).fit_transform(tr- np.mean(tr,axis=0))
        #     v1=X_pca[:,0]
        #
        #     print('PCA not compatible ')
        #
        #     #qEstimated = qTargetDistribution
        #     #kernelDiff=[]
        #
        #     if(nrOfFirstEigenVectors>2):
        #         print('For PCA, nrOfFirstEigenVectors must be 1')

        if method == 'Diffmap':

            if chosenmetric == 'rmsd':
                print('Vanilla diffusion map needs to be adjusted for an explicit metric option. TBD.')

            mydmap = pydm.DiffusionMap(alpha = 0.5, n_evecs = nrOfFirstEigenVectors, epsilon = epsilon,  k=nrNeigh, metric=chosenmetric)#, neighbor_params = {'n_jobs':-4})
            dmap = mydmap.fit_transform(tr)

            P = mydmap.P
            lambdas = mydmap.evals

            v1 = mydmap.evecs
            qEstimated = mydmap.q
            kernelDiff=mydmap.local_kernel



        elif method =='TMDiffmap': #target measure diffusion map

            print("Temperature in TMDmap is "+repr(sampler.kT/sampler.model.kB_const))

            ##########
            if chosenmetric=='rmsd':

                print('Epsilon cant be automatic if metric is rmsd. Setting epsilon = 0.1.')
                epsilon = 0.1
                trxyz=tr.reshape(tr.shape[0], sampler.model.testsystem.positions.shape[0],sampler.model.testsystem.positions.shape[1])
                traj = md.Trajectory(trxyz, sampler.model.testsystem.topology)

                indptr = [0]
                indices = []
                data = []

                # choose k-nearest neighbors
                k = nrNeigh
                #----

                for i in range(traj.n_frames):
                    # compute distances to frame i
                    distances = md.rmsd(traj, traj, i)
                    # this performs a partial sort so that idx[:k] are the indices of the k smallest elements
                    idx = np.argpartition(distances, k)
                    # retrieve corresponding k smallest distances
                    distances = distances[idx[:k]]
                    # append to data structure
                    data.extend(np.exp(-1.0/epsilon*distances**2).tolist())
                    indices.extend(idx[:k].tolist())
                    indptr.append(len(indices))

                kernel_matrix = sps.csr_matrix((data, indices, indptr), dtype=float, shape=(traj.n_frames, traj.n_frames))
                local_kernel=kernel_matrix
                # this is all stolen from pydiffmap
                weights_tmdmap = qTargetDistribution

                alpha = 1.0
                q = np.array(kernel_matrix.sum(axis=1)).ravel()
                # Apply right normalization
                right_norm_vec = np.power(q, -alpha)
                if weights_tmdmap is not None:
                    right_norm_vec *= np.sqrt(weights_tmdmap)

                m = right_norm_vec.shape[0]
                Dalpha = sps.spdiags(right_norm_vec, 0, m, m)
                kernel_matrix = kernel_matrix * Dalpha

                # Perform  row (or left) normalization
                row_sum = kernel_matrix.sum(axis=1).transpose()
                n = row_sum.shape[1]
                Dalpha = sps.spdiags(np.power(row_sum, -1), 0, n, n)
                P = Dalpha * kernel_matrix

                evals, evecs = spsl.eigs(P, k=(nrOfFirstEigenVectors+1), which='LM')
                ix = evals.argsort()[::-1][1:]
                evals = np.real(evals[ix])
                evecs = np.real(evecs[:, ix])
                dmap = np.dot(evecs, np.diag(evals))


                lambdas = evals
                v1 = evecs
                qEstimated = q
                kernelDiff=local_kernel

            elif chosenmetric=='euclidean':


                mydmap = pydm.DiffusionMap(alpha = 1, n_evecs = nrOfFirstEigenVectors, epsilon = epsilon,  k=nrNeigh, metric='euclidean')#, neighbor_params = {'n_jobs':-4})
                dmap = mydmap.fit_transform(tr, weights = qTargetDistribution)

                P = mydmap.P
                lambdas = mydmap.evals
                v1 = mydmap.evecs

                [evalsT, evecsT] = spsl.eigs(P.transpose(),k=1, which='LM')
                phi = np.real(evecsT.ravel())

                #q = mydmap.q

                qEstimated = mydmap.q
                kernelDiff=mydmap.local_kernel

            else:
                print('In tmdmap- metric not defined correctly: choose rmsd or euclidean.')

        else:
            print('Error in sampler class: dimension_reduction function did not match any method.\n Choose from: TMDiffmap, PCA, Diffmap')


        return v1, qTargetDistribution, qEstimated, E, kernelDiff, lambdas



def computeTargetMeasure(Erecompute, smpl):

    print('Computing the target measure.')
    qTargetDistribution=np.zeros(len(Erecompute))


    for i in range(0,len(Erecompute)):
                #Erecompute[i]=smpl.model.energy(X_FT[i,:,:]*smpl.model.x_unit).value_in_unit(smpl.model.energy_unit)
                tmp = Erecompute[i]
                betatimesH_unitless =tmp / smpl.kT.value_in_unit(smpl.model.energy_unit) #* smpl.model.temperature_unit
                qTargetDistribution[i]=np.exp(-(betatimesH_unitless))
    #print('Done')

    return qTargetDistribution#, Erecompute

def computeEnergy(X_reshaped, smpl, modelShape = False):

    print('Computing the energy.')

    if modelShape:
        X_FT=X_reshaped
    else:
        X_FT =X_reshaped.reshape(X_reshaped.shape[0],smpl.model.testsystem.positions.shape[0],smpl.model.testsystem.positions.shape[1] )
    qTargetDistribution=np.zeros(len(X_FT))
    Erecompute=np.zeros(len(X_FT))
    from simtk import unit
    for i in range(0,len(X_FT)):
                Erecompute[i]=smpl.model.energy(X_FT[i]*smpl.model.x_unit).value_in_unit(smpl.model.energy_unit)

    return  Erecompute

# def dimension_reduction(tr, eps, numberOfLandmarks, sampler, T, method):
#
#         assert(False), 'under construction: dominantEigenvectorDiffusionMap has changed '
#         v1, q, qEstimated, potEn, kernelDiff=dominantEigenvectorDiffusionMap(tr, eps, sampler, T, method)
#
#             #get landmarks
#         lm=dm.get_landmarks(tr, numberOfLandmarks, q , v1, potEn)
#
#         assert( len(v1) == len(tr)), "Error in function dimension_reduction: length of trajectory and the eigenvector are not the same."
#
#         #
#         # plt.scatter(v1, v2)
#         # plt.scatter(v1[lm], v2[lm], c='r')
#         # plt.xlabel('V1')
#         # plt.ylabel('V2')
#         # plt.show()
#
#         return lm, v1
#


#
# def min_rmsd(X):
#
#     #first frame
#     X0 = X[0] -  rmsd.centroid(X[0])
#
#     for i in range(0,len(X)):
#     #print "RMSD before translation: ", rmsd.kabsch_rmsd(Y[0],Y[1::])
#
#         X[i] = X[i] -  rmsd.centroid(X[i])
#         X[i] = rmsd.kabsch_rotate(X[i], X0)
#
#         ## alternatively use quaternion rotate  instead - does not work in this format
#         #rot = rmsd.quaternion_rotate(X[i], X0)
#         #X[i] = np.dot(X[i], rot)
#
#     return X



#from MDAnalysis.tests.datafiles import PSF, DCD, PDB_small
def align_with_mdanalysis(X_FT, smpl):

    import MDAnalysis as mda
    from MDAnalysis.analysis import align
    from MDAnalysis.analysis.rms import rmsd

    trj = mda.Universe(smpl.model.modelName+'.pdb', X_FT)
    ref = mda.Universe(smpl.model.modelName+'.pdb')
    # trj = mda.Universe('/Users/zofia/github/DFM/alanine.xyz', X_FT)
    # print(trj.trajectory)
    # ref = mda.Universe('/Users/zofia/github/DFM/alanine.xyz')#, X_FT[0,:,:])


    alignment = align.AlignTraj(trj, ref)#, filename='rmsfit.dcd')
    alignment.run()
    X_aligned = np.zeros(X_FT.shape)
    ci=0
    for ts in trj.trajectory:
        X_aligned[ci] = trj.trajectory.ts.positions
        ci=ci+1

    #X_aligned = (trj.trajectory.positions)
    #print(X_aligned.shape)
    #print(alignment)
    return X_aligned
