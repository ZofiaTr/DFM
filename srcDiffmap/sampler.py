
"""
The main class for Diffusionmap-directed sampling
Author: Zofia Trstanova
Comment:
The standard dynamics is in runStd
The diffmap dircted sampling is in runDiffmapCV_Eftad_local.
The rest of the functions is in progress or not working.

"""

import numpy as np

import diffusionmap as dm

from pydiffmap import diffusion_map as pydm

import integrator
import linear_approximation as aprx
import Averages as av
import model

import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.spatial.distance as scidist
import rmsd

import time
import mdtraj as md


import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd

#import kernel as krnl

maxDataLength=5000
everyN=1
writingEveryNSteps=1
savingEveryNIterations=100
adjustEpsilon=1
saveModNr=10;

class Sampler():
    """
    Read the comments in function run() to see which algorithms are working and which are under construction.
    """


    def __init__(self, model, integrator, algorithm=0, dataFileName='/Data', dataFrontierPointsName = '/FrontierPoints', dataEigenVectorsName='/Eigenvectors', dataEnergyName='/Energies'):

        self.model=model
        self.algorithm=algorithm
        self.integrator=integrator

        self.diffmap_metric = 'euclidean' # dm.myRMSDmetricPrecentered

        #set the temperature here - then passed to the integrator
        self.kT=self.integrator.kT

        self.timeAv=av.Average(0.0)

        self.dim=3
        self.dimCV=1

        self.method='TMDiffmap'#'TMDiffmap'#'Diffmap'

        #diffusion maps constants
        self.epsilon=0.1 #0.05
        self.epsilon_Max=self.epsilon* (2.0**4)

        #linear approximation
        self.numberOfLandmarks=10

        self.nrDecorrSteps=1

        self.modNr=10
        self.modEFTAD=10

        self.kTraj_LM_save=None
        self.V1_LM_save=None

        self.gridCV_long=None
        self.sampled_trajectory=np.array([self.integrator.x0.value_in_unit(self.model.x_unit)])
        self.landmarkedStates=None

        self.changeTemperature = 0
        self.corner = 0



        if self.algorithm==0:
            self.algorithmName='std'
        if self.algorithm==1:
            self.algorithmName='eftad_fixed_cv'
        if self.algorithm==2:
            self.algorithmName='initial_condition'
        # if self.algorithm==3:
        #     self.algorithmName='frontier_points'
        if self.algorithm==3:
            self.algorithmName='frontier_points_corner_change_temperature'
        # if self.algorithm==4:
        #     self.algorithmName='frontier_points_corner'
        if self.algorithm==4:
            self.algorithmName='frontier_points_corner_change_temperature_off'



        #print(self.model.testsystem.topology)
        self.topology = md.Topology().from_openmm(self.model.testsystem.topology)
        self.trajSave= md.Trajectory(self.model.positions.value_in_unit(self.model.x_unit), self.topology)
        self.savingFolder=dataFileName+'/'+self.model.modelName
        self.savingFolderEnergy=dataEnergyName
        self.savingFolderFrontierPoints = dataFrontierPointsName
        self.savingFolderEigenvectors = dataEigenVectorsName

        self.saveEigenvectors=1
        self.saveEnergy=1
        self.smoothEnergyFlag=0



    def resetInitialConditions(self):
            self.integrator.x0=self.integrator.xEnd
            self.integrator.v0=self.integrator.vEnd
            self.integrator.z0=self.integrator.zEnd
            self.integrator.vz0=self.integrator.vzEnd
            self.integrator.kineticTemperature.clear()



    def run(self, nrSteps, nrIterations, nrRep):

        print('Model: '+self.model.modelname )
        print('Algorithm: '+self.algorithmName)
        print('Running '+repr(int(nrIterations))+'x '+repr(int(nrSteps))+' steps')

        run_function = getattr(self, 'run_' + self.algorithmName)
        print(run_function)
        try:
            print("Running algorithm '%s'" % ('run_' + self.algorithmName))
            run_function(nrSteps, nrIterations, nrRep)
        except Exception:
            raise("Could not find run function '%s'" % ('run_' + self.algorithmName))

    def run_frontier_points_corner_change_temperature_off(self, nrSteps, nrIterations, nrRep):
        self.changeTemperature = 0
        self.corner = 1
        self.run_frontierPoints(nrSteps, nrIterations, nrRep)

    def run_frontier_points_corner_change_temperature(self, nrSteps, nrIterations, nrRep):
        self.changeTemperature = 1
        self.corner = 1
        self.run_frontierPoints(nrSteps, nrIterations, nrRep)

######----------------- STD ---------------------------------

    def run_std(self, nrSteps, nrIterations, nrRep):


            #reset time
            self.timeAv.clear()

            print('replicated std dynamics with '+repr(nrRep)+' replicas')

            initialPositions=[self.integrator.x0 for rep in range(0,nrRep)]
            #Xit=[self.integrator.x0 for i in range(nrRep*nrSteps)]

            ##############################

            for it in range(0,nrIterations):

                if(it>0):
                    tav0=time.time()

                if(np.remainder(it, writingEveryNSteps)==0):
                    print('Iteration '+ repr(it))
                    print('Kinetic Temperature is '+str(self.integrator.kineticTemperature.getAverage()))


                    if(it>0):
                        t_left=(self.timeAv.getAverage())*(nrIterations-it)
                        print(time.strftime("Time left %H:%M:%S", time.gmtime(t_left)))

                #------- simulate Langevin

                # xyz = list()
                #
                # for rep in range(0, nrRep):
                #
                #     # TODO: Also track initial and final velocities
                #
                #     self.integrator.x0=initialPositions[rep]
                #     #xyz += self.integrator.run_langevin(nrSteps, save_interval=self.modNr) # local Python
                #     xyz += self.integrator.run_openmm_langevin(nrSteps, save_interval=self.modNr)
                #     initialPositions[rep] = self.integrator.xEnd
                #
                #
                # if(it>0):
                #     tavEnd=time.time()
                #     self.timeAv.addSample(tavEnd-tav0)

                xyz = list()
                potentialEnergyList = list()

                for rep in range(0, nrRep):

                    # TODO: Also track initial and final velocities

                    # each intial replica has initial condition
                    self.integrator.x0=initialPositions[rep]

                    #xyz_iter, potEnergy = self.integrator.run_langevin(nrSteps, save_interval=self.modNr) # local Python
                    xyz_iter, potEnergy = self.integrator.run_openmm_langevin(nrSteps, save_interval=self.modNr)
                    xyz += xyz_iter
                    if self.saveEnergy==1:
                        potentialEnergyList +=potEnergy
                    # #only for replicas
                    initialPositions[rep] = self.integrator.xEnd

                #tmpselftraj=np.copy(self.sampled_trajectory)
                #self.sampled_trajectory=np.concatenate((tmpselftraj,np.copy(Xrep[-1].value_in_unit(self.model.x_unit))))

                self.trajSave=md.Trajectory(xyz, self.topology)
                #self.trajSave = self.trajSave.center_coordinates()
                print(self.trajSave)

                print('Saving traj to file')
                self.trajSave.save(self.savingFolder+'traj_'+repr(it)+'.h5')

                if self.saveEnergy==1:
                    print('Saving energy to file')
                    np.save(self.savingFolderEnergy+'E_'+repr(it), potentialEnergyList)

                #np.save(self.savingFolder+'E_'+repr(it),potEnergy)
                #if (nrSteps*nrIterations*nrRep)> 10**6:
            #    self.sampled_trajectory=Xrep[::modNr**2]
            #else:
            #    self.sampled_trajectory=Xit

#----------------- TAMD/EFTAD

    def run_eftad_fixed_cv(self,nrSteps, nrIterations, nrRep):
        # use the eftad with CV as a projection on a x axis

            #reset time
            self.timeAv.clear()

            print('replicated eftad dynamics with '+repr(nrRep)+' replicas')

            initialPositions=[self.integrator.x0 for rep in range(0,nrRep)]
            Xit=[self.integrator.x0 for i in range(nrIterations*nrRep*nrSteps)]

            ##############################

            for it in range(0,nrIterations):

                if(it>0):
                    tav0=time.time()

                if(np.remainder(it, writingEveryNSteps)==0):
                    print('Iteration '+ repr(it))

                    if(it>0):
                        t_left=(self.timeAv.getAverage())*(nrIterations-it)
                        print(time.strftime("Time left %H:%M:%S", time.gmtime(t_left)))

                #------- simulate Langevin

                Xrep=[self.model.positions for i in range(nrRep*nrSteps)]

                for rep in range(0, nrRep):

                    self.integrator.x0=initialPositions[rep]
                    xs,vx=self.integrator.run_EFTAD(nrSteps)
                    initialPositions[rep]=xs[-1]

                    for n in range(0,nrSteps):
                        Xrep[rep*nrSteps + n]=xs[n]

                for n in range(0,nrRep*nrSteps):
                    Xit[it* (nrRep*nrSteps) + n]=Xrep[n]

                if(it>0):
                    tavEnd=time.time()
                    self.timeAv.addSample(tavEnd-tav0)

                xyz=[x.value_in_unit(self.model.x_unit) for x in Xrep[::self.modNr]]
                self.trajSave=md.Trajectory(xyz, self.topology)

                print('Saving traj to file')
                self.trajSave.save(self.savingFolder+'traj_'+repr(it)+'.h5')


    def run_initial_condition(self, nrSteps, nrIterations, nrRep):
            """
            Reset initial conditions by maximizing the distance over the collective variable. Implemented for the dimer where we take simply maximal rmsd w.r.t initial condition.
            """
            #reset time
            self.timeAv.clear()

            print('runInitialCondition: replicated std dynamics with '+repr(nrRep)+' replicas')

            initialPositions=[self.integrator.x0 for rep in range(0,nrRep)]
            #Xit=[self.integrator.x0 for i in range(nrRep*nrSteps)]

            ##############################

            for it in range(0,nrIterations):

                if(it>0):
                    tav0=time.time()

                if(np.remainder(it, writingEveryNSteps)==0):
                    print('Iteration '+ repr(it))
                    print('Kinetic Temperature is '+str(self.integrator.kineticTemperature.getAverage()))


                    if(it>0):
                        t_left=(self.timeAv.getAverage())*(nrIterations-it)
                        print(time.strftime("Time left %H:%M:%S", time.gmtime(t_left)))

                #------- simulate Langevin

                xyz = list()

                for rep in range(0, nrRep):

                    # TODO: Also track initial and final velocities

                    self.integrator.x0=initialPositions[rep]

                    #xyz_iter=self.integrator.run_langevin(nrSteps, save_interval=self.modNr) # local Python

                    xyz_iter, potEnergy=  self.integrator.run_openmm_langevin(nrSteps, save_interval=self.modNr)
                    xyz += xyz_iter

                    Xmd = md.Trajectory(xyz_iter , self.topology)
                    #
                    distances = np.zeros(len(xyz_iter))
                    for j in range(len(xyz_iter)):
                        distances[j]=md.rmsd(Xmd[0], Xmd[j])

                    maxrmsd= Xmd[np.argmax(distances)].xyz[0]
                    initialPositions[rep] = maxrmsd * self.model.x_unit


                if(it>0):
                    tavEnd=time.time()
                    self.timeAv.addSample(tavEnd-tav0)


                self.trajSave=md.Trajectory(xyz, self.topology)

                print('Saving traj to file')
                self.trajSave.save(self.savingFolder+'traj_'+repr(it)+'.h5')


##------------------------------------------
    def run_frontierPoints(self, nrSteps, nrIterations, nrRep):
            """
            Corner stones algorithm: run Langevin dynamics, compute diffusion maps, find corner points and reset initial conditons
            and increase temperature for first n steps and then reset temperature to finish the iteration
            """

            #reset time
            self.timeAv.clear()

            print('runFrontierPoints: replicated std dynamics with '+repr(nrRep)+' replicas')
            print('reset initial condition by maximizing the domninant eigenvector obtained by diffusion map')


            if (self.changeTemperature == 1):
                print('Changing temperature on')

            if( self.corner == 1):
                print('Cornerstones algorithm on')
            else:
                print('Frontier points algorithm on')

            if self.method=='TMDiffmap':
                if (self.smoothEnergyFlag==1):
                    print('Smoothing energy for the target distribution is on')
                else:
                    print('Smoothing energy for the target distribution is off')



            #intialisation
            initialPositions=[self.integrator.x0 for rep in range(0,nrRep)]
            #Xit=[self.integrator.x0 for i in range(nrRep*nrSteps)]

            ##############################

            xyz_history = list()
            potentialEnergy_history = list()

            for it in range(0,nrIterations):

                if(it>0):
                    tav0=time.time()

                if(np.remainder(it, writingEveryNSteps)==0):
                    print('Iteration '+ repr(it))
                    print('Kinetic Temperature is '+str(self.integrator.kineticTemperature.getAverage()))


                    if(it>0):
                        t_left=(self.timeAv.getAverage())*(nrIterations-it)
                        print(time.strftime("Time left %H:%M:%S", time.gmtime(t_left)))

                #--------------------
                #change temperature for the integrator-> the target termperature (passed to diffusionmaps) remains unchanged
                if(self.changeTemperature == 1):
                    if(it>0):# and it < nrIterations-2):

                        T = self.kT/ self.model.kB_const
                        self.integrator.temperature = T+500  * self.model.temperature_unit
                        print("Changing temperature to T="+repr(self.integrator.temperature))

                        print('Simulating at higher temperature')
                    # if(it>0):
                    #     T = self.kT/self.model.temperature_unit / self.model.kB_const
                    #
                    #     T= T*(0.01+ ((0.25*np.abs(np.cos(0.2*np.pi*it))+1.0)))
                    #
                    #     self.integrator.temperature = np.asscalar(T) * self.model.temperature_unit
                    #     print("Changing temperature to T="+repr(self.integrator.temperature))
                #------- simulate Langevin
                    xyz = list()
                    potentialEnergyList=list()


                    ratioStepsHigherTemperature=0.1

                    for rep in range(0, nrRep):


                        # each intial replica has initial condition
                        self.integrator.x0=initialPositions[rep]

                        #xyz_iter=self.integrator.run_langevin(nrSteps, save_interval=self.modNr) # local Python
                        xyz_iter, potEnergy = self.integrator.run_openmm_langevin(int(ratioStepsHigherTemperature*nrSteps), save_interval=self.modNr)
                        xyz += xyz_iter
                        if self.saveEnergy==1:
                            potentialEnergyList +=potEnergy
                        # #only for replicas
                        initialPositions[rep] = self.integrator.xEnd


                    #reset temperature

                    self.integrator.temperature = self.kT/ self.model.kB_const
                    print("Changing temperature back to T="+repr(self.integrator.temperature))

                    print('Simulating at target temperature')

                    for rep in range(0, nrRep):

                        # each intial replica has initial condition
                        self.integrator.x0=initialPositions[rep]

                        #xyz_iter=self.integrator.run_langevin(nrSteps, save_interval=self.modNr) # local Python
                        xyz_iter, potEnergy = self.integrator.run_openmm_langevin(int((1.0-ratioStepsHigherTemperature)*nrSteps), save_interval=self.modNr)
                        xyz += xyz_iter
                        if self.saveEnergy==1:
                            potentialEnergyList +=potEnergy
                        # #only for replicas
                        initialPositions[rep] = self.integrator.xEnd

                #elif(self.changeTemperature == 0 or it> nrIterations-2):
                else:
                    xyz = list()
                    potentialEnergyList=list()

                    for rep in range(0, nrRep):

                        # TODO: Also track initial and final velocities

                        # each intial replica has initial condition
                        self.integrator.x0=initialPositions[rep]

                        #xyz_iter=self.integrator.run_langevin(nrSteps, save_interval=self.modNr) # local Python
                        xyz_iter, potEnergy= self.integrator.run_openmm_langevin(nrSteps, save_interval=self.modNr)
                        xyz += xyz_iter
                        if self.saveEnergy==1:
                            potentialEnergyList +=potEnergy


                        # #only for replicas
                        #initialPositions[rep] = self.integrator.xEnd

                # creat md traj object
                self.trajSave=md.Trajectory(xyz, self.topology)
                #self.trajSave = self.trajSave.center_coordinates()

                #DEBUG
                # for i in range(len(self.trajSave)):
                #     print(md.rmsd(self.trajSave[0], self.trajSave[i]))

                # print('DEBUG: Saving traj to file')
                # self.trajSave.save(self.savingFolder+'traj_'+repr(it)+'.h5')
                #------ rmsd ------------------------------


                # align all the frames wrt to the first one according to minimal rmsd
                #self.trajSave.superpose(self.trajSave[0])
                #print(self.trajSave.xyz.shape)
                #------ reshape data ------------------------------


                xyz_history += xyz
                potentialEnergy_history += potentialEnergyList

                xyz_tr = xyz_history
                trajMD=md.Trajectory(xyz_tr, self.topology)

                if self.saveEnergy ==1:
                    potentialEnergyShort=potentialEnergy_history
                else:
                    potentialEnergyShort=None

                print('Current shape of traj is '+repr(trajMD.xyz.shape))
                while(len(trajMD)>maxDataLength):
                    trajMD=trajMD[::2]
                if self.saveEnergy ==1:
                    while(len(potentialEnergyShort)>maxDataLength):
                        potentialEnergyShort = potentialEnergyShort[::2]

                print('Sparsed shape of traj is '+repr(trajMD.xyz.shape))

                # if self.saveEnergy ==1:
                #     assert( len(potentialEnergyShort) == len(trajMD)), 'energy length does not match the trajectory length'
                if self.diffmap_metric == 'euclidean':
                    trajMD = trajMD.center_coordinates()
                    trajMD=trajMD.superpose(trajMD, 0)

                traj =  trajMD.xyz.reshape((trajMD.xyz.shape[0], self.trajSave.xyz.shape[1]*self.trajSave.xyz.shape[2]))


                #------ compute CV and reset initial conditions------------------------------

                if(self.corner==0):
                    V1, q, qEstimated, potEn, kernelDiff=dominantEigenvectorDiffusionMap(traj, self.epsilon, self, self.kT, self.method, energy = potentialEnergyShort, smoothedEnergy = self.smoothEnergyFlag, metric = self.diffmap_metric)

                    idxMaxV1 = np.argmax(np.abs(V1))

                    tmp=traj[idxMaxV1].reshape(self.trajSave.xyz[0].shape)
                    frontierPoint = tmp* self.integrator.model.x_unit

                    self.integrator.x0=frontierPoint
                    initialPositions=[frontierPoint for rep in range(0,nrRep)]

                else:

                    nrFEV = 2
                    # V1 is matrix with first nrFEV eigenvectors
                    V1, q, qEstimated, potEn, kernelDiff=dominantEigenvectorDiffusionMap(traj, self.epsilon, self, self.kT, self.method, nrOfFirstEigenVectors=nrFEV+1, energy = potentialEnergyShort, smoothedEnergy = self.smoothEnergyFlag, metric=self.diffmap_metric)


                    ### replace this part till **
                    # select a random point and compute distances to it
                    m = np.shape(V1)[0]
                    idx_corner = np.random.randint(m)

                    dist = scidist.cdist(V1[[idx_corner],:], V1)[0]

                    # find first cornerstone
                    idx_corner = [np.argmax(dist)]
                    # print('idx_corner ')
                    # print(idx_corner)
                    # iteration to find the other cornerstones
                    for k in np.arange(1, nrRep):
                        # update minimum distance to existing cornerstones
                        if(k>1):
                            dist = np.minimum(dist, scidist.cdist(V1[[idx_corner[-1]],:], V1)[0])
                        else:
                            dist = scidist.cdist(V1[idx_corner,:], V1)[0]
                        # select new cornerstone
                        idx_corner.append(np.argmax(dist))

                    ####
                    tmp=traj[idx_corner].reshape(np.append(nrRep, self.trajSave.xyz[0].shape))
                    frontierPoint = tmp
                    frontierPointSave = tmp

                    #every replica can get different initial condition
                    initialPositions=[frontierPoint[rep]* self.integrator.model.x_unit for rep in range(0,nrRep)]


                    #self.integrator.x0=frontierPoint  # what does this do? frontier point should be an array of initial conditions, one for each replica.
                                                      # I hope it gives every replica the right initial condition.

                    # ** #################################################

                    frontierPointMDTraj=md.Trajectory(frontierPoint, self.topology)
                    print('Saving frontier points')
                    frontierPointMDTraj.save(self.savingFolderFrontierPoints+'/frontierPoints_at_iteration_'+repr(it)+'.h5')

                if self.saveEigenvectors==1:
                    trajShort=traj.reshape((traj.shape[0],self.trajSave.xyz.shape[1],self.trajSave.xyz.shape[2]))
                    print('Shape of traj '+repr(traj.shape))
                    self.trajEVSave=md.Trajectory(trajShort, self.topology)
                    print('Saving traj used for diffmaps to file')
                    self.trajEVSave.save(self.savingFolderEigenvectors+'traj_'+repr(it)+'.h5')
                    print('Saving EV to file')
                    np.save(self.savingFolderEigenvectors+'V1_'+repr(it), V1)


                print('Saving traj to file')
                self.trajSave.save(self.savingFolder+'traj_'+repr(it)+'.h5')

                if self.saveEnergy==1:
                    print('Saving energy to file')
                    np.save(self.savingFolderEnergy+'E_'+repr(it), potentialEnergyList)


                if(it>0):
                    tavEnd=time.time()
                    self.timeAv.addSample(tavEnd-tav0)




#--------------------- Functions --------------------------------

###########################################################


def dominantEigenvectorDiffusionMap(tr, eps, sampler, T, method, nrOfFirstEigenVectors=2, numberOfNeighborsPotential=1, energy = None, smoothedEnergy = False, metric='euclidean'):

        print("Temperature in dominantEigenvectorDiffusionMap is "+repr(sampler.kT/sampler.model.kB_const))

        chosenmetric = metric #'euclidean'#dm.myRMSDmetricPrecentered#
        print('Chosen metric for diffusionmap is '+str(chosenmetric))

        X_FT =tr.reshape(tr.shape[0],sampler.model.testsystem.positions.shape[0],sampler.model.testsystem.positions.shape[1] )
        tr = align_with_mdanalysis(X_FT, sampler);

        E = computeEnergy(tr, sampler, modelShape=True)
        qTargetDistribution= computeTargetMeasure(E, sampler)

        if method=='PCA':

            X_pca = decomposition.TruncatedSVD(n_components=1).fit_transform(tr- np.mean(tr,axis=0))
            v1=X_pca[:,0]

            #qEstimated = qTargetDistribution
            #kernelDiff=[]

            if(nrOfFirstEigenVectors>2):
                print('For PCA, nrOfFirstEigenVectors must be 1')

        elif method == 'Diffmap':

            # if isinstance(tr, md.Trajectory):
            #     kernelDiff=dm.compute_kernel_mdtraj(tr, eps)
            # else:
            #
            #     kernelDiff=dm.compute_kernel(tr, eps)
            #
            # P=dm.compute_P(kernelDiff, tr)
            # qEstimated = kernelDiff.sum(axis=1)
            #
            # #print(eps)
            # #X_se = manifold.spectral_embedding(P, n_components = 1, eigen_solver = 'arpack')
            # #V1=X_se[:,0]
            #
            # lambdas, V = sps.linalg.eigsh(P, k=(nrOfFirstEigenVectors))#, which='LM' )
            # ix = lambdas.argsort()[::-1]
            # X_se= V[:,ix]
            #
            # v1=X_se[:,1:]
            # #v2=X_se[:,2]
            # #V2=X_se[:,1]

            mydmap = pydm.DiffusionMap(alpha = 0.5, n_evecs = 1, epsilon = eps,  k=1000, metric=chosenmetric)#, neighbor_params = {'n_jobs':-4})
            dmap = mydmap.fit_transform(tr)

            P = mydmap.P
            labmdas = mydmap.evals
            v1 = mydmap.evecs
            qEstimated = mytdmap.q
            kernelDiff=mytdmap.local_kernel

        elif method =='TMDiffmap': #target measure diffusion map

            # P, qEstimated, kernelDiff = dm.compute_unweighted_P( tr,eps, sampler, qTargetDistribution )
            # lambdas, eigenvectors = sps.linalg.eigs(P, k=(nrOfFirstEigenVectors))#, which='LM' )
            # lambdas = np.real(lambdas)
            #
            # ix = lambdas.argsort()[::-1]
            # X_se= eigenvectors[:,ix]
            # lambdas = lambdas[ix]
            #
            #
            # v1=np.real(X_se[:,1:])
            # # scale eigenvectors with eigenvalues
            # v1 = np.dot(v1,np.diag(lambdas[1:]))


            # traj = md.Trajectory(X_FT, mdl.testsystem.topology)
            # distances = np.empty((traj.n_frames, traj.n_frames))
            # for i in range(traj.n_frames):
            #     distances[i] = md.rmsd(traj, traj, i)
            # print(distances.shape)



            ##########
            if chosenmetric=='rmsd':
                trxyz=tr.reshape(tr.shape[0], sampler.model.testsystem.positions.shape[0],sampler.model.testsystem.positions.shape[1])
                traj = md.Trajectory(trxyz, sampler.model.testsystem.topology)

                indptr = [0]
                indices = []
                data = []
                k = 1000
                epsilon = eps

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

                n_evecs = 2

                evals, evecs = spsl.eigs(P, k=(n_evecs+1), which='LM')
                ix = evals.argsort()[::-1][1:]
                evals = np.real(evals[ix])
                evecs = np.real(evecs[:, ix])
                dmap = np.dot(evecs, np.diag(evals))


                labmdas = evals
                v1 = evecs
                qEstimated = q
                kernelDiff=local_kernel

            elif chosenmetric=='euclidean':

                epsilon=eps

                tr = tr.reshape(tr.shape[0], tr.shape[1]*tr.shape[2])

                mydmap = pydm.DiffusionMap(alpha = 1, n_evecs = 1, epsilon = epsilon,  k=100, metric='euclidean')#, neighbor_params = {'n_jobs':-4})
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


            ##########

            # mydmap = pydm.DiffusionMap(alpha = 1, n_evecs = 1, epsilon = eps,  k=1000, metric=chosenmetric)#, neighbor_params = {'n_jobs':-4})
            # dmap = mydmap.fit_transform(tr, weights = qTargetDistribution)
            #
            # P = mydmap.P
            # labmdas = mydmap.evals
            # v1 = mydmap.evecs
            # qEstimated = mydmap.q
            # kernelDiff=mydmap.local_kernel

        else:
            print('Error in sampler class: dimension_reduction function did not match any method.\n CHoose from: TMDiffmap, PCA, Diffmap')

        return v1, qTargetDistribution, qEstimated, E, kernelDiff



def computeTargetMeasure(Erecompute, smpl):

    qTargetDistribution=np.zeros(len(Erecompute))


    for i in range(0,len(Erecompute)):
                #Erecompute[i]=smpl.model.energy(X_FT[i,:,:]*smpl.model.x_unit).value_in_unit(smpl.model.energy_unit)
                tmp = Erecompute[i]
                betatimesH_unitless =tmp / smpl.kT.value_in_unit(smpl.model.energy_unit) #* smpl.model.temperature_unit
                qTargetDistribution[i]=np.exp(-(betatimesH_unitless))
    print('Done')

    return qTargetDistribution#, Erecompute

def computeEnergy(X_reshaped, smpl, modelShape = False):

    if modelShape:
        X_FT=X_reshaped
    else:
        X_FT =X_reshaped.reshape(X_reshaped.shape[0],smpl.model.testsystem.positions.shape[0],smpl.model.testsystem.positions.shape[1] )
    qTargetDistribution=np.zeros(len(X_FT))
    Erecompute=np.zeros(len(X_FT))
    from simtk import unit
    for i in range(0,len(X_FT)):
                Erecompute[i]=smpl.model.energy(X_FT[i]*smpl.model.x_unit).value_in_unit(smpl.model.energy_unit)

    return  Erecompute/2.0

def dimension_reduction(tr, eps, numberOfLandmarks, sampler, T, method):

        assert(False), 'under construction: dominantEigenvectorDiffusionMap has changed '
        v1, q, qEstimated, potEn, kernelDiff=dominantEigenvectorDiffusionMap(tr, eps, sampler, T, method)

            #get landmarks
        lm=dm.get_landmarks(tr, numberOfLandmarks, q , v1, potEn)

        assert( len(v1) == len(tr)), "Error in function dimension_reduction: length of trajectory and the eigenvector are not the same."

        #
        # plt.scatter(v1, v2)
        # plt.scatter(v1[lm], v2[lm], c='r')
        # plt.xlabel('V1')
        # plt.ylabel('V2')
        # plt.show()

        return lm, v1



def currentGridCV(tr, lm, v1):


        dim=tr.shape[1]

        gridCV_new=np.zeros([len(lm), dim+1])

        for i in range(0,len(lm)):
            for j in range(0,dim):
                    gridCV_new[i,j]=tr[lm[i],j]
            gridCV_new[i,dim]=v1[lm[i]]#
        return gridCV_new


def min_rmsd(X):

    #first frame
    X0 = X[0] -  rmsd.centroid(X[0])

    for i in range(0,len(X)):
    #print "RMSD before translation: ", rmsd.kabsch_rmsd(Y[0],Y[1::])

        X[i] = X[i] -  rmsd.centroid(X[i])
        X[i] = rmsd.kabsch_rotate(X[i], X0)

        ## alternatively use quaternion rotate  instead - does not work in this format
        #rot = rmsd.quaternion_rotate(X[i], X0)
        #X[i] = np.dot(X[i], rot)

    return X



#from MDAnalysis.tests.datafiles import PSF, DCD, PDB_small
def align_with_mdanalysis(X_FT, smpl):
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
