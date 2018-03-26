
"""
The main class for Diffusionmap-directed sampling
Author: Zofia Trstanova
Comment:

"""

import numpy as np
import integrator
import Averages as av
import model
import dimension_reduction as dimred

import scipy.spatial.distance as scidist
import time
import mdtraj as md

maxDataLength=5000
everyN=1
writingEveryNSteps=1
savingEveryNIterations=100
adjustEpsilon=1
saveModNr=10;

class Sampler():
    """
    Main class
    """


    def __init__(self, model, integrator, algorithm=0, numberOfDCs = 2, diffusionMapMetric='euclidean', dataFileName='/Data', dataFrontierPointsName = '/FrontierPoints', dataEigenVectorsName='/Eigenvectors', dataEnergyName='/Energies', diffusionMap='Diffmap'):

        self.model=model
        self.algorithm=algorithm
        self.integrator=integrator

        self.diffmap_metric = diffusionMapMetric # dm.myRMSDmetricPrecentered

        # number of diffusion coordinates to be computed in diffusion maps
        self.numberOfDCs = numberOfDCs

        #set the temperature here - then passed to the integrator
        self.kT=self.integrator.kT

        self.maxDataLength=maxDataLength

        self.timeAv=av.Average(0.0)

        self.dim=3
        self.dimCV=1

        self.method=diffusionMap #'TMDiffmap' or 'Diffmap'

        #diffusion maps constants
        self.epsilon='bgh' #0.1 #0.05
        if self.diffmap_metric == 'rmsd':
            self.epsilon = 0.1
            print('If the metric is rmsd, epsilon must have numeric value.')
        #self.epsilon_Max=self.epsilon* (2.0**4)

        #linear approximation
        self.numberOfLandmarks=10

        self.nrDecorrSteps=1

        self.modNr=10

        self.kTraj_LM_save=None
        self.V1_LM_save=None

        self.gridCV_long=None
        self.sampled_trajectory=np.array([self.integrator.x0.value_in_unit(self.model.x_unit)])
        self.landmarkedStates=None

        self.changeTemperature = 0
        self.corner = 0

        # local or global collective variables
        # by default : global
        # if 1, then the diffusion map cv is computed only from trajectory from the last iteration
        # note that for the local algo: vanilla diffusion map should provide good approximation, since the
        # quasi-stationary distribution is assumed to be unbiased
        self.local_CV = 0


        if self.algorithm==0:
            self.algorithmName='std'
        if self.algorithm==1:
            self.algorithmName='eftad_fixed_cv'
        if self.algorithm==2:
            self.algorithmName='initial_condition'
        if self.algorithm==3:
            self.algorithmName='frontier_points_corner_change_temperature'
        if self.algorithm==4:
            self.algorithmName='frontier_points_corner_change_temperature_off'
        if self.algorithm==5:
            self.algorithmName='local_frontier_points_corner_change_temperature_off'
        if self.algorithm==6:
            self.algorithmName='local_frontier_points_corner_change_temperature'
        if self.algorithm==7:
            self.algorithmName = 'DiffmapABF'
        if self.algorithm==8:
            self.algorithmName = 'local_frontier_points'

        #print(self.model.testsystem.topology)
        self.topology = md.Topology().from_openmm(self.model.testsystem.topology)
        self.trajSave= md.Trajectory(self.model.positions.value_in_unit(self.model.x_unit), self.topology)
        self.savingFolder=dataFileName+'/'+self.model.modelName
        self.savingFolderEnergy=dataEnergyName
        self.savingFolderFrontierPoints = dataFrontierPointsName
        self.savingFolderEigenvectors = dataEigenVectorsName

        self.saveEigenvectors=1
        self.saveEnergy=1

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

    def run_local_frontier_points_corner_change_temperature_off(self, nrSteps, nrIterations, nrRep):

        self.local_CV = 1
        self.changeTemperature = 0
        self.corner = 1
        self.run_frontierPoints(nrSteps, nrIterations, nrRep)

    def run_local_frontier_points_corner_change_temperature(self, nrSteps, nrIterations, nrRep):

        self.local_CV = 1
        self.changeTemperature = 1
        self.corner = 1
        self.run_frontierPoints(nrSteps, nrIterations, nrRep)

    def run_local_frontier_points(self, nrSteps, nrIterations, nrRep):

        self.local_CV = 1
        self.changeTemperature = 0
        self.corner = 0
        self.run_frontierPoints(nrSteps, nrIterations, nrRep)

######----------------- STD ---------------------------------

    def run_std(self, nrSteps, nrIterations, nrRep):


            #reset time
            self.timeAv.clear()

            print('replicated std dynamics with '+repr(nrRep)+' replicas')

            initialPositions=[self.integrator.x0 for rep in range(0,nrRep)]
            initialVelocities=[self.integrator.v0 for rep in range(0,nrRep)]
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
                    self.integrator.v0=initialVelocities[rep]

                    #xyz_iter, potEnergy = self.integrator.run_langevin(nrSteps, save_interval=self.modNr) # local Python
                    xyz_iter, potEnergy = self.integrator.run_openmm_langevin(nrSteps, save_interval=self.modNr)
                    xyz += xyz_iter
                    if self.saveEnergy==1:
                        potentialEnergyList +=potEnergy
                    # #only for replicas
                    initialPositions[rep] = self.integrator.xEnd
                    initialVelocities[rep] = self.integrator.vEnd

                #tmpselftraj=np.copy(self.sampled_trajectory)
                #self.sampled_trajectory=np.concatenate((tmpselftraj,np.copy(Xrep[-1].value_in_unit(self.model.x_unit))))

                self.trajSave=md.Trajectory(xyz, self.topology)
                # align frames using mdtraj
                self.trajSave = self.trajSave.center_coordinates()
                self.trajSave=self.trajSave.superpose(self.trajSave, 0)

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
            initialVelocities=[self.integrator.v0 for rep in range(0,nrRep)]
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
                    self.integrator.v0=initialVelocities[rep]

                    xs,vx=self.integrator.run_EFTAD(nrSteps)
                    initialPositions[rep] = xs[-1]
                    initialVelocities[rep] =vx[-1]

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
            initialVelocities=[self.integrator.v0 for rep in range(0,nrRep)]
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
                    self.integrator.v0=initialVelocities[rep]

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
                    initialVelocities[rep] = self.integrator.v0


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

            if (self.local_CV == 1):
                print('Local CV: diffusion map computed from trajectory from the previous iteration.')
            else:
                print('Gloval CV: diffusion map computed from trajectory from the whole history.')

            #intialisation
            initialPositions=[self.integrator.x0 for rep in range(0,nrRep)]
            initialVelocities=[self.integrator.v0 for rep in range(0,nrRep)]
            ##############################

            xyz_history = list()
            potentialEnergy_history = list()

            for it in range(0,nrIterations):

                if(it>0):
                    tav0=time.time()

                if(np.remainder(it, writingEveryNSteps)==0):
                    print('*******************************')
                    print('********Iteration '+ repr(it)+' *****')
                    print('*******************************')
                    print('Kinetic Temperature is '+str(self.integrator.kineticTemperature.getAverage()))


                    if(it>0):
                        t_left=(self.timeAv.getAverage())*(nrIterations-it)
                        print(time.strftime("Time left %H:%M:%S", time.gmtime(t_left)))

                #--------------------
                #change temperature for the integrator-> the target termperature (passed to diffusionmaps) remains unchanged
                if(self.changeTemperature == 1):
                    if(it>0):

                        T = self.kT/ self.model.kB_const
                        self.integrator.temperature = T+200  * self.model.temperature_unit
                        print("Changing temperature to T="+repr(self.integrator.temperature))

                        print('Simulating at higher temperature'+repr(self.integrator.temperature))

                #------- simulate Langevin
                    xyz = list()
                    potentialEnergyList=list()

                    # ratio of steps at higher temperature
                    ratioStepsHigherTemperature=0.1

                    for rep in range(0, nrRep):

                        # each intial replica has initial condition
                        self.integrator.x0=initialPositions[rep]
                        self.integrator.v0=initialVelocities[rep]

                        #xyz_iter=self.integrator.run_langevin(nrSteps, save_interval=self.modNr) # local Python
                        xyz_iter, potEnergy = self.integrator.run_openmm_langevin(int(ratioStepsHigherTemperature*nrSteps), save_interval=self.modNr)
                        xyz += xyz_iter
                        if self.saveEnergy==1:
                            potentialEnergyList +=potEnergy
                        # #only for replicas
                        initialPositions[rep] = self.integrator.xEnd
                        initialVelocities[rep] = self.integrator.vEnd

                    #reset temperature

                    self.integrator.temperature = self.kT/ self.model.kB_const
                    print("Changing temperature back to T="+repr(self.integrator.temperature))

                    print('Simulating at target temperature '+ repr(self.integrator.temperature))

                    for rep in range(0, nrRep):

                        # each intial replica has initial condition
                        self.integrator.x0=initialPositions[rep]
                        self.integrator.v0=initialVelocities[rep]

                        #xyz_iter=self.integrator.run_langevin(nrSteps, save_interval=self.modNr) # local Python
                        xyz_iter, potEnergy = self.integrator.run_openmm_langevin(int((1.0-ratioStepsHigherTemperature)*nrSteps), save_interval=self.modNr)
                        xyz += xyz_iter
                        if self.saveEnergy==1:
                            potentialEnergyList +=potEnergy
                        # #only for replicas
                        initialPositions[rep] = self.integrator.xEnd
                        initialVelocities[rep] = self.integrator.vEnd

                #elif(self.changeTemperature == 0 or it> nrIterations-2):
                else:
                    xyz = list()
                    potentialEnergyList=list()

                    for rep in range(0, nrRep):

                        # each intial replica has initial condition
                        self.integrator.x0=initialPositions[rep]
                        self.integrator.v0=initialVelocities[rep]

                        #xyz_iter=self.integrator.run_langevin(nrSteps, save_interval=self.modNr) # local Python
                        xyz_iter, potEnergy= self.integrator.run_openmm_langevin(nrSteps, save_interval=self.modNr)
                        xyz += xyz_iter
                        if self.saveEnergy==1:
                            potentialEnergyList +=potEnergy

                        initialPositions[rep] = self.integrator.xEnd
                        initialVelocities[rep] = self.integrator.vEnd


                # creat md traj object
                self.trajSave=md.Trajectory(xyz, self.topology)

                # align frames using mdtraj
                self.trajSave = self.trajSave.center_coordinates()
                self.trajSave=self.trajSave.superpose(self.trajSave, 0)
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

                # assign points that will be used for diffusion maps
                if (self.local_CV == 1):
                    #  history is only from the previous iteration
                    xyz_history = xyz
                    potentialEnergy_history = potentialEnergyList
                else:
                    # accumulate history of all points
                    xyz_history += xyz
                    potentialEnergy_history += potentialEnergyList

                xyz_tr = xyz_history
                trajMD=md.Trajectory(xyz_tr, self.topology)

                if self.saveEnergy ==1:
                    potentialEnergyShort=potentialEnergy_history
                else:
                    potentialEnergyShort=None

                # stride the trajctory and check the convergence of eigenvalues
                # if it isnt sufficient, stride less

                maxNumberOfStrideAttempts=3
                self.check_convergence_eigenvalues=0
                nrOfStrideAttempt=0
                errorToleranceEigenvalues = 0.001

                errEW= 1

                maxDataLengthLoc=self.maxDataLength

                while (errEW > errorToleranceEigenvalues and nrOfStrideAttempt<maxNumberOfStrideAttempts):

                        if self.check_convergence_eigenvalues ==0:
                            nrOfStrideAttempt=maxNumberOfStrideAttempts
                        else:
                            nrOfStrideAttempt += 1

                        print('*******************************')
                        print('Stride iteration '+repr(nrOfStrideAttempt))

                        if nrOfStrideAttempt>1:
                            maxDataLengthLoc = maxDataLengthLoc+1000

                        print('Maximal local data lenght '+repr(maxDataLengthLoc))

                        print('Current shape of traj is '+repr(trajMD.xyz.shape))
                        while(len(trajMD)>maxDataLengthLoc):
                            trajMD=trajMD[::2]
                        if self.saveEnergy ==1:
                            while(len(potentialEnergyShort)>maxDataLengthLoc):
                                potentialEnergyShort = potentialEnergyShort[::2]

                        print('Sparsed shape of traj is '+repr(trajMD.xyz.shape))

                        # if self.saveEnergy ==1:
                        #     assert( len(potentialEnergyShort) == len(trajMD)), 'energy length does not match the trajectory length'
                        # if self.diffmap_metric == 'euclidean':
                        #     trajMD = trajMD.center_coordinates()
                        #     trajMD=trajMD.superpose(trajMD, 0)

                        # reshape trajectories, because dimension reduction takes array steps x nDOF
                        print('Reshaping the trajectory.')
                        traj =  trajMD.xyz.reshape((trajMD.xyz.shape[0], self.trajSave.xyz.shape[1]*self.trajSave.xyz.shape[2]))


                        #------ compute CV and reset initial conditions------------------------------
                        print('Find new initial conditions.')

                        # nrFEV: V1 will matrix with first nrFEV eigenvectors


                        if(self.corner==0):
                            nrFEV = 1
                        else:
                            nrFEV = self.numberOfDCs

                        dominantEV, q, qEstimated, potEn, kernelDiff, eigenvalues=dimred.dominantEigenvectorDiffusionMap(traj, self.epsilon, self, self.kT, self.method, nrOfFirstEigenVectors=nrFEV+1, energy = potentialEnergyShort,  metric=self.diffmap_metric)
                        # skip the zeroth eigenvector
                        V1 = dominantEV[:,1:]

                        lastTrajSteps = 10
                        if lastTrajSteps>= traj.shape[0]:
                            lastTrajSteps = int(0.1*traj.shape[0])

                        if self.check_convergence_eigenvalues:

                            V1_short, q_short, qEstimated_short, potEn_short, kernelDiff_short, eigenvalues_short=dimred.dominantEigenvectorDiffusionMap(traj[:-lastTrajSteps], self.epsilon, self, self.kT, self.method, nrOfFirstEigenVectors=nrFEV+1, energy = potentialEnergyShort,  metric=self.diffmap_metric)
                            errEW = np.abs(eigenvalues[0]-eigenvalues_short[0])
                            print('Error of the eigenvalues in length n and n-100 is '+repr(errEW))
                            print('error EW is '+repr(errEW))
                            print('tolerance is '+repr(errorToleranceEigenvalues))


                if(self.corner==0):

                    idxMaxV1 = np.argmax(np.abs(V1))

                    tmp=traj[idxMaxV1].reshape(self.trajSave.xyz[0].shape)
                    frontierPoint = tmp* self.integrator.model.x_unit

                    tmp=traj[idxMaxV1].reshape( self.trajSave.xyz[0].shape)
                    frontierPoint = tmp
                    frontierPointSave = tmp

                    self.integrator.x0=frontierPoint
                    initialPositions=[frontierPoint for rep in range(0,nrRep)]

                else:
                    ### replace this part till **
                    # select a random point and compute distances to it
                    m = np.shape(V1)[0]
                    idx_corner = np.random.randint(m)

                    dist = scidist.cdist(V1[[idx_corner],:], V1)[0]

                    # find first cornerstone
                    idx_corner = [np.argmax(dist)]

                    # # # ## take the point with maximal value of the first ev
                    # idx_corner = np.argmax(np.abs(V1[:,0]))
                    # dist = scidist.cdist(V1[[idx_corner],:], V1)[0]
                    #
                    # idx_corner=[idx_corner]
                    # #
                    # #idx_corner = [np.argmax(dist)]


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


######################################################

##------------------------------------------
    def run_DiffmapABF(self, nrSteps, nrIterations, nrRep):
            """
            DiffmapABF algorithm: run Langevin dynamics, compute diffusion maps, and apply biasing force
            """
            print('Error: ABF not working yet. ')

            #reset time
            self.timeAv.clear()

            print('run_DiffmapABF: replicated std dynamics with '+repr(nrRep)+' replicas')
            self.local_CV = 0

            #intialisation
            initialPositions=[self.integrator.x0 for rep in range(0,nrRep)]
            initialVelocities=[self.integrator.v0 for rep in range(0,nrRep)]
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

                xyz = list()
                potentialEnergyList=list()

                #------- simulate Langevin
                if it == 0:
                    print('Running standard simulation')
                    for rep in range(0, nrRep):

                        # each intial replica has initial condition
                        self.integrator.x0=initialPositions[rep]
                        self.integrator.v0=initialVelocities[rep]

                        xyz_iter, potEnergy = self.integrator.run_langevin(nrSteps, save_interval=self.modNr)
                        xyz += xyz_iter
                        if self.saveEnergy==1:
                            potentialEnergyList +=potEnergy
                        # #only for replicas
                        initialPositions[rep] = self.integrator.xEnd
                        initialVelocities[rep] = self.integrator.vEnd
                #else:
                if it>0 :#and it % 2 ==0:
                #------- simulate ABF
                    print('Running ABF')
                    for rep in range(0, nrRep):

                        # each intial replica has initial condition
                        self.integrator.x0=initialPositions[rep]
                        self.integrator.v0=initialVelocities[rep]

                        xyz_iter, potEnergy = self.integrator.run_langevin_ABF(nrSteps, self.X, self.V, save_interval=self.modNr)
                        #xyz_iter, potEnergy = self.integrator.run_langevin(nrSteps, save_interval=self.modNr)
                        xyz += xyz_iter
                        if self.saveEnergy==1:
                            potentialEnergyList +=potEnergy
                        # #only for replicas
                        initialPositions[rep] = self.integrator.xEnd
                        initialVelocities[rep] = self.integrator.vEnd


                # creat md traj object
                self.trajSave=md.Trajectory(xyz, self.topology)

                # align frames using mdtraj
                self.trajSave = self.trajSave.center_coordinates()
                self.trajSave=self.trajSave.superpose(self.trajSave, 0)
                #self.trajSave = self.trajSave.center_coordinates()

                # assign points that will be used for diffusion maps
                if (self.local_CV == 1):
                    #  history is only from the previous iteration
                    xyz_history = xyz
                    potentialEnergy_history = potentialEnergyList
                else:
                    # accumulate history of all points
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

                # reshape trajectories, because dimension reduction takes array steps x nDOF
                print('Reshaping the trajectory.')
                traj =  trajMD.xyz.reshape((trajMD.xyz.shape[0], self.trajSave.xyz.shape[1]*self.trajSave.xyz.shape[2]))

                nrFEV=1
                dominantEV, q, qEstimated, potEn, kernelDiff, eigenvalues=dimred.dominantEigenvectorDiffusionMap(traj, self.epsilon, self, self.kT, self.method, nrOfFirstEigenVectors=nrFEV+1,   metric=self.diffmap_metric)
                V1=dominantEV[:,0]

                #V1, q, qEstimated, potEn, kernelDiff=dimred.dominantEigenvectorDiffusionMap(traj, self.epsilon, self.kT, self.method, energy = potentialEnergyShort, metric = self.diffmap_metric)

                self.X = traj
                self.V = V1

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
