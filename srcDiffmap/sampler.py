
"""
The main class for Diffusionmap-directed sampling
Author: Zofia Trstanova
Comment:
The standard dynamics is in runStd
The diffmap dircted sampling is in runDiffmapCV_Eftad_local.
The rest of the functions is in progress or not working.

"""

import numpy as np
import matplotlib.pyplot as plt

import diffusionmap as dm

import integrator
import linear_approximation as aprx
import Averages as av
import model

import scipy.sparse as sps
import rmsd

import time
import mdtraj as md

maxDataLength=2000
everyN=1
writingEveryNSteps=1
savingEveryNIterations=100
adjustEpsilon=1

class Sampler():

    def __init__(self, model, integrator, algorithm=0, dataFileName='/Data'):

        self.model=model
        self.algorithm=algorithm
        self.integrator=integrator

        self.T=self.integrator.temperature

        self.timeAv=av.Average(0.0)

        self.x0=np.array([1.0,0.0])#,0]
        self.p0=[0.0,0.0]#,0]

        self.dim=2
        self.dimCV=1

        self.method='DiffMap'

        #diffusion maps constants
        self.epsilon=0.5 #0.05
        self.epsilon_Max=self.epsilon* (2.0**4)

        #linear approximation
        self.numberOfLandmarks=10

        self.nrDecorrSteps=1

        self.modNr=10
        self.modEFTAD=10

        self.Traj_LM_save=None
        self.V1_LM_save=None

        self.gridCV_long=None
        self.sampled_trajectory=np.array([self.integrator.x0.value_in_unit(self.model.x_unit)])
        self.landmarkedStates=None

        if self.algorithm==0:
            self.algorithmName='std'
        if self.algorithm==1:
            self.algorithmName='eftad_fixed_cv'
        if self.algorithm==2:
            self.algorithmName='eftad_diffmap_local'
        if self.algorithm==3:
            self.algorithmName='eftad_diffmap_only'
        if self.algorithm==4:
            self.algorithmName='eftad_diffmap_kinEn'
        if self.algorithm==5:
            self.algorithmName='modif_kinEn_force'
        if self.algorithm==6:
            self.algorithmName='modif_kinEn'




        self.topology = md.Topology().from_openmm(self.model.testsystem.topology)
        self.trajSave= md.Trajectory(self.model.positions.value_in_unit(self.model.x_unit), self.topology)
        self.savingFolder=dataFileName

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

        # Look up function automatically
        #run_function = getattr(self, 'run' + self.algorithmName)
        #try:
        #    run_function()
        #exception Exception as e:
        #    raise("Could not find run function '%s'" % ('run' + self.algorithmName))

        if self.algorithm==0:
            self.runStd(nrSteps, nrIterations, nrRep)
        if self.algorithm==1:
            self.runEftadFixedCV(nrSteps, nrIterations, nrRep)
        if self.algorithm==2:
            self.runDiffmapCV_Eftad_local(nrSteps, nrIterations, nrRep)
        if self.algorithm==3:
            self.runDiffmapCV_Eftad_only(nrSteps, nrIterations, nrRep)
        if self.algorithm==4:
            self.runDiffmap_modifKinEn_local(nrSteps, nrIterations, nrRep)
        if self.algorithm==5:
            self.runKinEnForce(nrSteps, nrIterations, nrRep)
        if self.algorithm==6:
            self.runModifiedKineticEnergy(nrSteps, nrIterations, nrRep)
######----------------- STD ---------------------------------

    def runStd(self, nrSteps, nrIterations, nrRep):

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

                    if(it>0):
                        t_left=(self.timeAv.getAverage())*(nrIterations-it)
                        print(time.strftime("Time left %H:%M:%S", time.gmtime(t_left)))

                #------- simulate Langevin

                xyz = list()

                for rep in range(0, nrRep):

                    # TODO: Also track initial and final velocities

                    self.integrator.x0=initialPositions[rep]
                    xyz += self.integrator.run_openmm_langevin(nrSteps, save_interval=self.modNr)
                    initialPositions[rep] = self.integrator.xEnd

                #for n in range(0,nrRep):
                #    Xit[it* (nrRep*nrSteps) + n]=Xrep[n]

                if(it>0):
                    tavEnd=time.time()
                    self.timeAv.addSample(tavEnd-tav0)

                #tmpselftraj=np.copy(self.sampled_trajectory)
                #self.sampled_trajectory=np.concatenate((tmpselftraj,np.copy(Xrep[-1].value_in_unit(self.model.x_unit))))

                self.trajSave=md.Trajectory(xyz, self.topology)
                print 'Saving traj to file'
                self.trajSave.save(self.savingFolder+self.algorithmName+'_traj_'+repr(it)+'.h5')
                #if (nrSteps*nrIterations*nrRep)> 10**6:
            #    self.sampled_trajectory=Xrep[::modNr**2]
            #else:
            #    self.sampled_trajectory=Xit

#----------------- TAMD/EFTAD

    def runEftadFixedCV(self,nrSteps, nrIterations, nrRep):
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

                print 'Saving traj to file'
                self.trajSave.save(self.savingFolder+self.algorithmName+'_traj_'+repr(it)+'.h5')


#----------------- TAMD/EFTAD

    def runDiffmapCV_Eftad_local(self, nrSteps, nrIterations, nrRep):
        # use the eftad with CV obtained from diffusion map

            #reset time
            self.timeAv.clear()
            nrStepsEftad=int(2.0*nrSteps)
            self.nrDecorrSteps=nrStepsEftad

            print('The eftad with CV obtained from diffusion map. Langevin - '+repr(nrRep)+' replicas')

            initialPositions=[self.integrator.x0 for rep in range(0,nrRep)]
            #Xit=[self.model.positions for i in range(nrIterations*nrRep*nrSteps)]

            self.landmarkedStates=[self.model.positions for i in range(nrIterations * self.numberOfLandmarks)]
            ##############################

            for it in range(0,nrIterations):

                Xrep=[self.model.positions for i in range(nrRep*nrSteps)]

                if(it>0):
                    tav0=time.time()

                if(np.remainder(it, writingEveryNSteps)==0):
                    print('Iteration '+ repr(it))
                    print ('Kinetic Temperature is '+str(self.integrator.kineticTemperature.getAverage()))


                    if(it>0):
                        t_left=(self.timeAv.getAverage())*(nrIterations-it)
                        print(time.strftime("Time left %H:%M:%S", time.gmtime(t_left)))

                #------- simulate Langevin
                for rep in range(0, nrRep):

                    self.integrator.x0=initialPositions[rep]
                    xs,vx=self.integrator.run_langevin(nrSteps)
                    initialPositions[rep]=xs[-1]


                    for n in range(0,nrSteps):
                        Xrep[rep*nrSteps + n]=xs[n]

                 #-----save trajectory from the current Iteration
                #for n in range(0,nrRep*nrSteps):
                #    Xit[it* (nrRep*nrSteps) + n]=Xrep[n]

                # creat md traj object
                xyz=[x.value_in_unit(self.model.x_unit) for x in Xrep[::self.modNr]]
                self.trajSave=md.Trajectory(xyz, self.topology)
                #------ rmsd ------------------------------

                #Y=min_rmsd(Xrep[::self.modNr])
                #Y=Y*self.integrator.model.x_unit

                # align all the frames wrt to the first one according to minimal rmsd
                self.trajSave.superpose(self.trajSave[0])

                #------ reshape data ------------------------------

                traj =  self.trajSave.xyz.reshape((self.trajSave.xyz.shape[0], self.trajSave.xyz.shape[1]*self.trajSave.xyz.shape[2]))
                #traj=reshapeData(self.integrator.model, Y)
                #traj=[self.trajSave.xyz]

                #------ compute CV ------------------------------

                landmarks, V1 = dimension_reduction(traj, self.epsilon, self.numberOfLandmarks, self.model, self.T, self.method)

                # change the epsilon if there is not enough of landmarks
                escape=0

                while escape==0:
                    if len(np.unique(landmarks)) < 0.8*len(landmarks) and self.epsilon < self.epsilon_Max:
                        print('Increasing epsilon: epsilon = '+repr(self.epsilon))
                        self.epsilon =self.epsilon*2
                        landmarks, V1 = dimension_reduction(traj, self.epsilon, self.numberOfLandmarks, self.model, self.T, self.method)
                    else:
                        escape=1


                for nL in range(0, self.numberOfLandmarks):
                    self.landmarkedStates[it*self.numberOfLandmarks+nL]=traj[landmarks]

                #traj=reshapeData(self.integrator.model, Y)

                dataLM=traj[landmarks]#* self.integrator.model.x_unit
                VLM=V1[landmarks]#*self.integrator.model.x_unit


                #dataLM=reshapeDataBack(dataLM, self.integrator.model.x_unit)

                #dataLM=dataLM * self.integrator.model.x_unit
                #VLM=VLM* self.integrator.model.x_unit

                #compute piecewise derivatives
                v=aprx.compute_all_derivatives(dataLM, V1[landmarks])#*
                #v=reshapeDataBack(v, self.integrator.model.velocity_unit)
                #print v.shape

                ## find frontier point to start tamd from this point
                idxMaxV1 = np.argmax(np.abs(V1))
                #idxMinV1 = np.argmin(np.abs(V1))
                tmp=traj[idxMaxV1].reshape(self.trajSave.xyz[0].shape)
                frontierPoint = tmp* self.integrator.model.x_unit
                #------- simulate eftad and reset initial positions for the next iteration

                self.integrator.x0=initialPositions[0]#frontierPoint
                #self.integrator.x0=frontierPoint
                xEftad,vEftad=self.integrator.run_EFTAD_adaptive(nrStepsEftad,  dataLM, VLM, v)

                #print xEftad[-1]
                if(np.isnan(xEftad[-1]).any() ):
                    print('TAMD/AFED is nan, resetting trajectory.')
                    self.integrator.x0=initialPositions[0]
                else:

                    self.integrator.x0=xEftad[-1]
                xsdecorr,vxdecorr=self.integrator.run_langevin(self.nrDecorrSteps)

                initialPositions=[xsdecorr[-1] for rep in range(0,nrRep)]


                if(it>0):
                    tavEnd=time.time()
                    self.timeAv.addSample(tavEnd-tav0)

                # xyz=[x.value_in_unit(self.model.x_unit) for x in Xrep[::self.modNr]]
                # self.trajSave=md.Trajectory(xyz, self.topology)

                xyz_eftad=[x.value_in_unit(self.model.x_unit) for x in xEftad[::self.modNr]]
                trajEftad=md.Trajectory(xyz_eftad, self.topology)

                print 'Saving traj to file'
                self.trajSave = self.trajSave[::self.modNr]
                self.trajSave.save(self.savingFolder+self.algorithmName+'_traj_'+repr(it)+'.h5')
                trajEftad.save(self.savingFolder+self.algorithmName+'_trajTAMD_'+repr(it)+'.h5')

#------------- TAMD only - also for constructing CV


    def runDiffmapCV_Eftad_only(self, nrSteps, nrIterations, nrRep):
        # use the eftad with CV obtained from diffusion map as for sampling

            #reset time
            self.timeAv.clear()
            nrStepsEftad=int(nrSteps)

            print('The eftad (only) with CV obtained from diffusion map: - '+repr(nrRep)+' replicas. No Langevin!\n')

            initialPositions=[self.integrator.x0 for rep in range(0,nrRep)]
            Xit=[self.model.positions for i in range(nrIterations*nrRep*nrSteps)]
            Xrep=[self.model.positions for i in range(nrRep*nrSteps)]
            self.landmarkedStates=[self.model.positions for i in range(nrIterations * self.numberOfLandmarks)]
            ##############################

            for it in range(0,nrIterations):

                if(it>0):
                    tav0=time.time()

                if(np.remainder(it, writingEveryNSteps)==0):
                    print('Iteration '+ repr(it))

                    if(it>0):
                        t_left=(self.timeAv.getAverage())*(nrIterations-it)
                        print(time.strftime("Time left %H:%M:%S", time.gmtime(t_left)))

                #------- at the first interation- simulate Langevin- then Eftad/tamd
                for rep in range(0, nrRep):

                    self.integrator.x0=initialPositions[rep]
                    if it==0:
                        xs,vx=self.integrator.run_langevin(nrSteps)
                    else:
                        xs,vx=self.integrator.run_EFTAD_adaptive(nrStepsEftad,  dataLM, VLM, v)
                    initialPositions[rep]=xs[-1]

                    for n in range(0,nrSteps):
                        Xrep[rep*nrSteps + n]=xs[n]

                 #-----save trajectory from the current Iteration
                for n in range(0,nrRep*nrSteps):
                    Xit[it* (nrRep*nrSteps) + n]=Xrep[n]

                #------ rmsd ------------------------------
                #Y=Xrep[::self.modNr]
                Y=min_rmsd(Xrep[::self.modNr])
                Y=Y*self.integrator.model.x_unit

                #------ reshape data ------------------------------

                traj=reshapeData(self.integrator.model, Y)


                #------ compute CV ------------------------------

                landmarks, V1 = dimension_reduction(traj, self.epsilon, self.numberOfLandmarks, self.model, self.T, self.method)

                # change the epsilon if there is not enough of landmarks
                escape=0

                while escape==0:
                    if len(np.unique(landmarks)) < 0.8*len(landmarks) and self.epsilon < self.epsilon_Max:
                        print('Increasing epsilon: epsilon = '+repr(self.epsilon))
                        self.epsilon =self.epsilon*2
                        landmarks, V1 = dimension_reduction(traj, self.epsilon, self.numberOfLandmarks, self.model, self.T, self.method)
                    else:
                        escape=1

                for nL in range(0, self.numberOfLandmarks):
                    self.landmarkedStates[it*self.numberOfLandmarks+nL]=Xit[landmarks[nL]]


                dataLM=traj[landmarks, :]* self.integrator.model.x_unit
                VLM=V1[landmarks]#*self.integrator.model.x_unit


                #compute piecewise derivatives
                v=aprx.compute_all_derivatives(traj[landmarks, :], V1[landmarks])#*

                if(it>0):
                    tavEnd=time.time()
                    self.timeAv.addSample(tavEnd-tav0)

                xyz=[x.value_in_unit(self.model.x_unit) for x in Xrep[::self.modNr]]
                self.trajSave=md.Trajectory(xyz, self.topology)

                print 'Saving traj to file'
                self.trajSave.save(self.savingFolder+self.algorithmName+'_traj_'+repr(it)+'.h5')


#----------------- modified Kinetic energy

    def runDiffmap_modifKinEn_local(self, nrSteps, nrIterations, nrRep):
        # use modified kinetic energy with CV instead of Langevin dynamics
            eftadFlag=0
            #reset time
            self.timeAv.clear()
            nrStepsEftad=nrRep*nrSteps;#int(0.5*nrSteps)

            print('Modified kinetic energy Langevin dynamics: '+repr(nrRep)+' replicas')

            initialPositions=[self.integrator.x0 for rep in range(0,nrRep)]
            #Xit=[self.model.positions for i in range(nrIterations*nrRep*nrSteps)]

            self.landmarkedStates=[self.model.positions for i in range(nrIterations * self.numberOfLandmarks)]
            ##############################

            for it in range(0,nrIterations):

                Xrep=[self.model.positions for i in range(nrRep*nrSteps)]

                if(it>0):
                    tav0=time.time()

                if(np.remainder(it, writingEveryNSteps)==0):
                    print('Iteration '+ repr(it))

                    if(it>0):
                        t_left=(self.timeAv.getAverage())*(nrIterations-it)
                        print(time.strftime("Time left %H:%M:%S", time.gmtime(t_left)))

                #------- simulate Langevin
                for rep in range(0, nrRep):

                    self.integrator.x0=initialPositions[rep]
                    if it==0:
                        xs,vx=self.integrator.run_langevin(nrSteps)
                    else:
                        xs,vx=self.integrator.run_modifKinEn_Langevin(nrSteps, dataLM, VLM, v)
                    initialPositions[rep]=xs[-1]

                    for n in range(0,nrSteps):
                        Xrep[rep*nrSteps + n]=xs[n]

                 #-----save trajectory from the current Iteration
                #for n in range(0,nrRep*nrSteps):
                #    Xit[it* (nrRep*nrSteps) + n]=Xrep[n]

                #------ rmsd ------------------------------
                #Y=Xrep[::self.modNr]
                Y=min_rmsd(Xrep[::self.modNr])
                Y=Y*self.integrator.model.x_unit

                #------ reshape data ------------------------------

                traj=reshapeData(self.integrator.model, Y)

                #------ compute CV ------------------------------

                landmarks, V1 = dimension_reduction(traj, self.epsilon, self.numberOfLandmarks, self.model, self.T, self.method)

                # change the epsilon if there is not enough of landmarks
                escape=0

                while escape==0:
                    if len(np.unique(landmarks)) < 0.8*len(landmarks) and self.epsilon < self.epsilon_Max:
                        print('Increasing epsilon: epsilon = '+repr(self.epsilon))
                        self.epsilon =self.epsilon*2
                        landmarks, V1 = dimension_reduction(traj, self.epsilon, self.numberOfLandmarks, self.model, self.T, self.method)
                    else:
                        escape=1


                for nL in range(0, self.numberOfLandmarks):
                    self.landmarkedStates[it*self.numberOfLandmarks+nL]=Xit[landmarks[nL]]



                #print len(np.unique(landmarks))

                dataLM=traj[landmarks, :]* self.integrator.model.x_unit

                VLM=V1[landmarks]#*self.integrator.model.x_unit

                #compute piecewise derivatives
                v=aprx.compute_all_derivatives(traj[landmarks, :], V1[landmarks])#*
                #v=reshapeDataBack(v, self.integrator.model.velocity_unit)
                #print v.shape

                ##simulate EFTAD of EFTAD_kinEn
                #trajEFTAD_full,pEFTAD, z_full=self.std.simulate_eftad(n_stepsEFTAD, dataLM, VLM, v, ComputeForcing=0)


                #------- simulate eftad and reset initial positions for the next iteration

                if eftadFlag:
                    self.integrator.x0=initialPositions[0]
                    xEftad,vEftad=self.integrator.run_EFTAD_adaptive(nrStepsEftad,  dataLM, VLM, v)

                    initialPositions=[xEftad[-1] for rep in range(0,nrRep)]

                tavEnd=time.time()
                self.timeAv.addSample(tavEnd-tav0)

                xyz=[x.value_in_unit(self.model.x_unit) for x in Xrep[::self.modNr]]
                self.trajSave=md.Trajectory(xyz, self.topology)

                print 'Saving traj to file'
                self.trajSave.save(self.savingFolder+self.algorithmName+'_traj_'+repr(it)+'.h5')

######----------------- pure kin en from force ---------------------------------

    def runKinEnForce(self, nrSteps, nrIterations, nrRep):


            #reset time
            self.timeAv.clear()


            print('modified kinetic energy (with force) dynamics with '+repr(nrRep)+' replicas')


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

                Xrep=[self.model.positions for i in range(0,nrRep*nrSteps)]

                for rep in range(0, nrRep):

                    self.integrator.x0=initialPositions[rep]
                    xs,vx=self.integrator.run_modifKinEn_Langevin(nrSteps)
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

                print 'Saving traj to file'
                self.trajSave.save(self.savingFolder+self.algorithmName+'_traj_'+repr(it)+'.h5')


######----------------- modif kin Energy ---------------------------------

    def runModifiedKineticEnergy(self, nrSteps, nrIterations, nrRep):

            #reset time
            self.timeAv.clear()

            print('replicated modified kinetic energy dynamics with '+repr(nrRep)+' replicas')

            initialPositions=[self.integrator.x0 for rep in range(0,nrRep)]
            #Xit=[self.integrator.x0 for i in range(nrRep*nrSteps)]

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

                Xrep=[self.model.positions for i in range(0,nrRep*nrSteps)]

                for rep in range(0, nrRep):

                    self.integrator.x0=initialPositions[rep]
                    xs,vx=self.integrator.run_modifKinEn_Langevin(nrSteps)
                    initialPositions[rep]=xs[-1]

                    for n in range(0,nrSteps):
                        Xrep[rep*nrSteps + n]=xs[n]

                #for n in range(0,nrRep):
                #    Xit[it* (nrRep*nrSteps) + n]=Xrep[n]

                if(it>0):
                    tavEnd=time.time()
                    self.timeAv.addSample(tavEnd-tav0)

                #tmpselftraj=np.copy(self.sampled_trajectory)
                #self.sampled_trajectory=np.concatenate((tmpselftraj,np.copy(Xrep[-1].value_in_unit(self.model.x_unit))))

                xyz=[x.value_in_unit(self.model.x_unit) for x in Xrep[::self.modNr]]
                self.trajSave=md.Trajectory(xyz, self.topology)
                print 'Saving traj to file'
                self.trajSave.save(self.savingFolder+self.algorithmName+'_traj_'+repr(it)+'.h5')


#--------------------- Functions --------------------------------

def reshapeData(model, Xrep):
            #convert to unitless
    xyz = np.array([model.positions.value_in_unit(model.x_unit) for model.positions in Xrep ])
            #print xyz.shape
    #xyz=Xrep

    traj=np.zeros([xyz.shape[0],xyz.shape[1]*xyz.shape[2]])

    for i in range(0,xyz.shape[0]):
        ni=0
        for n in range(0,xyz.shape[1]):
            for d in range(0,3):
                traj[i,ni]=xyz[i, n, d]
                ni=ni+1

    return traj

def reshapeDataBack(traj):

    xyz=np.zeros([traj.shape[0], int(traj.shape[1]/3.0),3])


    for i in range(0,xyz.shape[0]):
        for n in range(0,xyz.shape[1]):
            for d in range(0,xyz.shape[2]):

                xyz[i,n, d]=traj[i, n*xyz.shape[2]+d]

    #[model.positions.value_in_unit(model.x_unit) for model.positions in Xrep ]

    return xyz #* unitConst#model.x_unit

def dimension_reduction(tr, eps, numberOfLandmarks, model, T, method):

        if method=='PCA':

            X_pca = decomposition.TruncatedSVD(n_components=1).fit_transform(tr- np.mean(tr,axis=0))
            v1=X_pca[:,0]

            q=np.zeros(len(tr))
            for i in range(0,len(tr)):
                q[i]=np.exp(-(model.energy(tr[i,:]))/float(T))
            q=q/np.sum(q)

        else:
            kernelDiff=dm.compute_kernel(tr, eps)
            P=dm.compute_P(kernelDiff, tr)
            q=kernelDiff.sum(axis=1)

            #X_se = manifold.spectral_embedding(P, n_components = 1, eigen_solver = 'arpack')
            #V1=X_se[:,0]

            lambdas, V = sps.linalg.eigsh(P, k=3)#, which='LM' )
            ix = lambdas.argsort()[::-1]
            X_se= V[:,ix]

            v1=X_se[:,1]
            v2=X_se[:,2]
            #V2=X_se[:,1]

            #get landmarks
        lm=dm.get_landmarks(tr, numberOfLandmarks, q, v1)

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
    X0 = X[0]-  rmsd.centroid(X[0])

    for i in range(0,len(X)):
    #print "RMSD before translation: ", rmsd.kabsch_rmsd(Y[0],Y[1::])

        X[i] = X[i] -  rmsd.centroid(X[i])
        X[i] = rmsd.kabsch_rotate(X[i], X0)

        ## alternatively use quaternion rotate  instead - does not work in this format
        #rot = rmsd.quaternion_rotate(X[i], X0)
        #X[i] = np.dot(X[i], rot)

    return X

# class ParallelRun():
#
#     def __init__(self, f, modNr, nrStep):
#
#         from mpi4py import MPI
#
#         self.comm = MPI.COMM_WORLD
#         self.rank = self.comm.Get_rank()
#         self.size = self.comm.Get_size()
#
#         self.f=f
#
#         self.traj=np.zeros( self.size, nrStep)
#
#     def run(self, nrSteps):
#         for rep in range(0, nrRep):
#
#             traj_full,p_full=f
#             traj_rep=np.copy(traj_full[::modNr,:])
#
#
#
#             if rep>0:
#                 tmp=np.copy(traj_it)
#                 traj_it=np.concatenate((tmp, traj_rep))
#
#             else:
#                 traj_it=traj_rep
#
#         return traj_it

###############################
