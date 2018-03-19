from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import numpy as np

import sampler
import model
import helpers
import dimension_reduction

import mdtraj as md
from simtk import unit
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd



class Stein():
    """
    Stein Variational Importance Sampling Class:

    example :
    # folder where the data is saved
    dataFolderName = 'data/'

    # intialize sampler class together wuth the model
    mdl=model.Model('Alanine')
    intg=integrator.Integrator( model=mdl, gamma=1.0 / unit.picosecond, temperature=300 * unit.kelvin, dt=2.0 * unit.femtosecond,  temperatureAlpha=300 * unit.kelvin)
    smpl=sampler.Sampler(model=mdl, integrator=intg, algorithm=0, dataFileName='Data')

    # stein
    st = stein.Stein(smpl, dataFolderName, modnr = 10)
    # change the stein step size
    st.epsilon_step=unit.Quantity(0.01, smpl.model.x_unit)**2

    #run stein
    st.run_stein(numberOfSteinSteps = 100)

    # results
    X_steined = st.q

    """

    def __init__(self, smpl, dataFolderName, modnr = 1, noisy_gradient = False, kernel_metric = 'euclidean'):

         self.smpl = smpl
         self.topology = self.smpl.model.testsystem.topology
         self.modnr = modnr
         self.dataFolderName=dataFolderName
         self.noisy_gradient = noisy_gradient
         self.setup_stein(modnr = self.modnr)
         self.kernel_metric = kernel_metric



    def load_initial_state(self, modnr = 1):

        self.X_FT = helpers.loadData(self.dataFolderName+'/Traj/*.h5', self.topology, modnr, align=False)
        self.X_FT = dimension_reduction.align_with_mdanalysis(self.X_FT, self.smpl)
        return self.X_FT

    def align_loaded_state(self):

        self.X_FT =self.align_with_mdanalysis(self.X_FT, self.smpl)

    def create_trajectory_list(self, X_short):
        self.XL = []
        [self.XL.append( X_short[n] * self.smpl.model.x_unit) for n in range(X_short.shape[0])]
        return self.XL

    def compute_force(self, XL, leader_set):

        force = []

        for n in range(len(leader_set)):
            force.append( self.smpl.model.force(XL[leader_set[n]]))

        return force

    def compute_force_all(self,XL):

        force = []

        for n in range(len(XL)):
            force.append( self.smpl.model.force(XL[n]))

        return force

    def compute_stein_force(self, XL, leader_set, model):

        force = self.compute_force(XL, leader_set)
        # create numpy array from the list force which has also units
        forcenp = []
        # remove units (we needed the units for the force call before!)
        for n in range(len(leader_set)):
            forcenp.append(force[n].value_in_unit(self.smpl.model.force_unit))

        forcenp = np.asarray(forcenp)
        # create numpy array from the list XL which has also units

        XreshList = []
        # remove units (we needed the units for the force call before!)
        for n in range(len(XL)):
            XreshList.append(XL[n].value_in_unit(model.x_unit))

        Xresh = np.asarray(XreshList)
        # reshape for neighbor search steps x DOF
        Xresh = Xresh.reshape(Xresh.shape[0],Xresh.shape[1]*Xresh.shape[2])
        # choose leader particles using the index set leader_set
        X_leader = np.copy(Xresh[leader_set,:])

        force_resh = forcenp.reshape(forcenp.shape[0],forcenp.shape[1]*forcenp.shape[2])

        # kernel scaling parameter
        self.h=self.kernel_scaling_parameter

        distances = cdist(X_leader, Xresh, metric = self.kernel_metric)
        kernel = np.exp(-distances**2 / self.h)

        # this computes the first part (without the kernel derivatives)
        # f_MDforce is of the format (N_particles, dim)
        f_MDforce = np.dot(kernel.transpose(), force_resh)
        f_MDforce = f_MDforce / len(leader_set)
        # reshape to format (N_particles, n_atoms, 3)
        f_MDforce = f_MDforce.reshape(Xresh.shape[0], forcenp.shape[1], forcenp.shape[2])
        # add force unit and divide by kT
        f_MDforce = f_MDforce * model.force_unit / self.smpl.kT

        #derivative part
        f_der = -2.0/self.h * np.dot(kernel.transpose(), X_leader)
        f_der += 2.0/self.h * np.outer(np.sum(kernel,0),np.ones(Xresh.shape[1])) * Xresh
        f_der = f_der / len(leader_set)
        # reshape to format (N_particles, n_atoms, 3)
        f_der = f_der.reshape(Xresh.shape[0], forcenp.shape[1], forcenp.shape[2])
        # add unit
        f_der = f_der * model.x_unit**(-1)

        return f_MDforce + f_der



    def setup_stein(self, modnr = 1):


            self.X_FT = self.load_initial_state( modnr = modnr)
            self.align_loaded_state()
            self.XL_init = self.create_trajectory_list( self.X_FT)

            self.epsilon_step=unit.Quantity(0.015, self.smpl.model.x_unit)**2
            self.kernel_scaling_parameter = 0.1

            # choose leader set
            self.percentageOfLeaderParticles = 1.0
            self.numberOfLeaderParticles = int(self.percentageOfLeaderParticles*(self.X_FT.shape[0]))
            self.leader_set = np.random.choice(range(self.X_FT.shape[0]), self.numberOfLeaderParticles)#np.array(range(X_short.shape[0]))#

            self.kT = self.smpl.kT
            self.mass = self.smpl.model.masses

            self.q = np.copy(self.X_FT)
            self.qInit = np.copy(self.q)

    def run_stein(self, numberOfSteinSteps = 10):

            self.numberOfSteinSteps = numberOfSteinSteps
            self.f = self.compute_stein_force(self.XL,self.leader_set, self.smpl.model)
            #f = compute_force(XL)

            modit = int(self.numberOfSteinSteps/100)
            if modit ==0:
                modit = 1

            for ns in range(self.numberOfSteinSteps):
                if ns%modit==0:
                    print('Stein iteration '+repr(ns))
                    self.XL = self.rmsd_align(self.XL)

                if self.noisy_gradient :
                    self.leader_set = np.random.choice(range(self.qInit.shape[0]), self.numberOfLeaderParticles)

                self.f = self.compute_stein_force(self.XL,self.leader_set, self.smpl.model)
                for n in range(len(self.XL)):
                    self.XL[n] = (self.XL[n] + self.epsilon_step * self.f[n])
                    self.q[n,:,:] =  np.copy(self.XL[n].value_in_unit(self.smpl.model.x_unit))
                if ns%modit == 0:
                    np.save(self.dataFolderName+'/q_stein.npy', {'q' : self.q, 'it' : ns})
                ## plot progress
                #plotSamplingDihedrals_fromData(q, smpl.model.testsystem.topology, methodName=None, color='b', title = 'Iteration '+repr(ns))
                if np.isnan(self.q).any():
                    print('Explosion. Nan.')
                    break


    def run_langevin_stein(self, numberOfSteinSteps = 10):

            self.numberOfSteinSteps = numberOfSteinSteps
            self.f = self.compute_stein_force(self.XL,self.leader_set, self.smpl.model)
            #f = compute_force(XL)

            self.numberOfLangevinSteps = 2
            a = np.exp(-self.smpl.integrator.gamma * (self.smpl.integrator.dt))
            b = np.sqrt(1 - np.exp(-2 * self.smpl.integrator.gamma * (self.smpl.integrator.dt)))



            modit = int(self.numberOfSteinSteps/100)
            if modit ==0:
                modit =1

            for ns in range(self.numberOfSteinSteps):
                if ns%modit==0:
                    print('Stein iteration '+repr(ns))

                    self.XL = self.rmsd_align(self.XL)

                self.f = self.compute_stein_force(self.XL,self.leader_set, self.smpl.model)
                for n in range(len(self.XL)):
                    self.XL[n] = (self.XL[n] + self.epsilon_step * self.f[n])

                    fLan = self.smpl.model.force(self.XL[n])
                    v = np.random.randn(*self.XL[n].shape) * np.sqrt(self.smpl.kT / self.smpl.model.masses)
                    for i in range(self.numberOfLangevinSteps):
                        self.XL[n], v, fLan = self.Langevin_step(self.XL[n] , v, fLan, a, b,  self.smpl.integrator.dt)

                    self.q[n,:,:] =  np.copy(self.XL[n].value_in_unit(self.smpl.model.x_unit))
                if ns%modit == 0:
                    np.save(self.dataFolderName+'/q_stein.npy', {'q' : self.q, 'it' : ns})
                ## plot progress
                #plotSamplingDihedrals_fromData(q, smpl.model.testsystem.topology, methodName=None, color='b', title = 'Iteration '+repr(ns))
                if np.isnan(self.q).any():
                    print('Explosion. Nan.')
                    break

    def Langevin_step(self, x , v, f,a, b,  dt):

        v = v + ((0.5*dt ) * f/ self.smpl.model.masses)
        x = x + ((0.5*dt ) * v)


        v = (a * v) + b * np.random.randn(*x.shape) * np.sqrt(self.smpl.kT / self.smpl.model.masses)

        x = x + ((0.5*dt ) * v)
        f=self.smpl.model.force(x)

        v = v + ((0.5*dt ) * f / self.smpl.model.masses)

        return x, v , f


    def align_with_mdanalysis(self, X_FT, smpl):

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

    def rmsd_align(self, XL):

        xx=[]
        for i in range(len(XL)):
            xx.append(XL[i].value_in_unit(self.smpl.model.x_unit))

        xx=np.asarray(xx)
        #print(xx.shape)

        xx = self.align_with_mdanalysis(xx, self.smpl)

        XL=[]
        for i in range(xx.shape[0]):
            XL.append(unit.Quantity(xx[i], self.smpl.model.x_unit))
        return XL
