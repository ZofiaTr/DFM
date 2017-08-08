import numpy as np
import sys
from  scipy.spatial.distance import pdist

#import my_test_system

#i = importlib.import_module("matplotlib.text")

class Model():

    def __init__(self):

        #AlanineDipeptideVacuum


        from openmmtools.testsystems import AlanineDipeptideVacuum#TolueneVacuum#SrcImplicit#AlanineDipeptideVacuum#LennardJonesCluster#AlanineDipeptideVacuum#HarmonicOscillator#Diatom #AlanineDipeptideVacuum#Diatom

        #from my_test_system import AlanineDipeptideVacuum

        from simtk import openmm, unit
        from simtk.openmm import app

        name=AlanineDipeptideVacuum
        self.modelname=str(name)

        self.testsystem = name(constraints=None)
        ## LJ cluster
        #self.testsystem = name(nx=2, ny=2, nz=1, constraints=None)

        (self.system, self.positions) = self.testsystem.system, self.testsystem.positions

        dummy_integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)

        self.context = openmm.Context(self.system, dummy_integrator)
        self.context.setPositions(self.positions)

        self.x_unit = self.positions.unit
        self.energy_unit = unit.kilojoule_per_mole
        self.force_unit = unit.kilocalorie_per_mole / self.x_unit
        self.time_unit = unit.femtosecond
        self.velocity_unit = self.x_unit / self.time_unit
        self.mass_unit = unit.amu

        self.masses = 12.0 * self.mass_unit



    def energy(self,x):
        self.context.setPositions(x)
        e = self.context.getState(getEnergy=True).getPotentialEnergy()
        return e # returns simtk-unit'd energy
        #return e.value_in_unit(energy_unit) # strips unit

    def force(self,x):
        self.context.setPositions(x)
        f = self.context.getState(getForces=True).getForces(asNumpy=True)
        return f # returns simtk-unit'd force

    def x_projection(self,x):
        return x[:,0]

    def diff_x_projection(self,x):
        g=np.zeros(x.shape)
        g[:,0]=1.0
        g=g*self.x_unit
        return g

    def radius(self,x):
        return pdist(x, p=2)

    def diff_radius(self,x):
        r=pdist(x, p=2)

        g=x*self.x_unit/r

        return g
