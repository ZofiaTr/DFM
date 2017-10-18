import numpy as np
import sys
from  scipy.spatial.distance import pdist

#import my_test_system

from simtk import openmm, unit
from simtk.openmm import app
from openmmtools import testsystems

#i = importlib.import_module("matplotlib.text")


#modelName='Alanine'
#modelName='Lemon'
modelName='Dimer'

class Model():

    def __init__(self, modelName='Dimer'):


        self.modelName = modelName
        if (self.modelName == 'Dimer'):

            self.testsystem = self.createDimer();
            name = self.testsystem
            self.modelname=str(name)

        elif (self.modelName == 'Lemon'):

            self.testsystem = self.createLemon();
            name = self.testsystem
            self.modelname=str(name)

        elif(self.modelName == 'Alanine'):


            #AlanineDipeptideVacuum

            from openmmtools.testsystems import AlanineDipeptideVacuum#TolueneVacuum#SrcImplicit#AlanineDipeptideVacuum#LennardJonesCluster#AlanineDipeptideVacuum#HarmonicOscillator#Diatom #AlanineDipeptideVacuum#Diatom

            name=AlanineDipeptideVacuum
            self.modelname=str(name)

        #ala2
            self.testsystem = name(constraints=None)

        elif(self.modelName == 'LJCluster'):
            ## LJ cluster
            from openmmtools.testsystems import LennardJonesCluster
            name=LennardJonesCluster

            self.testsystem = name(nx=2, ny=2, nz=1, constraints=None)
        else:
            print("In class Model(): modelName not found")

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
        self.temperature_unit = unit.kelvin

        self.masses = 1.0 * self.mass_unit

        self.fixOneParticle=0



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

    def createDimer(self):


        K=1 * unit.kilocalories_per_mole / unit.angstrom**2#290.1 * unit.kilocalories_per_mole / unit.angstrom**2
        r0=1.550 * unit.angstroms
        w=0.5 * unit.angstroms
        h=0.01 * unit.kilocalories_per_mole / unit.angstrom**2
        m1=39.948 * unit.amu
        m2=39.948 * unit.amu
        constraint=False
        use_central_potential=False

        testsystem =testsystems.TestSystem( K=K,
                 r0=r0,
                 m1=m1,
                 m2=m2,
                 constraint=constraint,
                 use_central_potential=use_central_potential)

        # Create an empty system object.
        system = openmm.System()

        # Add two particles to the system.
        system.addParticle(m1)
        system.addParticle(m2)

        # Add a harmonic bond.
        #force = openmm.HarmonicBondForce()

        ## double well with width w and height h centered at r0: two stable states at r=r0 and  r = r0 + 2w
        print("Dimer model: double-well with 2 states")
        print("State1: r="+repr(r0 ))
        print("State2: r="+repr(r0 + 2.0*w))
        force = openmm.CustomBondForce("h * ( 1.0 - (( r - r0 - w )^2 / w^2) )^2");

        force.addGlobalParameter("r0", r0);
        force.addGlobalParameter("h", h);
        force.addGlobalParameter("w", w);

        force.addBond(0, 1, [])#, r0, h, w)
        system.addForce(force)

        if constraint:
            # Add constraint between particles.
            system.addConstraint(0, 1, r0)

        # Set the positions.
        positions = unit.Quantity(np.zeros([2, 3], np.float32), unit.angstroms)
        positions[1, 0] = r0

        if use_central_potential:
            # Add a central restraining potential.
            Kcentral = 1.0 * unit.kilocalories_per_mole / unit.nanometer**2
            energy_expression = '(K/2.0) * (x^2 + y^2 + z^2);'
            #energy_expression = '(K/2.0) * ((x^2 + y^2 + z^2)-1)^2;'
            energy_expression += 'K = testsystems_Diatom_Kcentral;'
            force = openmm.CustomExternalForce(energy_expression)
            force.addGlobalParameter('testsystems_Diatom_Kcentral', Kcentral)
            force.addParticle(0, [])
            force.addParticle(1, [])
            system.addForce(force)

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('N')
        chain = topology.addChain()
        residue = topology.addResidue('N2', chain)
        topology.addAtom('N', element, residue)
        topology.addAtom('N', element, residue)

        testsystem.topology = topology

        testsystem.system, testsystem.positions = system, positions
        #self.K, self.r0, self.m1, self.m2, self.constraint, self.use_central_potential = K, r0, m1, m2, constraint, use_central_potential

        # # Store number of degrees of freedom.
        # self.ndof = 6 - 1 * constraint

        return testsystem

    def createLemon(self):


        K=1 * unit.kilocalories_per_mole / unit.angstrom**2
        r0=1.550 * unit.angstroms
        k=7.0 * unit.kilocalories_per_mole / unit.angstrom**2
        m1=39.948 * unit.amu
        m2=39.948 * unit.amu
        constraint=False
        use_central_potential=True

        testsystem =testsystems.TestSystem( K=K,
                 r0=r0,
                 m1=m1,
                 m2=m2,
                 constraint=constraint,
                 use_central_potential=use_central_potential)

        # Create an empty system object.
        system = openmm.System()

        # Add two particles to the system.
        system.addParticle(m1)
        system.addParticle(m2)

        # Add a harmonic bond.
        #force = openmm.HarmonicBondForce()

        ## double well with width w and height h centered at r0: two stable states at r=r0 and  r = r0 + 2w
        print("Lemon model: lemon-lice potential with 7 states and 1 reaction coordinate")
        force = openmm.CustomBondForce("0.0*r");

        force.addGlobalParameter("k", k);

        force.addBond(0, 1, [])#, r0, h, w)
        system.addForce(force)

        if constraint:
            # Add constraint between particles.
            system.addConstraint(0, 1, r0)

        # Set the positions.
        positions = unit.Quantity(np.zeros([2, 3], np.float32), unit.angstroms)
        positions[1, 0] = r0

        if use_central_potential:
            # Add a central restraining potential.
            Kcentral = 1.0 * unit.kilocalories_per_mole / unit.nanometer**2
            energy_expression = ' cos(K * atan(x / y)) + 10*( (x^2 + y^2 + z^2) -1)^2 ;'
            #energy_expression = '(K/2.0) * ((x^2 + y^2 + z^2)-1)^2;'
            energy_expression += 'K = testsystems_Diatom_Kcentral;'
            force = openmm.CustomExternalForce(energy_expression)
            force.addGlobalParameter('testsystems_Diatom_Kcentral', Kcentral)
            force.addParticle(0, [])
            force.addParticle(1, [])
            system.addForce(force)

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('N')
        chain = topology.addChain()
        residue = topology.addResidue('N2', chain)
        topology.addAtom('N', element, residue)
        topology.addAtom('N', element, residue)

        testsystem.topology = topology

        testsystem.system, testsystem.positions = system, positions

        return testsystem


dummyModel=Model(modelName)
dummyTopology=dummyModel.testsystem.topology
