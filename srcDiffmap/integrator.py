from openmmtools.constants import kB

import model
import numpy as np

import model
import sampler
from simtk import openmm, unit
import Averages

import openmmtools

from sklearn.neighbors import NearestNeighbors
import helpers


class Integrator():

    def __init__(self, model, gamma, temperature, temperatureAlpha, dt, massScale=100.0, gammaScale=100.0, kappaScale=100.0):

         self.model=model
         self.force_fxn=self.model.force

         # MD parameters
         self.masses=self.model.masses
         self.invmasses=1.0/self.masses
         self.gamma=gamma
         self.dt=dt
         self.temperature = temperature
         self.kT = kB * self.temperature

         self.x0=self.model.positions



         self.v0=np.random.randn(*self.x0.shape) * np.sqrt(self.kT / self.masses)

         self.xEnd=self.model.positions
         self.vEnd=self.v0

         #TAMD parameters
         self.temperatureAlpha =temperatureAlpha
         self.kappaAlpha=kappaScale*self.model.force_unit/self.model.x_unit
         self.massAlpha = massScale* self.masses
         self.gammaAlpha = gammaScale * self.gamma
         self.kTAlpha =kB * self.temperatureAlpha

         self.z0=0.0*self.model.positions[0,0]
         self.vz0=np.random.randn(*self.z0.shape) * np.sqrt(self.kTAlpha / self.massAlpha)#*self.model.velocity_unit

         self.zEnd=self.z0
         self.vzEnd=self.vz0

         self.ndof= 3.0*self.model.system.getNumParticles() #

         kinTemp = self.computeKineticTemperature(self.v0)
         self.kineticTemperature=Averages.Average(kinTemp)
         self.kineticTemperature.addSample(kinTemp)
         print(self.kineticTemperature.getAverage())

         # Create a BAOAB integrator
         self.langevin_integrator = openmmtools.integrators.LangevinIntegrator(temperature=self.temperature, collision_rate=self.gamma, timestep=self.dt, splitting='V R O R V')

         # Create a Context for integration
         self.context = openmm.Context(self.model.system, self.langevin_integrator)

    def run_langevin(self, n_steps, save_interval=1):
        """Simulate n_steps of Langevin dynamics using Python implementation of BAOAB

        Parameters
        ----------
        n_steps : int
            The number of MD steps to run
        save_interval : int, optional, default=1
            The interval at which steps are saved

        Returns
        -------
        xyz : list of (natoms,3) unitless positions
            xyz[n] is the nth frame from the trajectory in units of self.model.x_units,
            saved with interval save_interval

        TODO
        ----
        * Handle box_vectors so that we can eventually run periodic systems

        """
        if n_steps % save_interval != 0:
            raise Exception("n_steps (%d) should be an integral multiple of save_interval (%d)" % (n_steps, save_interval))

        # Store trajectory
        xyz = list()
        potEnergy = list()

        x = self.x0

        zeroV = 0.0 * x[0,:]
        if (self.model.fixOneParticle):
            x[0,:]= zeroV

        #print(x)

        self.kT = kB * self.temperature

        v = self.v0

        a = np.exp(-self.gamma * (self.dt))
        b = np.sqrt(1 - np.exp(-2 * self.gamma * (self.dt)))

        f=self.force_fxn(x)

        for step in range(n_steps):
            v = v + ((0.5*self.dt ) * f/ self.masses)
            x = x + ((0.5*self.dt ) * v)

            if (self.model.fixOneParticle):
                x[0,:]= zeroV


            v = (a * v) + b * np.random.randn(*x.shape) * np.sqrt(self.kT / self.masses)

            x = x + ((0.5*self.dt ) * v)

            if (self.model.fixOneParticle):
                x[0,:]= zeroV

            f=self.force_fxn(x)

            v = v + ((0.5*self.dt ) * f / self.masses)

            if (self.model.fixOneParticle):
                x[0,:]= zeroV


            if (step+1) % save_interval == 0:
                xyz.append(x / self.model.x_unit)
                kinTemp = self.computeKineticTemperature(v)
                self.kineticTemperature.addSample(kinTemp)
                potEnergy.append(self.model.energy(x)/self.model.energy_unit)

            #print self.kineticTemperature.getAverage()

        self.xEnd=x
        self.vEnd=v

        return xyz, potEnergy

    def run_openmm_langevin(self, n_steps, save_interval=1):
        """Simulate n_steps of Langevin dynamics using openmmtools BAOAB

        Parameters
        ----------
        n_steps : int
            The number of MD steps to run
        save_interval : int, optional, default=1
            The interval at which steps are saved

        Returns
        -------
        xyz : list of (natoms,3) unitless positions
            xyz[n] is the nth frame from the trajectory in units of self.model.x_units,
            saved with interval save_interval

        TODO
        ----
        * Handle box_vectors so that we can eventually run periodic systems

        """
        if n_steps % save_interval != 0:
            raise Exception("n_steps (%d) should be an integral multiple of save_interval (%d)" % (n_steps, save_interval))

        #reset temperature if necessary
        self.langevin_integrator.setTemperature(self.temperature)

        # Intialize positions and velocities

        #print(self.x0)
        #if self.initialPosition is not None:
    #        self.x0=self.initialPosition

        self.context.setPositions(self.x0)
        self.context.setVelocities(self.v0)

        # print(self.x0)
        # while 1 :
        #     pass

        # Store trajectory
        xyz = list()
        potEnergy = list()

        #print('Temperature in openmmtools integrator set as '+repr(self.langevin_integrator.getTemperature()))


        # Run n_steps of dynamics
        for iteration in range(int(n_steps / save_interval)):
            self.langevin_integrator.step(save_interval)
            state = self.context.getState(getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True)
            x = state.getPositions(asNumpy=True)

            v = state.getVelocities(asNumpy=True)
            e = state.getPotentialEnergy()
            epot = e.value_in_unit(self.model.energy_unit)
            potEnergy.append(epot)
            # Append to trajectory
            xyz.append(x / self.model.x_unit)
            # Store kinetic temperature
            kinTemp = state.getKineticEnergy()*2.0/self.ndof/ (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA)
            self.kineticTemperature.addSample(kinTemp)


        # Save final state
        self.xEnd = x
        self.vEnd = v



        return xyz, potEnergy

    def run_EFTAD(self, n_steps):
        """Simulate n_steps of EFTAD/TAMD with fixed CV's discretized by BAOAB scheme
        rewrite project function to define the collective variable
        :param n_steps:
        :param dt:
        : optional parameters:

            :return:
        """

        #Number of CV=1

        xs = [self.x0]
        vs = [self.x0]
        zs = [self.z0]
         ## intialize by the position of one particle..

        x = self.x0
        v = self.v0

        z = self.z0
        vz = self.vz0

        a = np.exp(-self.gamma * (self.dt))
        b = np.sqrt(1 - np.exp(-2 * self.gamma * (self.dt)))

        az = np.exp(-self.gammaAlpha * (self.dt))
        bz = np.sqrt(1 - np.exp(-2 * self.gammaAlpha * (self.dt)))

        theta =self.model.x_projection(x)#self.model.positions[:,0]
        diffTheta=self.model.diff_x_projection(x)

        F2=-self.kappaAlpha*((theta - z)*diffTheta.T).T*self.model.x_unit
        f=self.force_fxn(x)+F2
        fAlpha=self.kappaAlpha*(theta-z)

        for _ in range(n_steps):
            v = v + ((0.5*self.dt ) * f / self.masses)
            vz = vz + ((0.5*self.dt ) * fAlpha / self.massAlpha)

            x = x + ((0.5*self.dt ) * v)
            z = z + ((0.5*self.dt ) * vz)

            v = (a * v) + b * np.random.randn(*x.shape) * np.sqrt(self.kT / self.masses)
            vz = (az * vz) + bz * np.random.randn(*z.shape) * np.sqrt(self.kTAlpha / self.massAlpha)

            x = x + ((0.5*self.dt ) * v)
            z = z + ((0.5*self.dt ) * vz)

            theta =self.model.x_projection(x)#self.model.positions[:,0]
            diffTheta=self.model.diff_x_projection(x)

            F2=-self.kappaAlpha*((theta - z)*diffTheta.T).T*self.model.x_unit
            f=self.force_fxn(x)+F2
            fAlpha=self.kappaAlpha*(theta-z)


            v = v + ((0.5*self.dt ) * f / self.masses)
            vz = vz + ((0.5*self.dt ) * fAlpha / self.massAlpha)

            xs.append(x)
            vs.append(v)

        self.xEnd=x
        self.vEnd=v
        self.zEnd=z
        self.vzEnd=vz

        return xs, vs

        #################

    def run_langevin_ABF(self, n_steps, X, V, save_interval=1):
        """Simulate n_steps of Langevin dynamics with adaptive force biasing using Python implementation of BAOAB

        Parameters
        ----------
        n_steps : int
            The number of MD steps to run
        X : trajectory used to compute diffusion map
        V : diffusion coordinate
        save_interval : int, optional, default=1
            The interval at which steps are saved

        Returns
        -------
        xyz : list of (natoms,3) unitless positions
            xyz[n] is the nth frame from the trajectory in units of self.model.x_units,
            saved with interval save_interval

        TODO
        ----
        * Handle box_vectors so that we can eventually run periodic systems

        """
        if n_steps % save_interval != 0:
            raise Exception("n_steps (%d) should be an integral multiple of save_interval (%d)" % (n_steps, save_interval))

        # Store trajectory
        xyz = list()
        potEnergy = list()

        x = self.x0
        x = self.periodicBoundaryCondition(x);

        zeroV = 0.0 * x[0,:]
        if (self.model.fixOneParticle):
            x[0,:]= zeroV

        #print(x)

        self.kT = kB * self.temperature

        v = self.v0

        a = np.exp(-self.gamma * (self.dt))
        b = np.sqrt(1 - np.exp(-2 * self.gamma * (self.dt)))

        f = self.force_fxn(x)

        forceunit = self.model.force_unit


        f -= unit.Quantity(compute_gradient_of_diffusionmap_coordinate(x.value_in_unit(self.model.x_unit), X, V),forceunit)

        for step in range(n_steps):
            v = v + ((0.5*self.dt ) * f/ self.masses)
            x = x + ((0.5*self.dt ) * v)

            x = self.periodicBoundaryCondition(x);

            if (self.model.fixOneParticle):
                x[0,:]= zeroV


            v = (a * v) + b * np.random.randn(*x.shape) * np.sqrt(self.kT / self.masses)

            x = x + ((0.5*self.dt ) * v)
            x = self.periodicBoundaryCondition(x);

            if (self.model.fixOneParticle):
                x[0,:]= zeroV

            f=self.force_fxn(x)
            f -= unit.Quantity(compute_gradient_of_diffusionmap_coordinate(x.value_in_unit(self.model.x_unit), X, V),forceunit)

            v = v + ((0.5*self.dt ) * f / self.masses)

            if (self.model.fixOneParticle):
                x[0,:]= zeroV


            if (step+1) % save_interval == 0:
                xyz.append(x / self.model.x_unit)
                kinTemp = self.computeKineticTemperature(v)
                self.kineticTemperature.addSample(kinTemp)
                potEnergy.append(self.model.energy(x)/self.model.energy_unit)

            #print self.kineticTemperature.getAverage()

        self.xEnd=x
        self.vEnd=v

        return xyz, potEnergy


    def computeKineticEnergy(self, v):

        p =  self.masses* v
        kinen = 0.0*self.model.energy_unit


        for i in range(self.model.system.getNumParticles()):

            t =  0.0*self.model.energy_unit
            for j in range(3):
                t = t  + p[i,j] * p[i,j]/self.masses

            kinen = kinen + t

        kinen = 0.5 * kinen
        return kinen

    def computeKineticTemperature(self, v):

        kinen= self.computeKineticEnergy(v)

        kinTemp =kinen*2.0/self.ndof/ (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA)#*self.model.energy_unit
        return kinTemp

    def computeKineticTemperatureModif(self, v, dK):

        p = v * self.masses
        kinen = 0.0*self.model.energy_unit

        for i in range(self.model.system.getNumParticles()):

            t =  0.0*self.model.energy_unit
            dKval = dK(v[i,:] / self.model.velocity_unit)

            for j in range(3):
                t = t  + v[i,j] * dKval[j]* self.model.velocity_unit * self.model.mass_unit

            kinen = kinen + t

        kinTemp =2.0*kinen/self.ndof/ (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA)
        return kinTemp

    def diffKineticEnergyFunction(self, p):

            #unitless
            powerKinEn=2.0
            DkinEn = np.sign(p)*np.abs(p)**(powerKinEn-1.0)/ (self.masses.value_in_unit(self.model.mass_unit) )

            return DkinEn

    def periodicBoundaryCondition(self, x):


        xt = x.value_in_unit(self.model.x_unit)
        nrPart = xt.shape[0]
        dim = xt.shape[1]

        for n in range(nrPart):
            for d in range(dim):
                if xt[n,d] < self.model.boxsize:
                    xt[n,d] = xt[n,d] + self.model.boxsize
                if xt[n,d] > self.model.boxsize:
                    xt[n,d] = xt[n,d] - self.model.boxsize

        x = unit.Quantity(xt, self.model.x_unit)

        return x

##############################################################

def compute_gradient_of_diffusionmap_coordinate(x, X, V):
    """
    Compute an approximation of the gradient of diffusionmap at point x using the finite differences of the nearest neighbors

    Parameters:
    x : point where the grad should be evalueated at, unitless,
        ndarray size (number of particles, dimension)
    X : set of points from which V was computed, unitless
        ndarray size (number of points, number of particles * dimension)
    V : diffusionmap coordinate, unitless
        ndarray size (number of points)
    """

    fe, bin_vals = helpers.compute_free_energy(V,   weights=None, nrbins=5, kBT=1)

    origshape0 = x.shape[0]
    origshape1 = x.shape[1]

    x = x.reshape(x.shape[0]*x.shape[1])
    xset = np.zeros((X.shape[0]+1, X.shape[1]))
    xset[0,:] =x
    xset[1:,:] = X

    fe_X = np.zeros(X.shape[0])
    idx = np.where(np.abs(np.exp(-bin_vals)-V)<0.001)
    print(idx)
    while(1):
        pass
    #fe_X =

    neigh = NearestNeighbors(n_neighbors=3,metric='euclidean')
    neigh.fit(xset)

    dist, indices = neigh.kneighbors(xset)
    indices = indices[0]-1

    #print(indices)



    Xresh = X.reshape(X.shape[0], origshape0, origshape1)
    f = (fe[indices[1]] - fe[indices[0]]) * np.ones((origshape0,origshape1))/(Xresh[indices[1],:,:] - Xresh[indices[0],:,:])


    #f = f.reshape(origshape0, origshape1)
    return f
