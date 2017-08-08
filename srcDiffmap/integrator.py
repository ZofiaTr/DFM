from openmmtools.constants import kB

import model
import numpy as np

import model
import linear_approximation as aprx
import sampler
from simtk import openmm, unit
import Averages


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
         print self.kineticTemperature.getAverage()



    def run_langevin(self,n_steps):
        """Simulate n_steps of Langevin dynamics using BAOAB"""

        xs = [self.x0]
        vs = [self.x0]

        x = self.x0
        v = self.v0

        a = np.exp(-self.gamma * (self.dt))
        b = np.sqrt(1 - np.exp(-2 * self.gamma * (self.dt)))

        f=self.force_fxn(x)

        for _ in range(n_steps):
            v = v + ((0.5*self.dt ) * f/ self.masses)
            x = x + ((0.5*self.dt ) * v)
            v = (a * v) + b * np.random.randn(*x.shape) * np.sqrt(self.kT / self.masses)
            x = x + ((0.5*self.dt ) * v)
            f=self.force_fxn(x)
            v = v + ((0.5*self.dt ) * f / self.masses)

            xs.append(x)
            vs.append(v)


            kinTemp = self.computeKineticTemperature(v)

            self.kineticTemperature.addSample(kinTemp)
            #print self.kineticTemperature.getAverage()

        self.xEnd=x
        self.vEnd=v

        return xs, vs


    def run_EFTAD(self, n_steps):
        """Simulate n_steps of EFTAD/TAMD with fixed CV's discretized by BAOAB scheme

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

        #theta, diffTheta=aprx.linApproxPsi(x, dataLandmarks, Vlandmarks, deriv_v)

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



    def run_EFTAD_adaptive(self, n_steps, dataLandmarks, Vlandmarks, deriv_v):
        """Simulate n_steps of EFTAD with cv adaptive

        :param n_steps:
        :param dt:
        : optional parameters:

            :return:
        """

        #Number of CV=1

        xs = [self.x0]
        vs = [self.x0]

        x = self.x0
        v = self.v0

        z = self.z0
        vz = self.vz0
        zs = z


        a = np.exp(-self.gamma * (self.dt))
        b = np.sqrt(1 - np.exp(-2 * self.gamma * (self.dt)))

        az = np.exp(-self.gammaAlpha * (self.dt))
        bz = np.sqrt(1.0 - np.exp(-2 * self.gammaAlpha * (self.dt)))


        theta, diffTheta=aprx.linApproxPsi(x, self.model, dataLandmarks, Vlandmarks, deriv_v)
        theta=theta*self.model.x_unit
        #diffTheta=diffTheta.reshape(x.shape)


        F2=-self.kappaAlpha*((theta - z)*diffTheta)
        f=self.force_fxn(x)+F2
        fAlpha=self.kappaAlpha*(theta-z)


        invmassesAlpha= 1.0/self.massAlpha
        sqrtKTmAlpha = np.sqrt(self.kTAlpha *invmassesAlpha)


        for _ in range(n_steps):
            v = v + ((0.5*self.dt ) * f / self.masses)
            vz = vz + ((0.5*self.dt ) * fAlpha / self.massAlpha)

            x = x + ((0.5*self.dt ) * v)
            z = z + ((0.5*self.dt ) * vz)

            v = (a * v) + b * np.random.randn(*x.shape) * np.sqrt(self.kT / self.masses)
            vz = (az * vz) + bz * np.random.randn(*z.shape) * sqrtKTmAlpha

            x = x + ((0.5*self.dt ) * v)
            z = z + ((0.5*self.dt ) * vz)

            theta, diffTheta=aprx.linApproxPsi(x, self.model, dataLandmarks, Vlandmarks, deriv_v)
            theta=theta*self.model.x_unit
            #diffTheta=diffTheta.reshape(x.shape)

            F2=-self.kappaAlpha*((theta - z)*diffTheta)
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
        #self.z0=z

        return xs, vs

################# the following functions are not finished -  work in progress

    def run_modifKinEn_Langevin(self, n_steps, dataLandmarks=None, Vlandmarks=None, deriv_v=None):
            """Simulate n_steps of modified kintic energy Langevin with the kinetic energy perturbed by an adaptively computed CV

            :param n_steps:
            :param dt:
            : optional parameters:

                :return:
            """

            #Number of CV=1

            xs = [self.x0]
            vs = [self.x0]

            x = self.x0
            v = self.v0


            #a = np.exp(-self.gamma * (self.dt))
            #b = np.sqrt(1 - np.exp(-2 * self.gamma * (self.dt)))

            nrSubSteps = 10
            dtInner=self.dt/nrSubSteps


            cke=1.0

            f=self.force_fxn(x)

            dKfct = self.diffKineticEnergyFunction
            #Kfct = self. kineticEnergyFunction

            for _ in range(n_steps):

                v = v + ((0.5*self.dt ) * f * self.invmasses)
                #dK=p/m - dK=v
                p = v * self.masses
                p = p / (self.model.velocity_unit*self.model.mass_unit)
                dK = dKfct(p)
                dK = dK * self.invmasses *(self.model.velocity_unit*self.model.mass_unit)
                x = x + (self.dt  * dK )

                f=self.force_fxn(x)
                v = v + ((0.5*self.dt ) * f * self.invmasses)

                for _ in range(nrSubSteps):

                    p = v * self.masses
                    p = p / (self.model.velocity_unit*self.model.mass_unit)
                    dK = dKfct(p)
                    dK = dK * self.invmasses *(self.model.velocity_unit*self.model.mass_unit)

                    v = v - self.gamma * dK * dtInner  +  np.sqrt(2.0 * self.gamma *self.kT* dtInner * self.invmasses) *np.random.randn(*x.shape)


                #theta, diffTheta=aprx.linApproxPsi(x, dataLandmarks, Vlandmarks, deriv_v)
                #theta=theta*self.model.x_unit
                #diffTheta=diffTheta.reshape(x.shape)*self.model.x_unit

                xs.append(x)
                vs.append(v)

            self.xEnd=x
            self.vEnd=v

            return xs, vs

            #################

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


#################

    # def run_modifKinEn_Langevin(self, n_steps):
    #         """Simulate n_steps of modified kintic energy Langevin with the kinetic energy perturbed by an adaptively computed CV
    #
    #         :param n_steps:
    #         :param dt:
    #         : optional parameters:
    #
    #             :return:
    #         """
    #
    #         #Number of CV=1
    #
    #         xs = [self.x0]
    #         vs = [self.x0]
    #
    #         x = self.x0
    #         v = self.v0
    #
    #         nrSubSteps = 3
    #         dtInner=self.dt/nrSubSteps
    #
    #         cke=1.0
    #
    #         f=self.force_fxn(x)
    #
    #         for _ in range(n_steps):
    #
    #             v = v + ((0.5*self.dt ) * f * self.invmasses)
    #             #dK=p/m - dK=v
    #             p = v * self.masses
    #             #p=p*self.model.velocity_unit*self.model.mass_unit
    #             pValue=p.value_in_unit(self.model.velocity_unit*self.model.mass_unit)
    #             diffTheta = pValue**7 #- self.force_fxn(p) / self.model.force_unit
    #
    #             diffTheta = diffTheta * (self.model.velocity_unit*self.model.mass_unit)
    #
    #             #diffTheta=diffTheta.reshape(v.shape)*self.model.velocity_unit*self.model.mass_unit
    #
    #             dK = (p + cke * diffTheta) * self.invmasses
    #             x = x + (self.dt  * dK )
    #
    #             f=self.force_fxn(x)
    #             v = v + ((0.5*self.dt ) * f * self.invmasses)
    #
    #             for _ in range(nrSubSteps):
    #
    #                 p = v * self.masses
    #
    #                 pValue=p.value_in_unit(self.model.velocity_unit*self.model.mass_unit)
    #                 diffTheta = pValue**7 #- self.force_fxn(p) / self.model.force_unit
    #                 diffTheta=diffTheta*(self.model.velocity_unit*self.model.mass_unit)
    #
    #                 #diffTheta=diffTheta.reshape(v.shape)*self.model.velocity_unit*self.model.mass_unit
    #
    #                 dK = (p + cke * diffTheta) * self.invmasses
    #
    #                 v = v - self.gamma * dK * dtInner  +  np.sqrt(2.0 * self.gamma *self.kT* dtInner * self.invmasses) *np.random.randn(*x.shape)
    #
    #
    #             #theta, diffTheta=aprx.linApproxPsi(x, dataLandmarks, Vlandmarks, deriv_v)
    #             #theta=theta*self.model.x_unit
    #             #diffTheta=diffTheta.reshape(x.shape)*self.model.x_unit
    #
    #             xs.append(x)
    #             vs.append(v)
    #
    #         return xs, vs

            #################

    def diffKineticEnergyFunction(self, p):

            powerKinEn=2.0
            return np.sign(p)*np.abs(p)**(powerKinEn-1.0)/ (self.masses.value_in_unit(self.model.mass_unit) )
