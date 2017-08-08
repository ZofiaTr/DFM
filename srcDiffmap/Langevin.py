"""Langevin Sampling"""

# Author: Zofia
# License:



import numpy as np
from numpy import linalg as LA

import model

import integrationSteps as step
import Averages as average

import linear_approximation as aprx


#np.random.seed(0)

#beta = 1.0  # inverse temperature





def write_trajectory_xyz(traj, nrAtoms, typ, name):

    file = open('{}.xyz'.format(name),"w")
    file.write('{}\n'.format(nrAtoms))

    nrSteps=traj.shape[0]
    dim=traj.shape[1]

    for i in range(1, nrSteps):
        if (dim==1):
            z=0
            y=0
            x=traj[i,0]
        if (dim ==2):
            z=0
            x=traj[i,0]
            y=traj[i,1]
        if (dim==3):
            x=traj[i,0]
            y=traj[i,1]
            z=traj[i,2]

        str='{} \t {} \t {} \t {} \n'.format(typ, x, y, z)
        file.write(str)
    file.close()




def Fspring(z,theta, diffTheta, kappa):

    #theta=np.array(theta)
    #z=np.array(z)
    #diffTheta=np.array(diffTheta)

    return kappa*(theta - z)*diffTheta


class Langevin:
    """Langevin sampling
    -----------
    params=[T, Talpha, kappaAlpha, hDimer, wDimer, mass, massAlpha, gamma, gammaAlpha, dt, potentialFlag, alpha];
    Attributes
    ----------
    trajectory_ : array, shape = (n_samples, n_components)

    References
    ----------
    - BAOAB, EFTAD/TAMD
    """

    def __init__(self, model, T=None, Talpha=None, kappaAlpha=None, mass=None, massAlpha=None, gamma=None, gammaAlpha=None, dt=None, x0=0, p0=0,dimCV=None):
        self.T=T
        self.TAlpha=Talpha
        self.kappaAlpha=kappaAlpha
        #self.hDimer=hDimer
        #self.wDimer=wDimer
        self.mass=mass
        self.massAlpha=massAlpha
        self.gamma=gamma
        self.gammaAlpha=gammaAlpha
        self.dt=dt
        self.dtEftad=dt
        #self.potentialFlag=potentialFlag
        #self.alpha=alpha
        self.x0=x0
        self.p0=p0
        self.kineticTemperature=average.Average()
        self.averagePosition=average.Average()
        self.averagePositionEFTAD=average.Average()
        self.averageEFTAD_z=average.Average()
        self.xEnd=x0
        self.pEnd=p0
        self.dim=len(x0)
        self.dimCV=1
        self.z0=np.zeros(self.dimCV)

        self.model=model




        # implement BAOAB / VRORV
    def simulate(self, n_steps, **kwargs):
        """Simulate n_steps of BAOAB Langevin

        :param x0:
        :param p0:
        :param n_steps:
        :param gamma:
        :param dt:
        : optional parameters:
            'Dimension': space dimension
            'Potential':
                        HO harmonic oscillator
                        DW double well
        :return:
        """


        if ('ComputeEnergy' in kwargs):
             energyFlag= kwargs['ComputeEnergy']
        else:
             energyFlag = 0

        #if ('Potential' in kwargs):
        #     potentialFlag= kwargs['Potential']
        #else:
        #     potentialFlag = 'HO'

        if (energyFlag==1):
            E_total = np.zeros(n_steps)
            E_potential = np.zeros(n_steps)
        x, p = np.array(self.x0), np.array(self.p0)
        xs, ps = np.zeros((n_steps, self.dim)), np.zeros((n_steps, self.dim))

        try:
            xs[0,:] = self.x0
            ps[0,:] = self.p0
        except:
            print 'Dimension of input arguments does not match'
            print self.x0, self.p0




        a2, b2=step.O_const(self.gamma, self.dt, self.mass, self.T)

        R = np.zeros((n_steps, self.dim))
        R[:,0]=np.random.randn(n_steps)

        if self.dim==2:
            R[:,1] = np.random.randn(n_steps)

        if self.dim==3:
            R[:,1] = np.random.randn(n_steps)
            R[:,2] = np.random.randn(n_steps)



        f=-self.model.force(x)



        for i in range(1, n_steps):

            p=step.B(p, f, 0.5*self.dt)
            x=step.A(x, p, 0.5*self.dt, self.mass)
            p=step.O(p,  R[i,:], self.dt, a2, b2)
            x=step.A(x, p, 0.5*self.dt, self.mass)

            f=-self.model.force(x)

            p=step.B(p, f, 0.5*self.dt)

            # store
            xs[i,:] = x
            ps[i,:] = p

            self.kineticTemperature.addSample(LA.norm(p)**2/float(self.mass)/float(self.dim))
            self.averagePosition.addSample(x[0])

            if (energyFlag==1):
                E_potential[i] = self.model.potential(x)
                E_total[i] = E_potential[i] + 0.5 * LA.norm(p)**2/float(self.mass)

        self.xEnd=x
        self.pEnd=p

        if (energyFlag==1):
            return xs, ps, E_total, E_potential
        else:
            return xs, ps


        """
        EFTAD trajectory
        """

    def simulate_eftad(self, n_steps, dataLandmarks, Vlandmarks, deriv_v, **kwargs):
        """Simulate n_steps of EFTAD

        :param n_steps:
        :param dt:
        : optional parameters:
            'ComputeForcing': return eftad force
            :return:
        """


        if ('ComputeForcing' in kwargs):
             energyFlag= kwargs['ComputeForcing']
        else:
             energyFlag = 0

        if ('StepSize' in kwargs):
             self.dtEftad= kwargs['StepSize']



        x, p = np.array(self.x0), np.array(self.p0)
        xs, ps = np.zeros((n_steps, self.dim)), np.zeros((n_steps, self.dim))
        zs=np.zeros((n_steps, self.dimCV))

        self.z0=self.x0

        if(energyFlag==1):
            Finst = np.zeros((n_steps, self.dimCV))


        try:

            x = self.x0#dataLandmarks[0,:]
            p = self.p0#pEnd
            z = 0

            xs[0,:] = x
            ps[0,:] = p
            zs[0,:] = z


            pAlpha = 0 #np.zeros(z.shape)


        except:
            print 'In simulate_eftad: Dimension of input arguments does not match'
            #print self.x0, self.p0


        #self.kineticTemperature.clear()


        a2, b2=step.O_const(self.gamma, self.dtEftad, self.mass, self.T)
        a2Alpha, b2Alpha=step.O_const(self.gammaAlpha, self.dtEftad, self.massAlpha, self.TAlpha)

        R = np.zeros((n_steps, self.dim))
        R[:,0]=np.random.randn(n_steps)
        if self.dim==2:
            R[:,1] = np.random.randn(n_steps)

        if self.dim==3:
            R[:,1] = np.random.randn(n_steps)
            R[:,2] = np.random.randn(n_steps)

        Ralpha = np.zeros((n_steps, self.dimCV))
        Ralpha[:,0]=np.random.randn(n_steps)
        if self.dimCV==2:
            Ralpha[:,1] = np.random.randn(n_steps)

        if self.dimCV==3:
            Ralpha[:,1] = np.random.randn(n_steps)
            Ralpha[:,2] = np.random.randn(n_steps)


        theta, diffTheta=aprx.linApproxPsi(x, dataLandmarks, Vlandmarks, deriv_v)
        #theta =x[0]
        #diffTheta=np.array([1, 0])

        f=-self.model.force(x) - Fspring(z,theta, diffTheta, self.kappaAlpha)
        fAlpha=self.kappaAlpha*(theta-z)

        #print fAlpha

        for i in range(1, n_steps):

            p=step.B(p, f, 0.5*self.dtEftad)
            pAlpha=step.B(pAlpha, fAlpha, 0.5*self.dtEftad)

            x=step.A(x, p, 0.5*self.dtEftad, self.mass)
            z=step.A(z, pAlpha, 0.5* self.dtEftad, self.massAlpha)


            p=step.O(p,  R[i,:], self.dtEftad, a2, b2)
            pAlpha=step.O(pAlpha,Ralpha[i,:],self.dtEftad, a2Alpha, b2Alpha)

            x=step.A(x, p,      0.5* self.dtEftad,      self.mass)
            z=step.A(z, pAlpha, 0.5* self.dtEftad, self.massAlpha)

            theta, diffTheta=aprx.linApproxPsi(x, dataLandmarks, Vlandmarks, deriv_v)
            #theta =x[0]

            f= -self.model.force(x) - Fspring(z,theta, diffTheta, self.kappaAlpha)
            fAlpha=self.kappaAlpha*(theta-z)

            #print fAlpha

            p=     step.B(p,      f,      0.5*self.dtEftad)
            pAlpha=step.B(pAlpha, fAlpha, 0.5*self.dtEftad)

            # store
            if(energyFlag==1):
                Finst[i,:] = fAlpha


            xs[i,:] = x
            ps[i,:] = p
            zs[i,:] = z

            self.averagePositionEFTAD.addSample(x[0])

            #if (energyFlag==1):
            #    E_potential[i] = potential(x, self.potentialFlag, self.hDimer, self.wDimer)
            #    E_total[i] = E_potential[i] + 0.5 * LA.norm(p)**2/float(self.mass)

        self.xEnd=x
        self.pEnd=p

        if (energyFlag==1):
            return xs, ps, zs, Finst
        #, E_total, E_potential
        else:
            return xs, ps, zs


    """
        EFTAD with given CV
        """

    def simulate_eftad_fix(self, n_steps, thetaFct, diffThetaFct, **kwargs):
        """Simulate n_steps of EFTAD

        :param n_steps:
        :param dt:
        : optional parameters:
            'ComputeForcing': return eftad force
            :return:
        """


        if ('ComputeForcing' in kwargs):
             energyFlag= kwargs['ComputeForcing']
        else:
             energyFlag = 0

        if ('StepSize' in kwargs):
             self.dtEftad= kwargs['StepSize']

        kineticTemperature_eftad=average.Average()
        averagePosition_eftad=average.Average()

        x, p = np.array(self.x0), np.array(self.p0)
        xs, ps = np.zeros((n_steps, self.dim)), np.zeros((n_steps, self.dim))
        zs=np.zeros((n_steps, self.dimCV))

        self.z0=self.x0

        if(energyFlag==1):
            Finst = np.zeros((n_steps, self.dimCV))


        try:

            x = self.x0#dataLandmarks[0,:]
            p = self.p0#pEnd
            z = 0

            xs[0,:] = x
            ps[0,:] = p
            zs[0,:] = z


            pAlpha = 0 #np.zeros(z.shape)


        except:
            print 'In simulate_eftad: Dimension of input arguments does not match'
            #print self.x0, self.p0


        #self.kineticTemperature.clear()


        a2, b2=step.O_const(self.gamma, self.dtEftad, self.mass, self.T)
        a2Alpha, b2Alpha=step.O_const(self.gammaAlpha, self.dtEftad, self.massAlpha, self.TAlpha)

        R = np.zeros((n_steps, self.dim))
        R[:,0]=np.random.randn(n_steps)
        if self.dim==2:
            R[:,1] = np.random.randn(n_steps)

        if self.dim==3:
            R[:,1] = np.random.randn(n_steps)
            R[:,2] = np.random.randn(n_steps)

        Ralpha = np.zeros((n_steps, self.dimCV))
        Ralpha[:,0]=np.random.randn(n_steps)
        if self.dimCV==2:
            Ralpha[:,1] = np.random.randn(n_steps)

        if self.dimCV==3:
            Ralpha[:,1] = np.random.randn(n_steps)
            Ralpha[:,2] = np.random.randn(n_steps)



        theta =thetaFct(x)
        diffTheta=diffThetaFct(x)

        F2= - Fspring(z,theta, diffTheta, self.kappaAlpha)

        f=-self.model.force(x)+ F2
        #Fspring(z,theta, diffTheta, self.kappaAlpha)
        fAlpha=self.kappaAlpha*(theta-z)

        #print fAlpha

        for i in range(1, n_steps):

            #p=step.B(p, f, 0.5*self.dtEftad)
            #pAlpha=step.B(pAlpha, fAlpha, 0.5*self.dtEftad)
            #
            #x=step.A(x, p, 0.5*self.dtEftad, self.mass)
            #z=step.A(z, pAlpha, 0.5* self.dtEftad, self.massAlpha)
            #
            #
            #p=step.O(p,  R[i,:], self.dtEftad, a2, b2)
            #pAlpha=step.O(pAlpha,Ralpha[i,:],self.dtEftad, a2Alpha, b2Alpha)
            #
            #x=step.A(x, p,      0.5* self.dtEftad,      self.mass)
            #z=step.A(z, pAlpha, 0.5* self.dtEftad, self.massAlpha)
            #
            #theta =thetaFct(x)
            #diffTheta=diffThetaFct(x)
            #
            #
            #f= -force(x, self.potentialFlag, self.hDimer, self.wDimer) - Fspring(z,theta, diffTheta, self.kappaAlpha)
            #fAlpha=self.kappaAlpha*(theta-z)
            #
            #p=     step.B(p,      f,      0.5*self.dtEftad)
            #pAlpha=step.B(pAlpha, fAlpha, 0.5*self.dtEftad)

            ###
            p=step.B(p, f, 0.5*self.dtEftad)
            x=step.A(x, p, 0.5*self.dtEftad, self.mass)
            p=step.O(p,  R[i,:], self.dtEftad, a2, b2)
            x=step.A(x, p,      0.5* self.dtEftad,      self.mass)

            theta =thetaFct(x)
            diffTheta=diffThetaFct(x)

            F2= - Fspring(z,theta, diffTheta, self.kappaAlpha)

            f= -self.model.force(x)+ F2
            p=     step.B(p,      f,       0.5*self.dtEftad)

            pAlpha=step.B(pAlpha, fAlpha, 0.5*self.dtEftad)
            z=step.A(z, pAlpha, 0.5* self.dtEftad, self.massAlpha)
            pAlpha=step.O(pAlpha,Ralpha[i,:],self.dtEftad, a2Alpha, b2Alpha)
            z=step.A(z, pAlpha, 0.5* self.dtEftad, self.massAlpha)



            fAlpha=self.kappaAlpha*(theta-z)
            pAlpha=step.B(pAlpha, fAlpha, 0.5*self.dtEftad)

            ###


            # store
            if(energyFlag==1):
                Finst[i,:] = fAlpha


            xs[i,:] = x
            ps[i,:] = p
            zs[i,:] = z

            kineticTemperature_eftad.addSample(LA.norm(p)**2/float(self.mass)/float(self.dim))
            self.averageEFTAD_z.addSample(z)
            self.averagePositionEFTAD.addSample(x[0])
            #print z

            #self.kineticTemperature.addSample(LA.norm(p)**2/float(self.mass)/float(self.dim))

            #if (energyFlag==1):
            #    E_potential[i] = potential(x, self.potentialFlag, self.hDimer, self.wDimer)
            #    E_total[i] = E_potential[i] + 0.5 * LA.norm(p)**2/float(self.mass)

        self.xEnd=x
        self.pEnd=p

        if (energyFlag==1):
            return xs, ps, zs, Finst,  kineticTemperature_eftad.getAverage(), averagePosition_eftad.getAverage()
        #, E_total, E_potential
        else:
            return xs, ps, zs#,  kineticTemperature_eftad.getAverage(), averagePosition_eftad.getAverage()





    def simulate_eftad_kinEn(self, n_steps, dataLandmarks, Vlandmarks, deriv_v, **kwargs):
        """Simulate n_steps of EFTAD with modified kinetic energy

        :param n_steps:
        :param dt:
        : optional parameters:
            'ComputeForcing': return eftad force
            :return:
        """


        if ('ComputeForcing' in kwargs):
             energyFlag= kwargs['ComputeForcing']
        else:
             energyFlag = 0

        if ('StepSize' in kwargs):
             self.dtEftad= kwargs['StepSize']



        x, p = np.array(self.x0), np.array(self.p0)
        xs, ps = np.zeros((n_steps, self.dim)), np.zeros((n_steps, self.dim))
        zs=np.zeros((n_steps, self.dimCV))

        self.z0=self.x0

        if(energyFlag==1):
            Finst = np.zeros((n_steps, self.dimCV))


        try:

            x = self.x0#dataLandmarks[0,:]
            p = self.p0#pEnd
            z = 0

            xs[0,:] = x
            ps[0,:] = p
            zs[0,:] = z


            pAlpha = 0 #np.zeros(z.shape)


        except:
            print 'In simulate_eftad kin en: Dimension of input arguments does not match'
            #print self.x0, self.p0


        #self.kineticTemperature.clear()


        a2, b2=step.O_const(self.gamma, self.dtEftad, self.mass, self.T)
        a2Alpha, b2Alpha=step.O_const(self.gammaAlpha, self.dtEftad, self.massAlpha, self.TAlpha)

        R = np.zeros((n_steps, self.dim))
        R[:,0]=np.random.randn(n_steps)
        if self.dim==2:
            R[:,1] = np.random.randn(n_steps)

        if self.dim==3:
            R[:,1] = np.random.randn(n_steps)
            R[:,2] = np.random.randn(n_steps)

        Ralpha = np.zeros((n_steps, self.dimCV))
        Ralpha[:,0]=np.random.randn(n_steps)
        if self.dimCV==2:
            Ralpha[:,1] = np.random.randn(n_steps)

        if self.dimCV==3:
            Ralpha[:,1] = np.random.randn(n_steps)
            Ralpha[:,2] = np.random.randn(n_steps)


        theta, diffTheta=aprx.linApproxPsi(x, dataLandmarks, Vlandmarks, deriv_v)
        #theta =x[0]
        #diffTheta=np.array([1, 0])

        f=-self.model.force(x) - Fspring(z,theta, diffTheta, self.kappaAlpha)
        fAlpha=self.kappaAlpha*(theta-z)

        #print fAlpha
        ekc=10.0

        for i in range(1, n_steps):

            p=step.B(p, f, self.dtEftad)

            thetaK, diffThetaK=aprx.linApproxPsi(p, dataLandmarks, Vlandmarks, deriv_v)
            fK=p/float(self.mass)+ ekc*diffTheta

            x=step.AK(x, fK, self.dtEftad)

            p=p-fK*self.dt+np.sqrt(2*self.dtEftad)*  R[i,:]

            theta, diffTheta=aprx.linApproxPsi(x, dataLandmarks, Vlandmarks, deriv_v)
            f= -self.model.force(x) - Fspring(z,theta, diffTheta, self.kappaAlpha)



            pAlpha=step.B(pAlpha, fAlpha, self.dtEftad)
            z=step.A(z, pAlpha, self.dtEftad, self.massAlpha)
            pAlpha=step.O(pAlpha,Ralpha[i,:],self.dtEftad, a2Alpha, b2Alpha)

            fAlpha=self.kappaAlpha*(theta-z)


            # store
            if(energyFlag==1):
                Finst[i,:] = fAlpha


            xs[i,:] = x
            ps[i,:] = p
            zs[i,:] = z




            #if (energyFlag==1):
            #    E_potential[i] = potential(x, self.potentialFlag, self.hDimer, self.wDimer)
            #    E_total[i] = E_potential[i] + 0.5 * LA.norm(p)**2/float(self.mass)

        self.xEnd=x
        self.pEnd=p

        if (energyFlag==1):
            return xs, ps, zs, Finst
        #, E_total, E_potential
        else:
            return xs, ps, zs


    #
    # def simulate_kinEn(self, n_steps, dataLandmarks, Vlandmarks, deriv_v, **kwargs):
    #     """Simulate n_steps with modified kinetic energy
    #
    #     :param n_steps:
    #     :param dt:
    #     : optional parameters:
    #         'ComputeForcing': return eftad force
    #         :return:
    #     """
    #
    #
    #     if ('ComputeForcing' in kwargs):
    #          energyFlag= kwargs['ComputeForcing']
    #     else:
    #          energyFlag = 0
    #
    #     if ('StepSize' in kwargs):
    #          self.dtEftad= kwargs['StepSize']
    #
    #
    #
    #     x, p = np.array(self.x0), np.array(self.p0)
    #     xs, ps = np.zeros((n_steps, self.dim)), np.zeros((n_steps, self.dim))
    #     zs=np.zeros((n_steps, self.dimCV))
    #
    #     self.z0=self.x0
    #
    #     if(energyFlag==1):
    #         Finst = np.zeros((n_steps, self.dimCV))
    #
    #
    #
    #
    #     x = self.x0#dataLandmarks[0,:]
    #     p = self.p0#pEnd
    #     z = 0
    #
    #     xs[0,:] = x
    #     ps[0,:] = p
    #     zs[0,:] = z
    #
    #
    #
    #
    #     R = np.zeros((n_steps, self.dim))
    #     R[:,0]=np.random.randn(n_steps)
    #     if self.dim==2:
    #         R[:,1] = np.random.randn(n_steps)
    #
    #     if self.dim==3:
    #         R[:,1] = np.random.randn(n_steps)
    #         R[:,2] = np.random.randn(n_steps)
    #
    #
    #     f=-force(x, self.potentialFlag, self.hDimer, self.wDimer)
    #
    #
    #     ekc=10.0
    #
    #     for i in range(1, n_steps):
    #
    #         p=step.B(p, f, self.dt)
    #
    #         thetaK, diffThetaK=aprx.linApproxPsi(p, dataLandmarks, Vlandmarks, deriv_v)
    #         fK=p/float(self.mass)+ ekc*diffTheta
    #
    #         x=step.AK(x, fK, self.dt)
    #
    #         p=p-fK*self.dt+np.sqrt(2*self.dt)*  R[i,:]
    #
    #         f= -force(x, self.potentialFlag, self.hDimer, self.wDimer)
    #
    #         # store
    #         if(energyFlag==1):
    #             Finst[i,:] = fAlpha
    #
    #
    #         xs[i,:] = x
    #         ps[i,:] = p
    #         zs[i,:] = z
    #         self.kineticTemperature.addSample((LA.norm(p)**2/float(self.mass)+np.dot(diffThetaK,diffThetaK))/float(self.dim))
    #
    #         self.averagePosition.addSample(x[0])
    #
    #
    #         #if (energyFlag==1):
    #         #    E_potential[i] = potential(x, self.potentialFlag, self.hDimer, self.wDimer)
    #         #    E_total[i] = E_potential[i] + 0.5 * LA.norm(p)**2/float(self.mass)
    #
    #     self.xEnd=x
    #     self.pEnd=p
    #
    #     if (energyFlag==1):
    #         return xs, ps, zs, Finst
    #     #, E_total, E_potential
    #     else:
    #         return xs, ps, zs
