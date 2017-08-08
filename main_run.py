"""
 python  main_run_mpi.py arg

"""


import numpy as np

import sys
import os
sys.path.append( os.getcwd()+'/srcDiffmap')

import mdtraj as md
import sampler
import integrator

from simtk import openmm, unit
from openmmtools.constants import kB

import model

#create model defined in class model
mdl=model.Model()

print mdl.x_unit
print mdl.modelname
print('System has %d particle(s)' % mdl.system.getNumParticles())
#systemName='Ala2Pept'

#read imput parameters to choose the algorithm
algoFlag=sys.argv[1]

if(algoFlag=='std' or algoFlag=='0'):
    iAlgo=0
elif(algoFlag=='eftad_fixed_cv' or algoFlag=='1'):
    iAlgo=1
elif(algoFlag=='eftad_diffmap_local' or algoFlag=='2'):
    iAlgo=2
elif(algoFlag=='eftad_diffmap_only' or algoFlag=='3'):
    iAlgo=3
elif(algoFlag=='eftad_diffmap_kinEn' or algoFlag=='4'):
    iAlgo=4
elif(algoFlag=='modif_kinEn_force' or algoFlag=='5'):
    iAlgo=5
elif(algoFlag=='modif_kinEn' or algoFlag=='6'):
    iAlgo=6
else:
    print('Error: wrong algorithm flag. ')

# parameters
T=400.0#400
temperature =  T * unit.kelvin#300 * unit.kelvin
kT = kB * temperature


gamma = 1.0 / unit.picosecond
dt = 4.0 * unit.femtosecond

TemperatureTAMDFactor=30.0
massScale=50.0

gammaScale=100.0
kappaScale=1000.0

print ("TemperatureTAMDFactor = " + repr(TemperatureTAMDFactor))
temperatureAlpha= (T*TemperatureTAMDFactor)* unit.kelvin

print('Gamma is '+repr(gamma))
print('Temperature is '+repr(temperature))
print('Temperature TAMD '+repr(TemperatureTAMDFactor)+'xTemperature')
print('Mass alpha is '+repr(massScale)+'x Mass')

#create folders to save the data
if iAlgo ==0:
    dataFileName='Data/Std/Traj/'
else:
    dataFileName='Data/Tsc'+str(int(TemperatureTAMDFactor))+'MS'+str(int(massScale))+'/Traj/'

newpath = os.path.join(os.getcwd(),dataFileName)#+ general_sampler.algorithmName
if not os.path.exists(newpath):
    os.makedirs(newpath)

# simulation class sampler takes integrator class with chosen parameters as input
integrator=integrator.Integrator( model=mdl, gamma=gamma, temperature=temperature, temperatureAlpha=temperatureAlpha, dt=dt, massScale=massScale, gammaScale=gammaScale, kappaScale=kappaScale)
general_sampler=sampler.Sampler(model=mdl, integrator=integrator, algorithm=iAlgo, dataFileName=dataFileName)

# nrSteps is number of steps for each nrRep , and iterate the algo nrIterations times - total simulation time is nrSteps x nrIterations
nrSteps=1000
nrIterations=100000
nrRep=10

print('Simulation time: '+repr(nrSteps*nrIterations*dt.value_in_unit(unit.femtosecond))+' '+str(unit.femtosecond)+'\n ***** \n' )

print('Equilibration\n')
general_sampler.run(1000, 1, 1)
general_sampler.resetInitialConditions()
# run the simulation
print('Starting simulation\n')
general_sampler.run(nrSteps, nrIterations, nrRep)
