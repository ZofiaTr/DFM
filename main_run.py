"""
 python  main_run.py arg
 example: python main_run.py --iterations 1000 --replicas 5 --nrsteps 10000 --folderName 'Data' --algorithm 6 --diffMapMetric 'rmsd'

 python main_run.py --iterations 1000 --replicas 2 --nrsteps 1000 --folderName 'test' --algorithm 0
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

modelName = 'Alanine'
mdl=model.Model(modelName)

print(mdl.x_unit)
print(mdl.modelname)
print('System has %d particle(s)' % mdl.system.getNumParticles())
#systemName='Ala2Pept'

# Parse command-line arguments
import argparse

parser = argparse.ArgumentParser(description='Diffusion map accelerated sampling.')
parser.add_argument('--algorithm', dest='algoFlag', type=str, default='0',
                    help='algorithm name or number to use (default: 0)')
parser.add_argument('--iterations', dest='niterations', type=int, default=10000,
                    help='number of iterations to run (default: 10000)')
parser.add_argument('--replicas', dest='nreplicas', type=int, default=10,
                    help='number of replicas to use per iteration (default: 10)')
parser.add_argument('--nrsteps', dest='nrSteps', type=int, default=1000,
                    help='number of replicas to use per iteration (default: 10)')
#parser.add_argument('--temporaryFolder', dest='temporaryFile', type=int, default=0,
#                    help='0: save data into Data folder; 1: Save data to temporary folder; 2: another temporary folder ')
parser.add_argument('--folderName', dest='saveToFileName', type=str, default='Data',
                    help='Create folder and save data there')
parser.add_argument('--diffMapMetric', dest='diffMapMetric', type=str, default='euclidean',
                    help='Metric for diffusion maps. Choose between rmsd and euclidean.')


args = parser.parse_args()

saveToFileName = args.saveToFileName

#read imput parameters to choose the algorithm
algoFlag=args.algoFlag
diffMapMetric=args.diffMapMetric



if(algoFlag=='std' or algoFlag=='0'):
    iAlgo=0
    algoName = 'std'
elif(algoFlag=='eftad_fixed_cv' or algoFlag=='1'):
    iAlgo=1
    algoName = 'eftad_fixed_cv'
elif(algoFlag=='initial_condition' or algoFlag=='2'):
    iAlgo=2
    algoName ='initial_condition'
elif(algoFlag=='frontier_points_corner_change_temperature' or algoFlag=='3'):
    iAlgo=3
    algoName = 'frontier_points_corner_change_temperature'
elif(algoFlag=='frontier_points_corner_change_temperature_off' or algoFlag=='4'):
    iAlgo=4
    algoName = 'frontier_points_corner_change_temperature_off'

else:
    print('Error: wrong algorithm flag. ')

# parameters
T=300.0
temperature =  T * unit.kelvin
kT = kB * temperature


gamma = 1.0 / unit.picosecond
dt = 0.5 * unit.femtosecond

TemperatureTAMDFactor=30.0
massScale=50.0

gammaScale=100.0
kappaScale=1000.0

if iAlgo >0:
    print("TemperatureTAMDFactor = " + repr(TemperatureTAMDFactor))
temperatureAlpha= (T*TemperatureTAMDFactor)* unit.kelvin

print('Gamma is '+repr(gamma))
print('Temperature is '+repr(temperature))
if iAlgo >0:
    print('Temperature TAMD '+repr(TemperatureTAMDFactor)+'xTemperature')
    print('Mass alpha is '+repr(massScale)+'x Mass')

#create folders to save the data

dataFileName=saveToFileName+'/'+str(modelName)+'/'+str(algoName)+'/Traj/'
dataFileNameFrontierPoints=saveToFileName+'/'+str(modelName)+'/'+str(algoName)+'/Traj/FrontierPoints/'
dataFileNameEigenVectors=saveToFileName+'/'+str(modelName)+'/'+str(algoName)+'/Traj/Eigenvectors/'
dataFileEnergy=saveToFileName+'/'+str(modelName)+'/'+str(algoName)+'/Traj/Energies/'


newpath = os.path.join(os.getcwd(),dataFileName)
if not os.path.exists(newpath):
        os.makedirs(newpath)

newpath = os.path.join(os.getcwd(),dataFileNameFrontierPoints)
if not os.path.exists(newpath):
        os.makedirs(newpath)

newpath = os.path.join(os.getcwd(),dataFileNameEigenVectors)
if not os.path.exists(newpath):
        os.makedirs(newpath)

newpath = os.path.join(os.getcwd(),dataFileEnergy)
if not os.path.exists(newpath):
        os.makedirs(newpath)




# simulation class sampler takes integrator class with chosen parameters as input
integrator=integrator.Integrator( model=mdl, gamma=gamma, temperature=temperature, temperatureAlpha=temperatureAlpha, dt=dt, massScale=massScale, gammaScale=gammaScale, kappaScale=kappaScale)


general_sampler=sampler.Sampler(model=mdl, integrator=integrator, algorithm=iAlgo, diffusionMapMetric=diffMapMetric, dataFileName=dataFileName, dataFrontierPointsName = dataFileNameFrontierPoints, dataEigenVectorsName =dataFileNameEigenVectors, dataEnergyName = dataFileEnergy)

# nrSteps is number of steps for each nrRep , and iterate the algo nrIterations times - total simulation time is nrSteps x nrIterations
nrSteps=args.nrSteps
nrEquilSteps = 10000 #10000
nrIterations=args.niterations
nrRep=args.nreplicas

print('Simulation time: '+repr(nrSteps*nrIterations*dt.value_in_unit(unit.femtosecond))+' '+str(unit.femtosecond)+'\n ***** \n' )
#
if (nrEquilSteps>0):
    print('Equilibration: '+repr(nrEquilSteps*dt.value_in_unit(unit.picosecond))+' '+str(unit.picosecond)+'\n ***** \n' )
    general_sampler.run_std(nrEquilSteps, 1, 1)
    general_sampler.resetInitialConditions()
# run the simulation
print('\n ****\n Starting simulation\n')
general_sampler.run(nrSteps, nrIterations, nrRep)
