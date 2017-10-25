"""
 python  main_run_mpi.py arg
 example: python main_run.py --algorithm 7 --iterations 10 --replicas 1 --nrsteps 1000
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
mdl=model.Model('Lemon')

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

args = parser.parse_args()

#read imput parameters to choose the algorithm
algoFlag=args.algoFlag

if(algoFlag=='std' or algoFlag=='0'):
    iAlgo=0
    algoName = 'std'
elif(algoFlag=='eftad_fixed_cv' or algoFlag=='1'):
    iAlgo=1
    algoName = 'eftad_fixed_cv'
elif(algoFlag=='eftad_diffmap_local' or algoFlag=='2'):
    iAlgo=2
    algoName = 'eftad_diffmap_local'
elif(algoFlag=='eftad_diffmap_only' or algoFlag=='3'):
    iAlgo=3
    algoName = 'eftad_diffmap_only'
elif(algoFlag=='eftad_diffmap_kinEn' or algoFlag=='4'):
    iAlgo=4
    algoName = 'eftad_diffmap_kinEn'
elif(algoFlag=='modif_kinEn_force' or algoFlag=='5'):
    iAlgo=5
elif(algoFlag=='modif_kinEn' or algoFlag=='6'):
    iAlgo=6
    algoName = 'modif_kinEn'
elif(algoFlag=='initial_condition' or algoFlag=='7'):
    iAlgo=7
    algoName ='initial_condition'
elif(algoFlag=='frontier_points' or algoFlag=='8'):
    iAlgo=8
    algoName = 'frontier_points'
elif(algoFlag=='frontier_points_change_temperature' or algoFlag=='9'):
    iAlgo=9
    algoName = 'frontier_points_change_temperature'
elif(algoFlag=='frontier_points_corner' or algoFlag=='10'):
    iAlgo=10
    algoName = 'frontier_points_corner'

else:
    print('Error: wrong algorithm flag. ')

# parameters
T=300.0#400
temperature =  T * unit.kelvin#300 * unit.kelvin
kT = kB * temperature


gamma = 1.0 / unit.picosecond
dt = 2 * unit.femtosecond#2.0 * unit.femtosecond

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
if iAlgo ==0:
    dataFileName='Data/'+str(algoName)+'/Traj/'
else:
    #dataFileName='Data/'+str(algoName)+'/Tsc'+str(int(TemperatureTAMDFactor))+'MS'+str(int(massScale))+'/Traj/'
    dataFileName='Data/'+str(algoName)+'/Traj/'

newpath = os.path.join(os.getcwd(),dataFileName)#+ general_sampler.algorithmName
if not os.path.exists(newpath):
    os.makedirs(newpath)

# simulation class sampler takes integrator class with chosen parameters as input
integrator=integrator.Integrator( model=mdl, gamma=gamma, temperature=temperature, temperatureAlpha=temperatureAlpha, dt=dt, massScale=massScale, gammaScale=gammaScale, kappaScale=kappaScale)


general_sampler=sampler.Sampler(model=mdl, integrator=integrator, algorithm=iAlgo, dataFileName=dataFileName)

# nrSteps is number of steps for each nrRep , and iterate the algo nrIterations times - total simulation time is nrSteps x nrIterations
nrSteps=args.nrSteps
nrEquilSteps = 0#10000 #10000
nrIterations=args.niterations
nrRep=args.nreplicas

print('Simulation time: '+repr(nrSteps*nrIterations*dt.value_in_unit(unit.femtosecond))+' '+str(unit.femtosecond)+'\n ***** \n' )
#
if (nrEquilSteps>0):
    print('Equilibration: '+repr(nrEquilSteps*dt.value_in_unit(unit.femtosecond))+' '+str(unit.femtosecond)+'\n ***** \n' )
    general_sampler.runStd(nrEquilSteps, 1, 1)
    general_sampler.resetInitialConditions()
# run the simulation
print('\n ****\n Starting simulation\n')
general_sampler.run(nrSteps, nrIterations, nrRep)
