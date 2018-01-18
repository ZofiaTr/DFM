import matplotlib.pyplot as plt
#%matplotlib inline

import sys
import os
sys.path.append( os.getcwd()+'/srcDiffmap')

import integrator
import sampler
import model

import numpy as np
from simtk import openmm, unit

import helpers
import stein

import argparse

parser = argparse.ArgumentParser(description='Run stein variational importance sampling on a trajectory.')
parser.add_argument('--dataFolder', dest='dataFolder', type=str, default='Data_Tests/Alanine/',
                    help='Read trajectory from dataFolder.')
parser.add_argument('--nrSteinSteps', dest='nrSteinSteps', type=int, default=2,
                    help='Number of stein iterations.')
parser.add_argument('--steinStepSize', dest='steinStepSize', type=float, default=0.001,
                    help='Number of stein iterations.')
parser.add_argument('--modnr', dest='modnr', type=int, default=1,
                    help='Stride the trajectory modulo modnr.')


args = parser.parse_args()
# folder where the data is saved
dataFolderName = args.dataFolder
nrSteps = args.nrSteinSteps
steinStepSize = args.steinStepSize
modnr = args.modnr

# intialize sampler class together wuth the model
mdl=model.Model('Alanine')
intg=integrator.Integrator( model=mdl, gamma=1.0 / unit.picosecond, temperature=300 * unit.kelvin, dt=2.0 * unit.femtosecond,  temperatureAlpha=300 * unit.kelvin)
smpl=sampler.Sampler(model=mdl, integrator=intg, algorithm=0, dataFileName='Data')

# stein
st = stein.Stein(smpl, dataFolderName, modnr = modnr)
# change the stein step size
st.epsilon_step=unit.Quantity(steinStepSize, smpl.model.x_unit)**2

#run stein
st.run_stein(numberOfSteinSteps = nrSteps)

np.save(dataFolderName+'/stein_final.npy', st.q)
