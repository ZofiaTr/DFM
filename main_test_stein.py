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

# folder where the data is saved
dataFolderName = 'Data_Tests/Alanine/'

# intialize sampler class together wuth the model
mdl=model.Model('Alanine')
intg=integrator.Integrator( model=mdl, gamma=1.0 / unit.picosecond, temperature=300 * unit.kelvin, dt=2.0 * unit.femtosecond,  temperatureAlpha=300 * unit.kelvin)
smpl=sampler.Sampler(model=mdl, integrator=intg, algorithm=0, dataFileName='Data')

# stein
st = stein.Stein(smpl, dataFolderName, modnr = 10)
# change the stein step size
st.epsilon_step=unit.Quantity(0.001, smpl.model.x_unit)**2

#run stein
st.run_stein(numberOfSteinSteps = 100)
