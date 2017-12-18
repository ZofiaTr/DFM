"""Integration steps"""

# Author: Zofia
# License: 

import numpy as np

    
def B(v, f, dt):
        
    v = v + dt * f

    return v

def A(x, p, dt, mass):
        
    x = x + dt * p / float(mass)

    return x

   
def O(p, R, dt, a2, b2):
        
        
    y = a2*p + b2*R ;

    return y
    
def O_const(gamma, dt, m, kBT):
        
    a=np.exp(-gamma*dt)
     
    b = np.sqrt(kBT*(1-a*a)*m)
    
    return a, b

def AK(x, fK, dt):
        
    x = x + dt * fK

    return x