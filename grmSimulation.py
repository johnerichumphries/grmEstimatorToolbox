#!/usr/bin/env python
''' ---------------------------------------------------------------------------

    Copyright 2013    Philipp Eisenhauer, Stefano Mosso
    
    This file is part of the Generalized Roy Toolbox. 
    
    The Generalized Roy Toolbox is free software: you can redistribute it 
    and/or modify it under the terms of the GNU General Public License as 
    published by the Free Software Foundation, either version 3 of the 
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.    
 
    ---------------------------------------------------------------------------
 
    This module contains the capabilities required for the simulation
    of the Generalized Roy Model.
 
'''

# standard library
import numpy as np

# project library
import grmReader  

def simulate():
    ''' Simulate data generation process of the Generalized Roy Model.
    
    '''
    # Process initFile.
    initDict = grmReader.read()

    ''' Distribute parametrization and (limited) type conversions.
    '''
    numAgents  = initDict['numAgents']
    fileName   = initDict['fileName']
    
    Y1_beta    = np.array(initDict['Y1_beta'])
    Y0_beta    = np.array(initDict['Y0_beta'])
    
    D_gamma    = np.array(initDict['D_gamma'])
    
    U1_var     = initDict['U1_var'] 
    U0_var     = initDict['U0_var'] 
    V_var      = initDict['V_var']
    
    U1V_rho    = initDict['U1V_rho']  
    U0V_rho    = initDict['U0V_rho']  
    
    randomSeed = initDict['randomSeed']  
    
    ''' Set random seed
    '''
    np.random.seed(randomSeed)
    
    ''' Construct auxiliary objects.
    '''
    numCovarsOut  = Y1_beta.shape[0]
    numCovarsCost = D_gamma.shape[0]
    
    U1V_cov      = U1V_rho*np.sqrt(U1_var)*np.sqrt(V_var)
    U0V_cov      = U0V_rho*np.sqrt(U0_var)*np.sqrt(V_var)
    
    ''' Simulate observable agent characteristics.
    '''
    means = np.tile(0.0, numCovarsOut)
    covs  = np.identity(numCovarsOut)
    
    X      = np.random.multivariate_normal(means, covs, numAgents)
    X[:,0] = 1.0
    
    means = np.tile(0.0, numCovarsCost)
    covs  = np.identity(numCovarsCost)
    
    Z     = np.random.multivariate_normal(means, covs, numAgents)
    
    ''' Construct level indicators for outcomes and choices. 
    '''
    print Y0_beta
    Y1_level = np.dot(Y1_beta, X.T)
    Y0_level = np.dot(Y0_beta, X.T)
    D_level  = np.dot(D_gamma, Z.T)
    
    ''' Simulate unobservables from the model.
    '''
    means = np.tile(0.0, 3)
    vars_ = [U1_var, U0_var, V_var]
    
    covs  = np.diag(vars_)
    
    covs[0,2] = U1V_cov 
    covs[2,0] = covs[0,2]
    
    covs[1,2] = U0V_cov
    covs[2,1] = covs[1,2]
    
    U = np.random.multivariate_normal(means, covs, numAgents)
    
    ''' Simulate individual outcomes and choices.
    '''
    Y1 = np.tile(np.nan, (numAgents))
    Y0 = np.tile(np.nan, (numAgents))
    Y  = np.tile(np.nan, (numAgents))
    
    D  = np.tile(np.nan, (numAgents))
    
    for i in range(numAgents):
        
        # Distribute unobservables.
        U1 = U[i,0]
        U0 = U[i,1]
        V  = U[i,2]
    
        # Decision Rule.
        expectedBenefits = Y1_level[i] - Y0_level[i]
        cost             = D_level[i]  + V 
        
        D[i] = np.float((expectedBenefits - cost > 0))
        
        # Potential outcomes.
        Y1[i] = Y1_level[i] + U1
        Y0[i] = Y0_level[i] + U0
        
        # Observed outcomes.
        Y[i]  = D[i]*Y1[i] + (1.0 - D[i])*Y0[i]
        
    ''' Check quality of simulated sample. 
    '''
    assert (np.all(np.isfinite(Y1)))
    assert (np.all(np.isfinite(Y0)))
    
    assert (np.all(np.isfinite(Y)))
    assert (np.all(np.isfinite(D)))
    
    assert (Y1.shape == (numAgents, ))
    assert (Y0.shape == (numAgents, ))
    
    assert (Y.shape  == (numAgents, ))
    assert (D.shape  == (numAgents, ))
    
    assert (Y1.dtype == 'float')
    assert (Y0.dtype == 'float')
    
    assert (Y.dtype == 'float')
    assert (D.dtype == 'float')
    
    assert ((D.all() in [1.0, 0.0]))
       
    ''' Export sample to *.txt file for further processing. 
    '''
    np.savetxt(fileName, np.column_stack((Y, D, X, Z)), fmt= '%8.3f')
    
''' Executable.
'''
if __name__ == '__main__':
    
    simulate()