#!/usr/bin/env python
''' ---------------------------------------------------------------------------

     
    
    This file is a proposed contribution to the Generalized Roy Toolbox by Philipp Eisenhauer, Stefano Mosso (  Copyright 2013 )
    
    This module was Developed by John Eric Humphries, for the fulfillment of a class in computational economics
    at the University of Chicago. This module comes with no warranty. 
    
    This code draws heavily on the methods and coding structure used by Eisenhauer and Mosso in grmSimulation and gdmEstimation
 
    ---------------------------------------------------------------------------
 
    This module contains the capabilities required for taking a converged run from grmEstimation.py,
    generating simulated data (in parallel), and using the simulation to estimate ATE, TT, and ATE.
 
'''

# standard library
import os
import sys
import json
import  numpy  as  np
#from    scipy.stats     import  norm
import random
#from mpi4py import MPI

#comm = MPI.COMM_WORLD
#size = comm.Get_size()
#rank = comm.Get_rank()

''' Public Interface.
'''
def postEstimate():
    ''' Public interface to request treatment effects from grmEstimation.py Run
    
    '''
    if 1 == 1:
    #if rank == 0:
        # Checks.
        assert (os.path.exists('grmRslt.json')) 
    
        # Reading in .
        paramDict =  open('grmRslt.json').read()
        paramDict =  json.loads(paramDict)
       
        ''' Distribute parametrization and (limited) type conversions.
        '''
        numAgents  = paramDict['numAgents']
        fileName   = paramDict['fileName']
        Y1_beta    = np.array(paramDict['Y1_beta'])
        Y0_beta    = np.array(paramDict['Y0_beta'])
        
        D_gamma    = np.array(paramDict['D_gamma'])
        
        U1_var     = paramDict['U1_var'] 
        U0_var     = paramDict['U0_var'] 
        V_var      = 1
        
        U1V_rho    = paramDict['U1V_rho']  
        U0V_rho    = paramDict['U0V_rho']  
        
        randomSeed = paramDict['randomSeed'] 
        
        print paramDict
        
        ''' Set random seed
        '''
        np.random.seed(randomSeed)
        
        ''' Construct auxiliary objects.
        '''
        #numCovarsOut  = Y1_beta.shape[0]
        #numCovarsCost = D_gamma.shape[0]
        
        U1V_cov      = U1V_rho*np.sqrt(U1_var)*np.sqrt(V_var)
        U0V_cov      = U0V_rho*np.sqrt(U0_var)*np.sqrt(V_var)
        
        ''' Reading in Observed Data
        '''
        simAgents = 10000 # simulated agents PER CORE!
        grmData =  np.fromfile('grmData.dat', sep=" ")
        grmData = np.reshape(grmData,(numAgents,7))
        grmData = grmData[:,2:8]
        
        # A test that its an np.array as needed)
        assert( isinstance(grmData,np.ndarray) == True)
    
        ''' Randomly Draw from actual Agents Characteristics.
        '''
        print "HI"
        choice  = random.choice
        grmData = grmData.tolist()
        
    #grmData = comm.scatter(data,root==0)
    
    
    simData = [choice(grmData) for _ in xrange(simAgents)]
    print "simData", simData[0:3]
    simData = np.reshape(simData,(simAgents,5))

     
    # A test that its an np.array as needed)
    assert( isinstance(simData,np.ndarray) == True)

    ''' Constructing SIMULATED X and Z, so we can then construct outcomes. 
    '''
    
    X     = simData[:,0:3]
    Z     = simData[:,3:5]
    print "X", X[1:3], "Z" , Z[1:3]
    ''' Construct level indicators for outcomes and choices. 
    '''
    Y1_level = np.dot(Y1_beta, X.T)
    Y0_level = np.dot(Y0_beta, X.T)
    D_level  = np.dot(D_gamma, Z.T)
    
    ''' Simulate unobservables from the model.
    '''
    # Should carefully consider 
    
    means = np.tile(0.0, 3)
    vars_ = [U1_var, U0_var, V_var]
    
    covs  = np.diag(vars_)
    
    covs[0,2] = U1V_cov 
    covs[2,0] = covs[0,2]
    
    covs[1,2] = U0V_cov
    covs[2,1] = covs[1,2]
    
    U = np.random.multivariate_normal(means, covs, simAgents)
    print "U", U
    ''' Simulate individual outcomes and choices.
    '''
    
        # Distribute unobservables.
    U1 = U[:,0]
    U0 = U[:,1]
    V  = U[:,2]
    
    # Potential outcomes.
    Y1 = Y1_level + U1
    Y0 = Y0_level + U0
    
    # Some calculations outside the loop
    EB = Y1_level - Y0_level

    # Decision Rule.
    cost = D_level  + V     
    D = np.array((EB - cost > 0))
        
    # Observed outcomes.
    Y  = D*Y1 + (1.0 - D)*Y0

    ''' Check quality of simulated sample. 
    '''
    assert (np.all(np.isfinite(Y1)))
    assert (np.all(np.isfinite(Y0)))
    
    assert (np.all(np.isfinite(Y)))
    assert (np.all(np.isfinite(D)))
    
    assert (Y1.shape == (simAgents, ))
    assert (Y0.shape == (simAgents, ))
    
    assert (Y.shape  == (simAgents, ))
    assert (D.shape  == (simAgents, ))
    
    assert (Y1.dtype == 'float')
    assert (Y0.dtype == 'float')
    
    assert (Y.dtype == 'float')
    
    assert ((D.all() in [1.0, 0.0]))
    
    
    ''' Building the Treatment Effect Variables
    '''
    ATE = (Y1-Y0).sum() / simAgents
    TT =  ((Y1-Y0)*D).sum() / np.sum(D)
    TUT =  np.sum((Y1-Y0)*(1-D)) / np.sum(1-D)
    
    ''' Some Anti-debugging
    '''
    assert(isinstance(ATE,float))
    assert(isinstance(TT,float))
    assert(isinstance(TUT,float))
    
    ests= {}
    ests['ATE'] = ATE
    ests['TT']  = TT
    ests['TUT'] =TUT
    print ests
    with open('grmEsts.json', 'w') as file_:
        json.dump(ests, file_)
        
# Calling Function
postEstimate()

##############################################################################################

''' Private Functions.
'''
def _initializeLogging():
    ''' Prepare logging.
    '''
    with open('grmLogging.txt', 'w') as file_:
        file_.write('\n LogFile of Generalized Roy Toolbox \n\n')
