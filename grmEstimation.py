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
 
    This module contains the capabilities required for the estimation
    of the Generalized Roy Model.
 
'''

# standard library
import os
import sys
import json

import  numpy           as      np

from    scipy.stats     import  norm
from    scipy.optimize  import  fmin_bfgs

# project library
import grmReader

''' Public Interface.
'''
def estimate():
    ''' Public interface to request an estimation of the Generalized
        Roy Model.
    
    '''
    # Checks.
    assert (os.path.exists('grmInit.ini')) 
    
    # Process initialization file.
    initDict = grmReader.read()
    
    # Checks.
    assert (os.path.exists(initDict['fileName'])) 
    
    # Process initFile.
    _initializeLogging()
    
    ''' Distribute useful covariates.
    '''
    numAgents = initDict['numAgents']

    Y1_beta   = np.array(initDict['Y1_beta'])
    Y0_beta   = np.array(initDict['Y0_beta'])
    
    D_gamma   = np.array(initDict['D_gamma'])    
    
    U1_var    = initDict['U1_var'] 
    U0_var    = initDict['U0_var'] 
    
    U1V_rho   = initDict['U1V_rho']  
    U0V_rho   = initDict['U0V_rho']  
    
    maxiter   = initDict['maxiter']
    
    ''' Construct auxiliary objects.
    '''
    numCovarsOut  = Y1_beta.shape[0]
    numCovarsCost = D_gamma.shape[0]
    
    ''' Read and check dataset, distribute entries.
    '''
    data = np.genfromtxt(initDict['fileName'], dtype = 'float')

    assert (_checkData(data, numAgents, numCovarsOut, numCovarsCost) == True)
    
    Y = data[:,0]
    D = data[:,1]
    
    X = data[:,2:(numCovarsOut + 2)]
    Z = data[:,-numCovarsCost:]
    
    ''' Maximization Script.
    '''
    
    # Construct starting values.    
    startVals = np.concatenate((Y1_beta, Y0_beta, D_gamma, 
                    [U1_var], [U0_var], [U1V_rho], [U0V_rho]))
                               
    # Run maximization.
    sys.stdout = open('grmLogging.txt', 'a')
    
    rslt = fmin_bfgs(_maxAlgorihtmInterface, startVals, \
                     args = (Y, D, X, Z), maxiter = maxiter, \
                     full_output = False)
    sys.stdout = sys.__stdout__
     
    # Construct dictionary with results.
    rslt = _distributeEvaluationValues(rslt, numCovarsOut, False)
    rslt['Y1_beta']= rslt['Y1_beta'].tolist()
    rslt['Y0_beta']= rslt['Y0_beta'].tolist()
    rslt['D_gamma']= rslt['D_gamma'].tolist()
    rslt['numAgents']   = numAgents
    rslt['fileName']    = initDict['fileName']
    rslt['randomSeed']  = initDict['randomSeed']
    print initDict
    #  Write out the *.json file.
    with open('grmRslt.json', 'w') as file_:
        
        json.dump(rslt, file_)
    
''' Private Functions.
'''
def _initializeLogging():
    ''' Prepare logging.
    
    '''
    with open('grmLogging.txt', 'w') as file_:
        
        file_.write('\n LogFile of Generalized Roy Toolbox \n\n')
        
def _checkData(data, numAgents, numCovarsOut, numCovarsCost):
    ''' Basic checks for the data.
    
    '''
    # Antibugging.
    assert (isinstance(data, np.ndarray))
    assert (isinstance(numAgents, int))
    assert (numAgents > 0)
    assert (isinstance(numCovarsOut, int))
    assert (numCovarsOut > 0)
    assert (isinstance(numCovarsCost, int))
    assert (numCovarsCost > 0)
        
    # Checks.
    assert (data.shape == (numAgents, (1 + 1 + numCovarsOut +  numCovarsCost))) 
    assert (np.all(np.isfinite(data)))
    assert ((data[:,1].all() in [1.0, 0.0]))
    
    # Finishing.
    return True
    
def _maxAlgorihtmInterface(x, Y, D, X, Z):
    ''' Interface to the SciPy maximization routines.
    
    '''
    # Auxiliary objects.
    numCovarsOut = X.shape[1]
    
    # Collect maximization arguments.
    rslt = _distributeEvaluationValues(x, numCovarsOut)
    
    # Calculate likelihood.
    likl = _negLogLiklContribution(rslt, Y, D, X, Z)
    
    # Finishing.
    return likl
  
def _distributeEvaluationValues(x, numCovarsOut, isList = False):
    ''' Distribute the evaluation values.
    
    '''
    #Antibugging.
    assert (isList in [True, False])
    
    if(isList):
        
        pass
            
    else:
        
        assert (isinstance(x, np.ndarray))
        assert (np.all(np.isfinite(x)))
        assert (x.ndim == 1)
        
    assert (isinstance(numCovarsOut, int))
    assert (numCovarsOut > 0)
        
    # Distribution of parameters.
    rslt = {}

    rslt['Y1_beta'] = x[:numCovarsOut]
    rslt['Y0_beta'] = x[numCovarsOut:(2*numCovarsOut)]
    
    rslt['D_gamma'] = x[(2*numCovarsOut):(-4)]
    
    ''' Parameters with natural bounds. 
    '''    
    rslt['U1_var']  = np.exp(x[(-4)])
    rslt['U0_var']  = np.exp(x[(-3)])

    rslt['U1V_rho'] = -1.0 + 2.0/(1.0 + np.exp(-x[-2]))    
    rslt['U0V_rho'] = -1.0 + 2.0/(1.0 + np.exp(-x[-1])) 

    # Finishing.
    return rslt

def _negLogLiklContribution(rslt, Y, D, X, Z):
    ''' Negative log-likelihood function of the Generalized Roy Model.
    
    '''
    # Distribute input.
    Y1_beta = rslt['Y1_beta']
    Y0_beta = rslt['Y0_beta']

    D_gamma = rslt['D_gamma']

    U1_var  = rslt['U1_var']
    U0_var  = rslt['U0_var']

    U1V_rho = rslt['U1V_rho']
    U0V_rho = rslt['U0V_rho']

    # Construct auxiliary objects.
    numAgents = D.shape[0]

    # Likelihood calculation.
    choiceCoeffs  = np.concatenate((Y1_beta  - Y0_beta, - D_gamma))
    choiceCovars  = np.concatenate((X, Z), axis = 1)
    choiceIndices = np.dot(choiceCoeffs, choiceCovars.T) 
    
    argOne = D * (Y - np.dot(Y1_beta, X.T))/np.sqrt(U1_var) + (1 - D)*(Y - np.dot(Y0_beta, X.T))/np.sqrt(U0_var)
   
    argTwo = D * (choiceIndices - U1V_rho*argOne)/np.sqrt(1.0 - U1V_rho**2) + \
            (1 - D) *  (choiceIndices - U0V_rho*argOne)/np.sqrt(1.0 - U0V_rho**2)
    
    cdfEvals = norm.cdf(argTwo)
    pdfEvals = norm.pdf(argOne)
    
    likl = D*(1.0/np.sqrt(U1_var))*pdfEvals*cdfEvals + \
                (1 - D) *(1.0/np.sqrt(U0_var))*pdfEvals*(1.0  - cdfEvals)
    
    # Transformations.
    likl =  np.clip(likl, 1e-20, 1.0)
    
    likl = -np.log(likl)
    
    likl = likl.sum()
    
    lik = (1.0/numAgents)*likl
        
    # Quality checks.
    assert (isinstance(lik, float))    
    assert (np.isfinite(lik))
    assert (lik > 0.0)
    
    #Finishing.        
    return lik

''' Executable.
'''
if __name__ == '__main__':
    
    estimate()

