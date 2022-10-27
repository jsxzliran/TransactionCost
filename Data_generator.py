import numpy as np
from datetime import datetime
import pandas as pd
import math
import bottleneck as bn

# a class describing the features of a simulated path matrix
# Only used for one asset
class OneStock:
    def __init__(self, seed, mu, sigma, s0, npaths, nsteps, T):
        self.seed = seed
        self.mu = mu
        self.sigma = sigma
        self.s0 = s0
        self.npaths = npaths
        self.nsteps = nsteps
        self.T = T
    
    # create a price matrix
    def Prices(self):
        np.random.seed(self.seed)
        dt = self.T/self.nsteps
        W1 = np.random.normal(size=(self.nsteps,self.npaths))
        logS0 = np.log(self.s0)
        dlogS = (self.mu - 1/2*self.sigma*self.sigma)*dt+math.sqrt(dt)*self.sigma*W1
        prices = np.exp(logS0+np.cumsum(dlogS,axis=0))
        prices = np.transpose(np.row_stack((self.s0*np.ones((1,self.npaths)),prices)))
        return prices

    # create a return matrix
    def Returns(self):
        np.random.seed(self.seed)
        dt = self.T/self.nsteps
        W2 = np.random.normal(size=(self.nsteps,self.npaths))
        logS0 = np.log(self.s0)
        dlogS = (self.mu - 1/2*self.sigma*self.sigma)*dt+math.sqrt(dt)*self.sigma*W2
        prices = np.exp(logS0+np.cumsum(dlogS,axis=0))
        prices = np.row_stack((self.s0*np.ones((1,self.npaths)),prices))
        returns = np.transpose(np.exp(np.diff(np.log(prices), axis=0))-1)
        return returns