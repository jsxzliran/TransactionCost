import numpy as np
from datetime import datetime
import pandas as pd
import math
#import bottleneck as bn

# a class describing the features of a simulated path matrix
class ManyStocks:
    def __init__(self,  seed, num, mu, s0, cov, npaths, nsteps, T):
        self.seed = seed
        self.mu = mu
        self.s0 = s0
        self.npaths = npaths
        self.nsteps = nsteps
        self.T = T
        self.num = num
        self.cov = cov
    

    # create a price matrix
    def Prices(self):
        np.random.seed(self.seed)
        dt = self.T/self.nsteps
        W1 = np.random.normal(size=(self.num,self.nsteps,self.npaths ))
        CHOL = np.linalg.cholesky(self.cov)
        sigma_square = np.diag(self.cov)
        logS0 = np.log(self.s0).reshape([self.num, 1, 1])
        W2 = np.transpose(np.dot(np.transpose(W1),np.transpose(CHOL)))

        if len(self.mu.shape) > 1:
            dlogS = ((self.mu - \
                      1/2*sigma_square.reshape([self.num_stocks, 1]))*dt).reshape([self.num,self.nsteps,1])+\
                        (math.sqrt(dt))*W2
        else:
            dlogS = ((self.mu - 1/2*sigma_square)*dt).reshape([self.num,1,1])+\
            (math.sqrt(dt))*W2
        prices = np.exp(logS0.reshape([self.num,1,1])+np.cumsum(dlogS,axis = 1))
        s0 = self.s0.reshape((self.num,1,1))*np.ones((self.num,1,self.npaths))
        prices = np.concatenate((s0,prices),axis=1)
        return prices

    # create a return matrix
    def Returns(self):
        np.random.seed(self.seed)
        dt = self.T/self.nsteps
        W2 = np.random.normal(size=(self.num,self.nsteps,self.npaths ))
        CHOL = np.linalg.cholesky(self.cov)
        sigma_square = np.diag(self.cov)
        logS0 = np.log(self.s0).reshape([self.num, 1, 1])
        W2 = np.transpose(np.dot(np.transpose(W2),np.transpose(CHOL)))
        if len(self.mu.shape) > 1:
            dlogS = ((self.mu - \
                      1/2*sigma_square.reshape([self.num_stocks, 1]))*dt).reshape([self.num,self.nsteps,1])+\
                        (math.sqrt(dt))*W2
        else:
            dlogS = ((self.mu - 1/2*sigma_square)*dt).reshape([self.num,1,1])+\
            (math.sqrt(dt))*W2

        prices = np.exp(logS0.reshape([self.num,1,1])+np.cumsum(dlogS,axis = 1))
        s0 = self.s0.reshape((self.num,1,1))*np.ones((self.num,1,self.npaths))
        prices = np.concatenate((s0,prices),axis=1)
        returns = np.exp(np.diff(np.log(prices), axis=1))-1
        return returns
        
    # Output the dW_t matrix
    def BM(self):
        np.random.seed(self.seed)
        dt = self.T/self.nsteps
        W2 = np.random.normal(size=(self.num,self.nsteps,self.npaths ))
        std_devs = np.sqrt(np.diag(self.cov))
        outer_std_devs = np.outer(std_devs, std_devs)
        corr = self.cov / outer_std_devs
        CHOL = np.linalg.cholesky(corr)
        W2 = np.transpose(np.dot(np.transpose(W2),np.transpose(CHOL)))
        dW = (math.sqrt(dt))*W2
        return dW