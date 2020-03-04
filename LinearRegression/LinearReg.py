import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.getcwd()) 

from Plots.ErrorPlot import Plots  as plot

class LinearReg():
    
    def __init__(self,iters=1500):
        #TODO To let the model switch between different linear algorithms
        self.__iters = iters
        # self.initial_cost=0
        # self.optimal_params = 0
        # self.J_history = []
        pass 
              
    def __compute_cost(self,X, y, params):
        n_samples = len(y)
        h = X @ params
        return (1/(2*n_samples))*np.sum((h-y)**2)

    def __gradient_descent(self,X, y, params, learning_rate):
        n_samples = len(y)
        J_history = np.zeros((self.__iters,1))

        for i in range(self.__iters):
            params = params - (learning_rate/n_samples) * X.T @ (X @ params - y) 
            J_history[i] = self.__compute_cost(X, y, params)

        return (J_history, params)
    
    def plotError(self):
         plot.plotError(self)
    
    def run(self, X , Y):
        n_samples = len(Y)
        mu = np.mean(X, 0)
        sigma = np.std(X, 0)
        X = (X-mu) / sigma
        X = np.hstack((np.ones((n_samples,1)),X))
        n_features = np.size(X,1)
        params = np.zeros((n_features,1))
        learning_rate = 0.01
        initial_cost = self.__compute_cost(X, Y, params)
        (J_history, optimal_params) = self.__gradient_descent(X,Y, params, learning_rate)
        self.initial_cost = initial_cost
        self.J_history = J_history
        self.optimal_params = optimal_params
        #self.plot(initial_cost , optimal_params,J_history)
