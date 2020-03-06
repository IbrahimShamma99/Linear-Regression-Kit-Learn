import numpy as np
from Plots.main import Plots
from Cleaner.cleaner import Cleaner

class LinearReg(Plots,Cleaner):
    
    #NOTE Initial state
    def __init__(self,n_iterations=300):
        self.__iters = n_iterations
        self.n_samples = 0 
        pass
    
    def __compute_cost(self,X, y, params):
        n_samples = len(y)
        h = X @ params
        return (1/(2*n_samples))*np.sum((h-y)**2)

    def __gradient_descent(self,X, y, params, learning_rate):
        n_samples = len(y)
        cost_history = np.zeros((self.__iters,1))

        for i in range(self.__iters):
            params = params - (learning_rate/n_samples) * X.T @ (X @ params - y) 
            cost_history[i] = self.__compute_cost(X, y, params)
        return (cost_history, params)
    
    #NOTE Here we implement the algorithm
    def __Implementation(self, X , Y):
        if (self.n_samples == 0):
            self.n_samples = len(Y)
        learning_rate = 0.01
        mu = np.mean(X, 0)
        sigma = np.std(X, 0)
        X = (X-mu) / sigma
        X = np.hstack((np.ones((self.n_samples,1)),X))
        n_features = np.size(X,1)
        params = np.zeros((n_features,1))
        self.initial_cost = self.__compute_cost(X, Y, params)
        (self.cost_history, self.optimal_params) = self.__gradient_descent(X,Y, params, learning_rate)
    
    def run(self, X , Y, train_ratio):
        self.X = X 
        self.Y = Y
        print("\nHERE\n")
        # X_train ,Y_train , X_test , Y_test  = self.Split(X , Y , train_ratio)
        self.__Implementation(X, Y)
    
    def getInitialCost(self):
        try:
            return self.initial_cost
        except TypeError:
            raise Exception('Model was not run yet')
        
    def getOptimalParams(self):
        try:
            return self.optimal_params
        except TypeError:
            raise Exception('Model was not run yet')
            
    def getCostHistory(self):
        try:
            return self.cost_history
        except TypeError:
            raise Exception('Model was not run yet')