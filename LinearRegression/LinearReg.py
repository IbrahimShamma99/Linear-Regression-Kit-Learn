import numpy as np
from Plots.main import Plots

class LinearReg(Plots):
    
    def __init__(self,iters=300):
        self.__iters = iters
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
    
    def __Implementation(self, X , Y):
        self.n_samples = len(Y)
        learning_rate = 0.01
        self.n_samples = len(Y)
        mu = np.mean(X, 0)
        sigma = np.std(X, 0)
        X = (X-mu) / sigma
        X = np.hstack((np.ones((self.n_samples,1)),X))
        n_features = np.size(X,1)
        params = np.zeros((n_features,1))
        self.initial_cost = self.__compute_cost(X, Y, params)
        (self.cost_history, self.optimal_params) = self.__gradient_descent(X,Y, params, learning_rate)
    
    def run(self, X , Y):
        self.X = X 
        self.Y = Y
        self.Split()
        self.__Implementation(X, Y)
    
    def getInitialCost(self):
        return self.initial_cost
    
    def getOptimalParams(self):
        return self.optimal_params
    
    def getCostHistory(self):
        return self.cost_history
    
    def splitProperty(self,prop):
        #FIXME 
        indices = np.random.permutation(prop.shape[0])
        training_idx , test_idx = indices[:80], indices[80:]
        X_training, X_test = prop[training_idx,:], prop[test_idx,:]
        
    def Split(self):
        pass
