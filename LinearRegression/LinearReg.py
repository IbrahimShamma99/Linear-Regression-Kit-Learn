import numpy as np
from Plots.main import Plots

class LinearReg(Plots):
    
    #NOTE Initial state
    def __init__(self,iters=300):
        self.__iters = iters
        self.n_samples=0
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
        learning_rate = 0.01
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
        try:
            return self.initial_cost
        except TypeError:
            return 0
    def getOptimalParams(self):
        try:
            return self.optimal_params
        except TypeError:
            return 0    
            
    def getCostHistory(self):
        try:
            return self.cost_history
        except TypeError:
            return 0
    
    def splitProperty(self,prop,ratio):
        #FIXME 
        if (self.n_samples != 0):
            self.n_samples = len(prop)
        division = np.round(self.n_samples * ratio)     
        indices = np.random.permutation(prop.shape[0])
        training_idx , test_idx = indices[:division], indices[division:]
        prop_training, prop_test = prop[training_idx,:], prop[test_idx,:]
        return prop_training , prop_test
        
    def Split(self,X , Y , train_ratio=0.7):
        #Train 
        X_train , X_test = self.splitProperty(self,X,train_ratio)
        #Test
        Y_train , Y_test = self.splitProperty(self,Y,1-train_ratio)        
        return X_train , Y_train , X_test , Y_test