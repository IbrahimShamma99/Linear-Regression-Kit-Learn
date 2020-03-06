import numpy as np
from Plots.main import Plots
from Cleaner.main import Cleaner
from .Commons import Common
from Evaluator.main import Evaluator
'''FIXME 
Let your parameters private to not affect your model
'''
class LinearReg(Plots,Cleaner,Common,Evaluator):
    
    #NOTE Initial state
    def __init__(self,n_iterations=300):
        self.__iters = n_iterations
        self.n_samples = 0 
        pass
    
    def __gradient_descent(self,X, y, learning_rate):
        n_samples = len(y)
        cost_history = np.zeros((self.__iters,1))

        for i in range(self.__iters):
            self.params = self.params - (learning_rate/n_samples) * X.T @ (X @ self.params - y) 
            cost_history[i] = self.compute_cost(X, y)
        return (cost_history, self.params)
    
    #NOTE Here we implement the algorithm
    #TODO fit 
    def fit(self, X , Y):
        self.intializeParameters(X ,Y)
        self.initial_cost = self.compute_cost(self.X, self.Y)
        (self.cost_history, self.optimal_params) = self.__gradient_descent(self.X,Y, self.learning_rate)
    
    def run(self, X , Y, train_ratio):
        self.X = X 
        self.Y = Y
        self.n_samples = len(Y)
        self.X_train ,self.Y_train , self.X_test , self.Y_test  = self.Split(X , Y , train_ratio)
        self.fit(self.X_train, self.Y_train)
        self.test(self.X_test, self.Y_test)
        
    def intializeParameters(self,X , Y):
        self.n_train_samples = len(Y)
        self.learning_rate = 0.01
        mu = np.mean(X, 0)
        sigma = np.std(X, 0)
        X = (X-mu) / sigma
        X = np.hstack((np.ones((self.n_train_samples,1)),X))
        n_features = np.size(X,1)
        params = np.zeros((n_features,1))
        self.params = params
        self.X = X
    
    def predict(self, X): 
        return X @ self.params
    
    def test(self, X , Y):
        self.rmse(X , Y)