import numpy as np
from sklearn.datasets import make_classification
import sys
import os
sys.path.append(os.getcwd())
from Plots.ErrorPlot import ErrPlots


class LogisticReg(ErrPlots):
    '''
    LogReg = LogisticReg()
    LogReg.run(X , y)
    LogReg.plotError()
    '''
    def __init__(self):
        pass
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def compute_cost(self,X,Y, theta):
        m = len(Y)
        h = self.sigmoid(X @ theta)
        epsilon = 1e-5
        cost = (1/m)*(((-Y).T @ np.log(h + epsilon))-((1-Y).T @ np.log(1-h + epsilon)))
        return cost

    def gradient_descent(self,X, Y, params, learning_rate, iterations):
        m = len(Y)
        cost_history = np.zeros((iterations,1))

        for i in range(iterations):
            params = params - (learning_rate/m) * (X.T @ (self.sigmoid(X @ params) - Y)) 
            cost_history[i] = self.compute_cost(X, Y, params)

        return (cost_history, params)

    def predict(self , X, params):
        return np.round(self.sigmoid(X @ params))
    
    def score(self):
        y_pred = predict(X, params_optimal)
        score = float(sum(y_pred == y))/ float(len(y))
        print(score)
        return score
    
    def run(self, X , y):
        m = len(y)
        X = np.hstack((np.ones((m,1)),X))
        n = np.size(X,1)
        params = np.zeros((n,1))
        iterations = 1500
        learning_rate = 0.03
        initial_cost = self.compute_cost(X, y, params)
        print("Initial Cost is: {} \n".format(initial_cost))
        (cost_history, optimal_params) = self.gradient_descent(X, y, params, learning_rate, iterations)
        self.cost_history = cost_history
        self.optimal_params = optimal_params
        self.initial_cost = initial_cost