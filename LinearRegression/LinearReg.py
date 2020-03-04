import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

dataset = load_boston()
X = dataset.data
Y = dataset.target[:,np.newaxis]

n_iters = 1500


class LinearReg():
    def compute_cost(self,X, y, params):
        n_samples = len(y)
        h = X @ params
        return (1/(2*n_samples))*np.sum((h-y)**2)

    def gradient_descent(self,X, y, params, learning_rate, n_iters):
        n_samples = len(y)
        J_history = np.zeros((n_iters,1))

        for i in range(n_iters):
            params = params - (learning_rate/n_samples) * X.T @ (X @ params - y) 
            J_history[i] = self.compute_cost(X, y, params)

        return (J_history, params)


    def run(self, X , Y):
        n_samples = len(Y)
        mu = np.mean(X, 0)
        sigma = np.std(X, 0)
        X = (X-mu) / sigma
        X = np.hstack((np.ones((n_samples,1)),X))
        n_features = np.size(X,1)
        params = np.zeros((n_features,1))
        learning_rate = 0.01
        initial_cost = self.compute_cost(X, Y, params)
        (J_history, optimal_params) = self.gradient_descent(X,Y, params, learning_rate, n_iters)
        print("Initial cost is: ", initial_cost, "\n")
        print("Optimal parameters are: \n", optimal_params, "\n")
        print("Final cost is: ", J_history[-1])
        plt.plot(range(len(J_history)), J_history, 'r')
        plt.title("Convergence Graph of Cost Function")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.show()
    
    
lin = LinearReg()
lin.run(X,Y)