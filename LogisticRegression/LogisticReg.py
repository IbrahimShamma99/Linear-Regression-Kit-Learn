import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import seaborn as sns


class LogisticReg():
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

    def run(self, X , y):
        m = len(y)
        X = np.hstack((np.ones((m,1)),X))
        n = np.size(X,1)
        params = np.zeros((n,1))
        iterations = 1500
        learning_rate = 0.03
        initial_cost = self.compute_cost(X, y, params)
        print("Initial Cost is: {} \n".format(initial_cost))
        (cost_history, params_optimal) = self.gradient_descent(X, y, params, learning_rate, iterations)

        print("Optimal Parameters are: \n", params_optimal, "\n")
        plt.figure()
        sns.set_style('white')
        plt.plot(range(len(cost_history)), cost_history, 'r')
        plt.title("Convergence Graph of Cost Function")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.show()



X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=1,
                            n_clusters_per_class=1, random_state=14)
y = y[:,np.newaxis]
sns.set_style('white')
sns.scatterplot(X[:,0],X[:,1],hue=y.reshape(-1));


LogReg = LogisticReg()
LogReg.run(X , y)

y_pred = predict(X, params_optimal)
score = float(sum(y_pred == y))/ float(len(y))
print(score)