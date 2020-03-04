import matplotlib.pyplot as plt

# NOTE For internal usage
class ErrPlots():
    def __init__(self):
        pass
    
    def plot(self):
        print("Initial cost is: ", self.initial_cost, "\n")
        print("Optimal parameters are: \n", self.optimal_params, "\n")
        print("Final cost is: ", self.cost_history[-1])
        plt.plot(range(len(self.J_history)), self.J_history, 'r')
        plt.title("Convergence Graph of Cost Function")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.show()
