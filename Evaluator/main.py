import numpy as np

class Evaluator():
    def __init__(self):
         pass
    
    def compute_cost(self,X, Y):
        n_samples = len(Y)
        h = self.predict(X)
        return (1/(2*n_samples))*np.sum((h-Y)**2)

    def rmse(self,predictions, targets):
        n_samples = len(targets)
        Error = 0
        for n in range(n_samples):
            differences = predictions[n] - targets[n]                       
            differences_squared = differences ** 2                    
            mean_of_differences_squared = differences_squared.mean()  
            rmse_val = np.sqrt(mean_of_differences_squared)  
            Error += rmse_val      
        self.RMSE_Error = Error