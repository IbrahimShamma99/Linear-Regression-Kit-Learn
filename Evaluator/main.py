import numpy as np

class Evaluator():
    def __init__(self):
         pass
    
    def rmse(self,predictions, targets):
        differences = predictions - targets                       
        differences_squared = differences ** 2                    
        mean_of_differences_squared = differences_squared.mean()  
        rmse_val = np.sqrt(mean_of_differences_squared)           
        return rmse_val