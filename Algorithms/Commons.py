
class Common():
    def __init__(self):
        pass
        
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
    
    def getXData(self):
        return self.X
    
    def getYData(self):
        return self.Y
    
    def getXTrainData(self):
        self.X_train
    
    def getYTrainData(self):
        self.Y_train
    
    def getYTestData(self):
        self.Y_test
    
    def getXTestData(self):
        return self.X_test
    
    def getTrainData(self):
        return self.getXTrainData , self.getYTrainData
    
    def getTestData(self):
        return self.getXTestData , self.getYTestData
