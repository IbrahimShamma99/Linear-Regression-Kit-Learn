import numpy as np

class Cleaner():
    #Super is for changing super class behavior
    def __init(self):
        self.n_samples=0

    def splitProperty(self,prop,ratio):
        division = np.int(np.round(self.n_samples * ratio))     
        indices = np.random.permutation(prop.shape[0])
        training_idx , test_idx = indices[:division], indices[division:]
        if (len(prop.shape)>1):
            prop_training, prop_test = prop[training_idx,:], prop[test_idx,:]            
        else:
            prop_training, prop_test = prop[test_idx], prop[training_idx]
        return prop_training , prop_test
        
    def Split(self,X , Y , train_ratio=0.7):
        test_ratio =1-train_ratio
        #NOTE Train 
        X_train , X_test = self.splitProperty(X,train_ratio)
        #NOTE Tests
        Y_train , Y_test = self.splitProperty(Y,test_ratio) 
        # print("X_train "+str(X_train.shape))       
        # print("Y_train "+str(Y_train.shape))       
        # print("X_test "+str(X_test.shape))       
        # print("Y_test "+str(Y_test.shape))   
        return X_train , Y_train , X_test , Y_test