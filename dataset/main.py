import csv
import numpy as np
import os

dest_file = os.getcwd()+'/dataset/Data/breast_cancer.csv'

def read_csv():
    with open(dest_file,'r') as dest_f:
        data_iter = csv.reader(dest_f,
                            delimiter = ',',
                            quotechar = '"')
        data = [data for data in data_iter]
        data_array = np.asarray(data)
        
    return data_array
        
def load_breast_cancer():
    data = read_csv()
    X = []
    Y = []
    for i in range(1,len(data)):
        Y.append(np.int(data[i][-1]))
        X.append(data[i][:-1])
    X = np.array(X)
    Y = np.array(Y)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i,j] = np.float64(X[i,j]).item()
    return X,Y


X , Y = load_breast_cancer()

print(type(X[0,0]))