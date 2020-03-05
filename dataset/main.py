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

def preprocess(data):
    X = []
    Y = []
    for i in range(1,len(data)):
        Y.append(data[i][-1])
        X.append(data[i][:-1])
    X = np.array(X)
    Y = np.array(Y)
    X = X.astype(np.float)
    Y = Y.astype(np.int)
    return (X , Y) 
        
def load_breast_cancer():
    data = read_csv()
    return preprocess(data)