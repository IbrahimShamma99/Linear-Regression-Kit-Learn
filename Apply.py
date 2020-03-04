from LinearRegression.LinearReg import LinearReg
from sklearn.datasets import load_boston , load_breast_cancer
from LogisticRegression.LogisticReg import LogisticReg
import numpy as np

dataset = load_boston()
X = dataset.data
Y = dataset.target[:,np.newaxis]

linreg = LinearReg()
linreg.run(X,Y)
linreg.plot()


# LogReg = LogisticReg()
# LogReg.run(X,Y)
