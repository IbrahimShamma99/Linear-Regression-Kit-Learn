from LinearRegression.LinearReg import LinearReg
from sklearn.datasets import load_boston
from LogisticRegression.LogisticReg import LogisticReg
import numpy as np

dataset = load_boston()
X = dataset.data
Y = dataset.target[:,np.newaxis]

linreg = LinearReg()
linreg.run(X,Y)
linreg.plotError()

LogReg = LogisticReg()
LogReg.run(X , Y)
LogReg.plotError()