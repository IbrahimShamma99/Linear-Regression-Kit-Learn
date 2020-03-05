from LinearRegression.LinearReg import LinearReg
from dataset.main import load_breast_cancer

X , Y = load_breast_cancer()

linreg = LinearReg()
linreg.run(X,Y, train_ratio=0.7)
linreg.ErrorHistoryPlots()
