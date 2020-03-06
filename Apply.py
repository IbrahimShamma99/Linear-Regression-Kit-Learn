from Algorithms.LinearReg import LinearReg
from dataset.main import load_breast_cancer

X , Y = load_breast_cancer()

linreg = LinearReg()
# print(linreg.getCostHistory())
linreg.run(X,Y, train_ratio=0.7)
linreg.ErrorHistoryPlot()
