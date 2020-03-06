from Algorithms.LinearReg import LinearReg
from dataset.main import load_breast_cancer

X , Y = load_breast_cancer()

model = LinearReg()
# print(linreg.getCostHistory())
model.run(X,Y, train_ratio=0.7)
model.ErrorHistoryPlot()
