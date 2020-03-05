from LinearRegression.LinearReg import LinearReg
from dataset.main import load_breast_cancer

X , Y = load_breast_cancer()

linreg = LinearReg()
linreg.run(X,Y)
linreg.ErrorHistoryPlot()


# LogReg = LogisticReg()
# LogReg.run(X,Y)
# LogReg.plot()