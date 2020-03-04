from LinearRegression.LinearReg import LinearReg
from dataset.main import load_breast_cancer
from sklearn.datasets import load_boston

# data = load_boston()

# X = data.data
# Y = data.target

X , Y = load_breast_cancer()

print(type(X[0,0]))

linreg = LinearReg()
linreg.run(X,Y)
linreg.plot()


# LogReg = LogisticReg()
# LogReg.run(X,Y)
# LogReg.plot()