import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the data set
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values #matrix
y = dataset.iloc[:,2].values  #vector



#fiting polinomial regression
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 5)
X_poly = polyReg.fit_transform(X)

linReg2 = LinearRegression()
linReg2.fit(X_poly,y)


#visualisation polynomial regression
X_grid = np.arange(min(X),max(X),0.1 )
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color='blue')
plt.plot(X_grid,linReg2.predict(polyReg.fit_transform(X_grid)),color='red')
plt.show()


#prediction for polymial regression
linReg2.predict(polyReg.fit_transform([[6.5]]))
