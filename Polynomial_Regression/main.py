import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the data set
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values #matrix
y = dataset.iloc[:,2].values  #vector


"""
#spliting the data sets 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

#scaling data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
#fiting linear regression
from sklearn.linear_model import LinearRegression

linReg = LinearRegression()
linReg.fit(X,y)

#fiting polinomial regression
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 5)
X_poly = polyReg.fit_transform(X)

linReg2 = LinearRegression()
linReg2.fit(X_poly,y)

#visualisation linear regression
#plt.scatter(X,y, color='blue')
plt.plot(X,linReg.predict(X),color='green')
plt.title('Linear Regresion')
plt.show()

#visualisation polynomial regression
X_grid = np.arange(min(X),max(X),0.1 )
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color='blue')
plt.plot(X_grid,linReg2.predict(polyReg.fit_transform(X_grid)),color='red')
plt.title('Polynomial Regresion')
plt.show()

#prediction for linear regression
linReg.predict([[6.5]]) 

#prediction for polymial regression
linReg2.predict(polyReg.fit_transform([[6.5]]))
