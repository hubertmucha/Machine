import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the data set
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values #matrix
y = dataset.iloc[:,2].values  #vector

#scaling data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

#SVR
from sklearn.svm import SVR

regressorSVR = SVR(kernel='rbf')
regressorSVR.fit(X, y)


#y_pred =regressorSVR.predict([[6.5]])
#visualisation SVR regression
plt.scatter(X,y, color='blue')
plt.plot(X,regressorSVR.predict(X),color='green')
plt.title('Linear Regresion')
plt.show()
y_pred = regressorSVR.predict(sc_X.transform(np.array([[6.5]])))
y_pred = sc_y.inverse_transform(y_pred)
