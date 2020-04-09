import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the data set
dataset = pd.read_csv('Salary_Data.csv',sep=',')

X = dataset.iloc[:,:-1].values #expirience
Y = dataset.iloc[:,-1:].values #salary



#spliting the data sets 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 1/3,random_state = 0)

#fitting simple linear regresion to the training set
from sklearn.linear_model import LinearRegression

linRegressor = LinearRegression()
linRegressor.fit(X_train,Y_train)

#predicting the test set result
Y_pred = linRegressor.predict(X_test)

# Visualisation trrain
plt.scatter(X_train, Y_train,color = 'blue')
plt.plot(X_train,linRegressor.predict(X_train),color = 'red')
plt.title('Salary(Expirience) (Training set)')
plt.xlabel('Years of expirience')
plt.ylabel('Salary')
plt.show()

# visualisation test
plt.scatter(X_test, Y_test,color = 'blue')
plt.plot(X_train,linRegressor.predict(X_train),color = 'red')
plt.title('Salary(Expirience) (Test set)')
plt.xlabel('Years of expirience')
plt.ylabel('Salary')
plt.show()
