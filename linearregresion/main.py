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

"""
#scaling expirience years
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

#fitting simple linear regresion to the training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#predicting the test set result
Y_pred = regressor.predict(X_test)

# Visualisation trrain
plt.scatter(X_train, Y_train,color = 'blue')
plt.plot(X_train,regressor.predict(X_train),color = 'red')
plt.title('Salary(Expirience) (Training set)')
plt.xlabel('Years of expirience')
plt.ylabel('Salary')
plt.show()

# visualisation test set

plt.scatter(X_test, Y_test,color = 'blue')
plt.plot(X_train,regressor.predict(X_train),color = 'red')
plt.title('Salary(Expirience) (Test set)')
plt.xlabel('Years of expirience')
plt.ylabel('Salary')
plt.show()
