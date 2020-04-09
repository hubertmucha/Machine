import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the data set
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values 
Y = dataset.iloc[:,4].values 
#categorical category
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 

ct = ColumnTransformer([("State", OneHotEncoder(),[3])], remainder="passthrough")
X = ct.fit_transform(X)

#Avoiding dummy variable trap

X = X[:,1:]


#spliting the data sets 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

"""
#scaling expirience years
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred  = regressor.predict(X_test)


import statsmodels.regression.linear_model as sm

X = np.append(arr = np.ones((50,1)).astype(int),values = X ,axis = 1)

X_opt = X[:,[0,3,5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(Y, X_opt).fit()

stat_table = regressor_OLS.summary()

print(stat_table)