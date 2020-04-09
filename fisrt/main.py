import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the data set
dataSet = pd.read_csv('Data.csv')
X = dataSet.iloc[:,:-1].values
Y = dataSet.iloc[:,-1:].values

#taking care of missing data
from sklearn.impute import SimpleImputer

imputerObj = SimpleImputer()
imputerObj  = imputerObj .fit(X[:,1:3])
X[:,1:3] = imputerObj .transform(X[:,1:3])

#categorical category
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer 
ct = ColumnTransformer([("Country", OneHotEncoder(),[0])], remainder="passthrough")
X = ct.fit_transform(X)

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#spliting the data sets 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

#scaling data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
