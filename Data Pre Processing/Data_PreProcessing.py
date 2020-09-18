# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
print(X)

y = dataset.iloc[:, -1].values
df=pd.read_csv("Data.csv")
df.head()

#Handling Missing Values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
print(X)

#converting categorical Veriable To Quantitetive Veeriable
#encoding categorical data #independent variable / feature set
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CT=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])],remainder="passthrough")
temp=CT.fit_transform(X)
X=np.array(temp)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train)

print(X_test)

#Standerdisation and Normalization
from sklearn.preprocessing import StandardScaler

#we will fit the standerd scaler object with Xtrain set because it can create information leakage if we 
#Normelize  the data befoe scaling ; after fitting the scaler object we will use that object instant to transform
# Test set or in future for predicting the target lable out of a feature set
#we use Normelization when the data is Normally Distributed
#otherwise Standerization

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)


# we will exclude the encoded catagorical variable coloumns

