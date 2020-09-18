# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df=pd.read_csv("50_Startups.csv")
X_data=df.iloc[:,:4].values
Y_data=df.iloc[:,-1:].values
print(X_data)
print(Y_data)
#Handling Missing Values
print("there is no missing values in the feature or the target set")
#converting categorical Veriable To Quantitetive Veeriable
#encoding categorical data #independent variable / feature set
print("the state variable is catagorical so we have to encode the variable using one hot")
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[("Encoder",OneHotEncoder(),[3])],remainder="passthrough")

X_data=ct.fit_transform(X_data)
print(X_data)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_data,Y_data,test_size=0.2,random_state=0)
#Standerdisation and Normalization
#for multiple Linear Regression feature Scaling is Not Reqired because
#the cofficient will compensate for that

####################
# MULTIPLE LINEAR REGRESSION MODEL 
####################
 
from sklearn.linear_model import  LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
coeffs=regressor.coef_
intercept = regressor.intercept_
print(coeffs,intercept)

Y_pred=regressor.predict(X_test)

## printing NUMPY ARRAY values upto two decimal places

np.set_printoptions(precision=2)

##Printing predicted result and test set in a concatinated form
#print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))
print(np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.reshape(len(Y_test),1)),1))


###
## Question 1: How do I use my multiple linear regression model to make a single prediction,#
# for example, the profit of a startup with R&D Spend = 160000, Administration Spend = 130000,
# Marketing Spend = 300000 and State = California?
###
print("Profit of the Startup== {} ".format(regressor.predict(ct.transform([[160000,130000,300000,"California"]]))[0]))


