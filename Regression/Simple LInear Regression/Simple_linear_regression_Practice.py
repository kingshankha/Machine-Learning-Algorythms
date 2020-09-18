# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 21:17:45 2020

@author: Shankha
"""

# =============================================================================
# Simple Linear Regression Practice
# =============================================================================
##  Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Importing the dataset
df=pd.read_csv("Salary_Data.csv")
X_data=df.iloc[:,:1]
Y_data=df.iloc[:,1:2]


##Data Preprocessing

##checking for missing data 
print(X_data.count(),X_data.shape)
print(df["Salary"].count())
## No missing Values found
## No categorical veriable present
## Checking for correct data formating 

print(X_data.dtypes,"\n" )
print(Y_data.dtypes)


## converting feature veriables and the target veriables to numpy array

X_data=df.iloc[:,:1].values
Y_data=df.iloc[:,1:2].values

## Spliting data set into train and Test set

## Model Development

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_data,Y_data, test_size = 0.2, random_state = 0)

###Model Development

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
##predicted value using our LR model
Y_hat_train=regressor.predict(X_train)
Y_hat=regressor.predict(X_test)

print((Y_train,Y_hat_train),(Y_test,Y_hat))

## temporary prediction check of 1.5 2.2 and 4.1 years of experience

temp=np.array([1,2,3]).reshape(-1,1)
print(regressor.predict(temp))

### visualizing the train Set
plt.scatter(X_train,Y_train,color="r")
plt.plot(X_train,Y_hat_train,color="b")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.title("Years Vs Salary ,Train Set ")
plt.show()

### visualizing the test Set

plt.scatter(X_test,Y_test,color="r")
plt.plot(X_test,Y_hat,color="b")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.title("Years Vs Salary Test Set ")

## ploting using only seaborn
sns.regplot(X_data,Y_data,data=df)


