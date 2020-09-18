# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df=pd.read_csv("Position_Salaries.csv")
X_data=df.iloc[:,1:-1].values
Y_data=df.iloc[:,-1].values
#Handling Missing Values

#dataset has no missing values

#converting categorical Veriable To Quantitetive Veeriable
#encoding categorical data #independent variable / feature set

## Level veriable is the quantetetive  description of
# Position variable So Position coloumn is not relevent for 
#our model

# Splitting the dataset into the Training set and Test set

### AS we have a very small dataSet sowe exeptionaly want to 
#build the model on the whole dataset
 
#######
##Polynominal Regression 
#######

####
##Training Linear Regression Model with the whole dataset
####

from sklearn.linear_model import LinearRegression
Linear_reg=LinearRegression()
Linear_reg.fit(X_data,Y_data)

#####
##Training The  Polynominal Regression with whole dataset
###
## transforming the feature variables into higher order polynominal 
#variable  then fitting into Linear regression
## Degree= NO of predictor variables
## Order = maximum exponent amongst the polynominal feature veriables
##here degree parameter is order of the regressor

from sklearn.preprocessing import PolynomialFeatures
PolyTrans=PolynomialFeatures(degree=2)
X_poly=PolyTrans.fit_transform(X_data)
Poly_Linear_reg=LinearRegression()
Poly_Linear_reg.fit(X_poly,Y_data)


#####
##Visualizing The Linear Models 
#####
plt.scatter(X_data,Y_data,color="b")
plt.plot(X_data,Linear_reg.predict(X_data),color="g")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Linear Model FIT")
plt.show()
## conclusion==The Linear Model does not fits the data set So 
#Polynominal Linear Regression  aprroch should be adpted and evaluated

plt

######
##VIsualizing The Polynominal Model curve fitting
######
 
plt.scatter(X_data,Y_data,color="r")
plt.plot(X_data,Poly_Linear_reg.predict(PolyTrans.fit_transform(X_data)),color="c")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Polynominal Regression Model FIT")
plt.show()

