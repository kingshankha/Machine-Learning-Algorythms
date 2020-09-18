# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df=pd.read_csv("Social_Network_Ads.csv")
X_data=df.iloc[:,:-1].values
Y_data=df.iloc[:,-1].values
print(X_data)
print(Y_data)

## NO Missing Values to be handled

## NO data Formating is Required
## variable data types are correct

##Training and Testing  Split

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X_data,Y_data,test_size=0.2,random_state=0)

## WE will visualize the datato avoid amy bias in model which will generate Error
##if the data is normaly distributed 
## for any variable we will do normalization otherwize we will do
## Standerization
## For classification model the target variable is catagorical 
## So Stnd of Y_data is unnnesesary
df.hist("Age")
df.hist("EstimatedSalary")

## Age quatitetive veriable data is Normaly distributed 

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

SC=StandardScaler()
X_train=SC.fit_transform(X_train)
X_test=SC.transform(X_test)

## Standerization of the data points after  Spliting  eliminates
## imfromation leakage  because test data is the out of sample data
## and scaler cannot be fitted with that data which will create
## Overfitting for the Model


##################################################################

## TRAINING  THE KNN MODEL WITH TRAINING SET 

from sklearn.neighbors import KNeighborsClassifier

Classifier=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)

Classifier.fit(X_train,Y_train)

Y_pred=Classifier.predict((X_test))

res=(np.concatenate((Y_test.reshape(len(Y_test),1),Y_pred.reshape(len(Y_pred),1)),1))


## MAKING CONFUSION MATRIX TO TEST THE ACCURACY OF OUR MODEL

from sklearn.metrics import confusion_matrix, accuracy_score

CM=confusion_matrix(Y_test,Y_pred)

score=accuracy_score(Y_test, Y_pred)

print("confusion matix = ",CM,"\n accuracy = ",score)

###########################################################

## VISUALIZING THE TEST RESULT

