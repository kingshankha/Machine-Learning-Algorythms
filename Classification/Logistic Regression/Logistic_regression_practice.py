# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df=pd.read_csv("Social_Network_Ads.csv")
df.head()
X_data=df.iloc[:,:2].values
Y_data=df.iloc[:,-1:].values
print(X_data);
print(Y_data);
#Handling Missing Values
from sklearn.impute import SimpleImputer
SI=SimpleImputer(missing_values=np.nan,strategy="mean")
SI.fit_transform(X_data)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X_data,Y_data,test_size=0.25,random_state=None)
print(X_test)
print


#Standerdisation and Normalization
from sklearn.preprocessing import StandardScaler

#we will fit the standerd scaler object with Xtrain set because it can create information leakage if we 
#Normelize  the data befoe scaling ; after fitting the scaler object we will use that object instant to transform
# Test set or in future for predicting the target lable out of a feature set
#we use Normelization when the data is Normally Distributed
#otherwise Standerization

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)
print(X_train)
print(X_test)


#####################

# Building Logistic regression MOdel

#####################

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

######
# Predicting the test result
######
Y_hat_test = classifier.predict(X_test)

print(np.concatenate((Y_hat_test.reshape(len(Y_hat_test),1),Y_test.reshape(len(Y_test),1)),1))


####
## CONFUSION METRIX 
####

##confusion metrix will tel us about the fasle positive
## False Negetive ,True Positive and True negetive
from sklearn.metrics import confusion_matrix
CM=confusion_matrix(Y_test,Y_hat_test)
print(CM)
###
###CALCULATING THE ACCURACY OF OUR MODEL

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,Y_hat_test)

print(accuracy)


##########
# Visualizing the TRaining Set Result
##########


