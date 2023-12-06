##     NAIVE BAYES      ##

# Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Load and print the dataset
data_set= pd.read_csv('car_data.csv')
df=pd.DataFrame(data_set)
print("Actual Dataset")
print(df.to_string())
df.info()
df.describe()
# Checking for null values
df.isnull().sum()
# Checking for duplicate values
df.duplicated().sum()
# Extracting the independent and dependent variables
x= data_set.iloc[:,[2,3]].values
y= data_set.iloc[:, 4].values
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
# Initialize Gaussian Naive Bayes classifier
# Fitting the classifier for the training data
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
# Make predictions on the test set
y_pred=classifier.predict(x_test)
df2=pd.DataFrame(x_test)
print(y_pred)
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df2.to_string())
# Calculating the accuracy
from sklearn.metrics import accuracy_score,classification_report
print("Accuracy:",accuracy_score(y_test, y_pred)*100)
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)