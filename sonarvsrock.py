# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

## Data Preprocessing part starts from here

# Importing the dataset to pandas dataframe
data = pd.read_csv("Copy of sonar data.csv", header=None)
data.describe()    # describe statistical values of data
data[60].value_counts()
data.groupby(60).mean()

# Separate the data and label
X = data.drop(columns=60, axis=1)
Y = data[60]
print(X)
print(Y)

# Splitting the training and test dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.1,
                                                    stratify = Y, random_state=1)
print(X.shape, X_train.shape, Y_test.shape)

## Applying logistic Regression model to our training dataset
model = LogisticRegression()
model.fit(X_train,Y_train)

# Model Evaluation
# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy of training data", training_data_accuracy)

# Accuracy on testing data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy of training data", testing_data_accuracy )






