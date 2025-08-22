# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages.
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.


## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: INDHUMATHI L
RegisterNumber:  212224220037

# IMPORT REQUIRED PACKAGES
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from google.colab import files
uploaded = files.upload()
from google.colab import files
uploaded = files.upload()
dataset = pd.read_csv("student_scores.csv")

print("First 5 rows of dataset:\n", dataset.head())
print("\nLast 5 rows of dataset:\n", dataset.tail())

# SPLIT INTO FEATURES AND TARGET
x = dataset.iloc[:, :-1].values   # Hours
y = dataset.iloc[:, 1].values     # Scores
print("\nX values:\n", x)
print("\nY values:\n", y)

# TRAIN-TEST SPLIT
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=1/3, random_state=0
)

# LINEAR REGRESSION MODEL
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

# PREDICTIONS
y_pred = reg.predict(x_test)
print("\nPredicted values:", y_pred)
print("Actual values:", y_test)

# GRAPH: TRAINING SET
plt.scatter(x_train, y_train, color='purple')
plt.plot(x_train, reg.predict(x_train), color='black')
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# GRAPH: TESTING SET
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, reg.predict(x_train), color='black')
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# ERRORS
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nMean Squared Error =", mse)
print("Mean Absolute Error =", mae)
print("Root Mean Square Error =", rmse)

```

## Output:

To Read Head and Tail Files

<img width="772" height="411" alt="image" src="https://github.com/user-attachments/assets/866cb639-2e54-4d61-be44-781ece419de9" />

Compare dataset

<img width="929" height="661" alt="image" src="https://github.com/user-attachments/assets/39f1467d-b808-454d-8146-0b024d0f4e80" />

Predicted Values and Actual values

<img width="913" height="71" alt="image" src="https://github.com/user-attachments/assets/3d62e731-12fc-43ee-9680-37d8f8e31f31" />

Graph for training set 

<img width="698" height="561" alt="image" src="https://github.com/user-attachments/assets/d7029187-d535-46a7-bd86-2b2ac8f9dee6" />


Graph for testing set

<img width="689" height="568" alt="image" src="https://github.com/user-attachments/assets/4f1d2716-8d4a-4499-8e25-09aeca3acb25" />

Error

<img width="673" height="83" alt="image" src="https://github.com/user-attachments/assets/662711c8-bb9c-43ec-85f7-0f0cefab368e" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
