import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Study_hours': [1,2,3,4,5,6,7,8,9,10],
    'Attandance': [57,45,56,74,67,78,99,97,67,86],
    'class_bunk': [5,6,7,2,3,4,7,8,9,0],
    'Marks': [35,44,55,67,67,57,76,26,98,86]
}

x = df[['Study_hours', 'Attandance', 'class_bunk']]
y = df['Marks']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict (x_test)
y_pred 

mae=mean_squared_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Absolute Error:{mae}")
print("Mean Squared Error:{mse}")
print("R2 Score:{r2}")

new_student=[[7,80,9]]
predict_marks=model.predict(new_student)
print("Predicted Marks:",predict_marks)