# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 12:08:48 2025

@author: gopal.ghule
"""
import pandas as pd


df = pd.read_csv('StudentsPerformance.csv')

#print(df.head())

print(df.info())
#print(df.describe())

#print(df.isnull().sum())

df['average_score'] = df[['math score', 'reading score',  'writing score']].mean(axis=1)
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df['average_score'], kde=True)
plt.title("Distribution of Average Scores")
plt.xlabel("Average Exam Score")
plt.ylabel("Number of Students")
plt.show()

df_encoded = pd.get_dummies(df,columns=['gender', 
        'race/ethnicity', 
        'parental level of education', 
        'lunch', 
        'test preparation course'],drop_first=True)
print(df_encoded)
df_encoded.head()
df_encoded.shape


# 1. Define the target (label/output) — what we're predicting
y = df_encoded['average_score']
# 2. Define the features (input variables) — what the model will use to make predictions
X = df_encoded.drop(columns=['math score', 'reading score', 'writing score', 'average_score'])
print('-------------------------------------------')
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


print("Training data size:", X_train.shape, y_train.shape)
print("Testing data size:", X_test.shape, y_test.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

print(model)

model.fit(X_train, y_train)



# Intercept (β₀)
print("Intercept (β₀):", model.intercept_)

# Coefficients (β₁, β₂, ..., βₙ)
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
    
    
    
    
y_pred = model.predict(X_test)

print(y_pred)


from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)


r2 = r2_score(y_test, y_pred)


print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")



import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Average Score")
plt.ylabel("Predicted Average Score")
plt.title("Actual vs Predicted Scores")
plt.grid(True)
plt.plot([0, 100], [0, 100], color='red', linestyle='--')  # perfect prediction line
plt.show()



