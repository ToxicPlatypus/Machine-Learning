# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 21:13:56 2021

@author: rafid
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])

plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')