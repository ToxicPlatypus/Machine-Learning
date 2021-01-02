# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 19:29:41 2021

@author: rafid
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.linear_model import  LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#visualizing the linear regression model

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.show()

#visualizing the polynomial regression model

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.show()

#predict

lin_reg_2.predict(poly_reg.fit_transform(6.5))