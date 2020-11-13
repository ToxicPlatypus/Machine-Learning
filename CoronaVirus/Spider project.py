# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:16:21 2020

@author: rafid
"""

import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt
import matplotlib
CVD = pd.read_csv(r'C:\Users\rafid\OneDrive\Desktop\ML Project\full_data.csv', encoding='latin-1')
print(CVD.dtypes)
CVD['date'] = [dt.datetime.strptime(x,'%m/%d/%Y') for x in CVD['date']]
print(CVD.dtypes)
countries = ['Bangladesh']
CVD_country = CVD[CVD.location.isin(countries)]
CVD_country.set_index('date', inplace = True)
print(CVD_country.head())
CVD_country = CVD_country.copy()
CVD_country['mortality_rate'] = CVD_country['total_deaths']/CVD_country['total_cases']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14,14))

CVD_country.groupby('location')['new_cases'].plot(ax=axes[0,0], legend=True) #for log scale add logy=True
CVD_country.groupby('location')['new_deaths'].plot(ax=axes[0,1], legend=True)
CVD_country.groupby('location')['total_cases'].plot(ax=axes[1,0], legend=True)
CVD_country.groupby('location')['total_deaths'].plot(ax=axes[1,1], legend=True)
#CVD_country.groupby('location')['mortality_rate'].plot(ax=axes[1,1], legend=True)
#CVD_country.to_csv('data/output.csv')

axes[0, 0].set_title("New Cases")
axes[0, 1].set_title("New Deaths")
axes[1, 0].set_title("Total Cases")
axes[1, 1].set_title("Total Deaths")