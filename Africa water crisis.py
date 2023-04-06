#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 23:42:29 2023

@author: harisankar
"""

from pandas_datareader import wb
import pandas as pd
import awoc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pandas.plotting import scatter_matrix

indicator = ['NY.GDP.PCAP.PP.CD', 'SI.POV.GINI', 'SE.XPD.TOTL.GD.ZS', 
             'SE.PRM.CMPT.FE.ZS','SE.PRM.CMPT.MA.ZS', 'EG.ELC.ACCS.ZS',
             'ER.H2O.FWTL.ZS', 'SH.XPD.CHEX.GD.ZS', 'SL.UEM.TOTL.FE.ZS',
             'SL.UEM.TOTL.MA.ZS']

df_wb_raw = wb.download(country='all', indicator=indicator, start=2011, end=2022)

df_wb = df_wb_raw.reset_index() # dataframe to work on
df_wb = df_wb.rename(columns={'country':'Country', 'year':'Year', 
                              'NY.GDP.PCAP.PP.CD': 'Per Capita GDP', 
                              'SI.POV.GINI': 'GINI Index',
                              'SE.XPD.TOTL.GD.ZS':'% GDP on Edu',
                              'SE.PRM.CMPT.FE.ZS':'% Fem Pri Edu',
                              'SE.PRM.CMPT.MA.ZS': '% Male Pri Edu',
                              'EG.ELC.ACCS.ZS': 'Access to Electricity (% of pop)',
                              'ER.H2O.FWTL.ZS': '% Annual Freshwater Withdrawals (internal)',
                              'SH.XPD.CHEX.GD.ZS': '% of GDP on Health',
                              'SL.UEM.TOTL.FE.ZS': '% Unemp Fem',
                              'SL.UEM.TOTL.MA.ZS': '% Unemp Male'})

my_world = awoc.AWOC() # Creating the class
countries_africa= my_world.get_countries_list_of('Africa')
df_country = pd.DataFrame (countries_africa, columns = ['Country'])
df_country = df_country.assign(Continent = 'Africa')

# Checking countries with different spellings 
#df_temp = df_wb.merge(df_country,on='Country', how='inner')
#df_temp['Country'].nunique()

#temp_country = df_temp['Country'].unique().tolist()
#missing = list(set(countries_africa) - set(temp_country))
#print(missing)

# renaming countries in df_country based on countries in world bank data
df_country.loc[df_country['Country']=='Egypt'] = 'Egypt, Arab Rep.'
df_country.loc[df_country['Country']=="Ivory Coast"] = "Cote d'Ivoire"
df_country.loc[df_country['Country']=='Republic of the Congo'] = 'Congo, Rep.'
df_country.loc[df_country['Country']=='Democratic Republic of the Congo'] = 'Congo, Dem. Rep.'
df_country.loc[df_country['Country']=='Gambia'] = 'Gambia, The'

df_wb = df_wb.merge(df_country,on='Country', how='inner') # merging columns
df_wb['Country'].nunique() # number of countries


# Exploring data
#df_wb.head
#list(df_wb.columns)



# Basic histogram
plt.hist(df_wb['% Annual Freshwater Withdrawals (internal)'], bins=10)
plt.title('Annual Freshwater Withdrawals in sub-Saharan Africa')
plt.xlabel('% Annual Freshwater Withdrawals (internal)')
plt.ylabel('Frequency')


# Basic boxplot across years year
df_bp = df_wb[df_wb['% Annual Freshwater Withdrawals (internal)'].notnull()]
plt.boxplot(df_bp['% Annual Freshwater Withdrawals (internal)'].groupby(df_bp['Year']).apply(list), labels=df_bp['Year'].unique())
plt.title('Quartiles of Annual Freshwater Withdrawals')
plt.xlabel('Year')
plt.ylabel('Freshwater Withdrawals (% of total renewable water resources)')


# Boxplot by country
df_wb_water = df_wb[df_wb['% Annual Freshwater Withdrawals (internal)'].notna()]
grouped_data = df_wb_water.groupby('Country')['% Annual Freshwater Withdrawals (internal)'].apply(list)
plt.boxplot(grouped_data.values)
plt.title('Quartiles of Annual Freshwater Withdrawals')
plt.xlabel('Country')
plt.ylabel('Freshwater Withdrawals (% of total renewable water resources)')


# Average % withdrawal for all years across countries
df_drop = df_wb.dropna(subset=['% Annual Freshwater Withdrawals (internal)'])
grouped_data = df_drop.groupby('Country')['% Annual Freshwater Withdrawals (internal)'].mean()
grouped_data = grouped_data.sort_values(ascending=False) #sort in descending order
plt.bar(grouped_data.index, grouped_data.values)
plt.title('Mean Freshwater Withdrawal by Country')
plt.xlabel('Country')
plt.ylabel('Freshwater Withdrawal (billion cubic meters)')
plt.xticks(rotation=90)
plt.show()


# Plot quartiles 
df_drop = df_wb.dropna(subset=['% Annual Freshwater Withdrawals (internal)'])

#create a new column with quartile information
df_drop['Quartile'] = pd.qcut(df_drop['% Annual Freshwater Withdrawals (internal)'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

#group the data by country and quartile
grouped_data = df_drop.groupby(['Country', 'Quartile'])['% Annual Freshwater Withdrawals (internal)'].mean().reset_index()

# Create subplots for each quartile
g = sns.FacetGrid(grouped_data, col='Quartile', col_wrap=2, height=4)
g = g.map(sns.barplot, 'Country', '% Annual Freshwater Withdrawals (internal)')
g.set_xticklabels(rotation=90)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Mean Freshwater Withdrawal by Country Quartile')



# Plot quartiles separately 
df_drop = df_wb.dropna(subset=['% Annual Freshwater Withdrawals (internal)'])

#create a new column with quartile information
df_drop = df_drop.sort_values('% Annual Freshwater Withdrawals (internal)')
df_drop['Quartile'] = pd.qcut(df_drop['% Annual Freshwater Withdrawals (internal)'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

#group the data by country and quartile
grouped_data = df_drop.groupby(['Country', 'Quartile'])['% Annual Freshwater Withdrawals (internal)'].mean().reset_index()
grouped_data.dropna(subset=['% Annual Freshwater Withdrawals (internal)'], inplace=True)
grouped_data = grouped_data.sort_values('% Annual Freshwater Withdrawals (internal)')


quartiles = ['Q1', 'Q2', 'Q3', 'Q4']
for quartile in quartiles:
    subset = grouped_data[grouped_data['Quartile'] == quartile]
    plt.figure()
    ax = sns.barplot(x='Country', y='% Annual Freshwater Withdrawals (internal)', data=subset)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title('Mean Freshwater Withdrawal by Country - ' + quartile)
    ax.set_xlabel('Country')
    ax.set_ylabel('% Annual Freshwater Withdrawals (internal)')
    
    
# Drop outliers

#calculate the lower and upper bounds for outliers using the IQR method
Q1 = df_wb['% Annual Freshwater Withdrawals (internal)'].quantile(0.25)
Q3 = df_wb['% Annual Freshwater Withdrawals (internal)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_wb.loc[(df_wb['% Annual Freshwater Withdrawals (internal)'] < lower_bound) | 
                      (df_wb['% Annual Freshwater Withdrawals (internal)'] > upper_bound)]
df_reg = df_wb.drop(outliers.index)
df_reg = df_reg = df_reg.dropna()


# Regressions
X = df_reg[['Per Capita GDP', '% GDP on Edu', '% of GDP on Health']]
X = sm.add_constant(X)
y = df_reg['% Annual Freshwater Withdrawals (internal)']
model = sm.OLS(y, X).fit()
print(model.summary())


#plot the predicted values against the actual values
fig, ax = plt.subplots()
ax.scatter(y, model.predict(), edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Annual Freeshwater Withdrawals vs. Predicted Annual Freshwater Withdrawals')
plt.show()


#lasso
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = df_reg[['% Annual Freshwater Withdrawals (internal)', '% GDP on Edu', '% of GDP on Health']]
y = df_reg['Per Capita GDP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train the model
lasso = Lasso(alpha=0.1)  # Adjust alpha as needed
lasso.fit(X_train, y_train)

# Evaluate the model
y_pred = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: %.2f" % mse)










    