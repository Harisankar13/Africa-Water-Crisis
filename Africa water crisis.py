#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 23:42:29 2023

@author: harisankar
"""


from pandas_datareader import wb
import pandas as pd
import awoc


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


