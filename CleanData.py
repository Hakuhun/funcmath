#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import keras
import math
get_ipython().run_line_magic('matplotlib', 'inline')


pd.set_option('display.html.table_schema', True) # to can see the dataframe/table as a html
pd.set_option('display.precision', 5) 


# In[2]:


rawPath = "C:/DEV/hadoop/data.csv"


# In[3]:


rawData = pd.read_csv(rawPath, sep=';', header=0)
rawData.replace(',','.', regex=True, inplace=True)
print("Beolvasás kész")


# In[4]:


#list(rawData)
cleanData = rawData.drop(columns=['VeichleID', 'VeichleType', 'VeichleModel', 'VeichleLongitude', 'VeichleLatitude', 'StopID', 'StopSequance', 'PredictedDepartureTime', 'DepartureTime', 'TripStatus', 'ArrivalTime', 'PredictedArrivalTime'])
print("Feleseges cellák törlve")
#cleanData


# In[5]:


cleanData['CurrentTime'] = pd.to_datetime(cleanData['CurrentTime'], unit ='ms')
cleanData['CurrentTime'] = cleanData['CurrentTime'].dt.hour
cleanData['Temperature'] = pd.to_numeric(cleanData['Temperature'])
cleanData['WindIntensity'] = pd.to_numeric(cleanData['WindIntensity'])
cleanData['RainIntensity'] = pd.to_numeric(cleanData['RainIntensity'])
cleanData['SnowIntesity'] = pd.to_numeric(cleanData['SnowIntesity'])
cleanData.head(10)


# In[6]:


aggregatedData = cleanData.groupby(['CurrentTime','RouteID','TripID']).agg({
    'ArrivalDiff':'sum',
    'DepartureDiff':'sum',
    'Temperature': 'mean',
    'WindIntensity': 'mean',
    'RainIntensity': 'mean',
    'SnowIntesity': 'mean',
})


# In[7]:


aggregatedData['result'] = aggregatedData[['ArrivalDiff', 'DepartureDiff']].mean(axis=1)
aggregatedData['result'] = aggregatedData['result'].apply(lambda x: 3600 if x > 3600 else x )
aggregatedData = aggregatedData.drop(columns=['ArrivalDiff', 'DepartureDiff'])
aggregatedData.to_csv("C:/DEV/hadoop/clean_data.csv", sep=';', header = True, float_format='%.15f')
aggregatedData[['result']].head(10)


# In[8]:


normalizeddData = aggregatedData.copy(deep=True)
standardizedData = aggregatedData.copy(deep=True)


# In[9]:


data = normalizeddData[['result']].values
normalizeddData[['result']] = keras.utils.normalize(
    data,
    axis=0,
    order=2
)
normalizeddData.to_csv("C:/DEV/hadoop/normalized_data.csv", sep=';', header = True, float_format='%.15f')
normalizeddData[['result']].head(10)


# In[10]:


scaler = StandardScaler()
scaler = scaler.fit(standardizedData[['result']])

normalized = scaler.transform(standardizedData[['result']])
standardizedData['result'] = normalized
standardizedData[['result']].head(10)


# In[11]:


normalizeddData.to_csv("C:/DEV/hadoop/standardized_data.csv", sep=';', header = True, float_format='%.15f')

