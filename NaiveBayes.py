#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
get_ipython().run_line_magic('matplotlib', 'inline')


pd.set_option('display.html.table_schema', True) # to can see the dataframe/table as a html
pd.set_option('display.precision', 5) 


# In[2]:


rawPath = "C:/DEV/hadoop/normalized_data.csv"
data = pd.read_csv(rawPath, sep=';', header=0)
print("Beolvasás kész")
data = data.drop('TripID', axis = 1)


# In[3]:


print('Minta')
data.head(10)


# In[4]:


print('ELoszlás ')
print('Dispersion of lates, normalied the maximum to 3600 seconds (1 hour)')
data['result'].plot.kde(bw_method=0.3)


# In[5]:


data['RouteID'] = data['RouteID'].apply(lambda row: pd.to_numeric(row.split('_')[1]))

x = data.iloc[:,0:6]
y = data.iloc[:,6:7]
y['result'] = y['result'].apply(lambda row: 1 if row >= 0.05 else 0)

y


# In[6]:


model = MultinomialNB()

# Train the model using the training sets 
model.fit(x, np.ravel(y.values))

#Predict Output


# In[7]:


prediction = model.predict_proba(data.iloc[0:1, 0:6])
print (prediction)


# In[ ]:




