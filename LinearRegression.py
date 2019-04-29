#!/usr/bin/env python
# coding: utf-8

# In[4]:


#import
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras import Sequential
from keras.layers import Dense

dataPath = "C:/DEV/hadoop/normalized_data.csv"


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
dataset = pd.read_csv(dataPath, sep=';', header=0)
dataset.head(2)


# In[6]:


#check the data distribution
sns.pairplot(dataset)


# In[7]:


#creating training set
X=dataset.iloc[:,3:9]
y=dataset.iloc[:,9].values


# In[ ]:


#normalizing the values
from sklearn.preprocessing import  MinMaxScaler
sc= MinMaxScaler()
x= sc.fit_transform(X)
y= y.reshape(-1,1)
y=sc.fit_transform(y)


# In[ ]:


#dividing into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[ ]:


#creating the model 
def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units=7, input_dim=6))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae','accuracy'])
    return regressor


# In[ ]:


from keras.wrappers.scikit_learn import KerasRegressor
regressor = KerasRegressor(build_fn=build_regressor, batch_size=32,epochs=100)


# In[ ]:


results=regressor.fit(X_train,y_train)


# In[ ]:


y_pred= regressor.predict(X_test)


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# In[ ]:




