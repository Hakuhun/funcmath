#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras import Sequential
from keras.layers import Dense

dataPath = "C:/DEV/hadoop/normalized_data.csv"

dataset = pd.read_csv(dataPath, sep=';', header=0)
dataset = dataset.drop(columns=['TripID'])
dataset['RouteID'] = dataset['RouteID'].apply(lambda row: pd.to_numeric(row.split('_')[1]))
dataset.head(2)
dataset['result'].plot(kind='bar')


# In[3]:


#check the data distribution
sns.pairplot(dataset)


# In[4]:


#creating training set
X=dataset.iloc[:,0:6]
y=dataset.iloc[:,6:7].values


# In[5]:


#normalizing the values
from sklearn.preprocessing import  MinMaxScaler
sc= MinMaxScaler()
x= sc.fit_transform(X)
#y= y.reshape(-1,1)
#y=sc.fit_transform(y)


# In[6]:


#dividing into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[7]:


#creating the model 
def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(1, input_dim=6, activation='linear'))
    #regressor.add(Dense(units=10, input_shape=(7,0))
    #regressor.add(Dense(units=1))
    #regressor.add(Dense(units=7))
    optimizer = keras.optimizers.adam(amsgrad = True)
    regressor.compile(optimizer= optimizer, loss='mse',  metrics=['mae','accuracy'])
    return regressor


from keras.wrappers.scikit_learn import KerasRegressor
regressor = KerasRegressor(build_fn=build_regressor, batch_size=2,epochs=100)
print(y_train.shape)
print(X_train.shape)

results=regressor.fit(X_train,y_train)

y_pred= regressor.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

