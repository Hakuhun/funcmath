import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib as plt

dataPath = "C:/DEV/hadoop/clean_data.csv"

dataset = pd.read_csv(dataPath, sep=';', header=0)

# check the data distribution
sns.pairplot(dataset)

# creating training set
x = dataset.to_numpy()[:, 0:6]
y = dataset.to_numpy()[:, 6]
y = np.reshape(y, (-1, 1))

# normalizing the values

scaler = MinMaxScaler()
print(scaler.fit(x))
print(scaler.fit(y))

# dividing into train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y)

# Import the model we are using
from sklearn.tree import DecisionTreeRegressor
# Instantiate model with 1000 decision trees
regressor = DecisionTreeRegressor(random_state = 0)

# fit the regressor with X and Y data
regressor.fit(x, y)

# import export_graphviz
from sklearn.tree import export_graphviz

# export the decision tree to a tree.dot file
# for visualizing the plot easily anywhere
export_graphviz(regressor, out_file='tree.dot',
                feature_names=['CurrentTime', 'RouteID', 'Temperature', 'Wind', 'Rain', 'Snow'])