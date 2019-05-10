import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

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
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, np.ravel(y_train))

# Use the forest's predict method on the test data
predictions = rf.predict(X_test)
print(predictions)

print(rf.predict([[19,3040,20, 10, 0,0]]))

