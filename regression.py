import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor

dataPath = "C:/DEV/hadoop/clean_data.csv"

dataset = pd.read_csv(dataPath, sep=';', header=0)
#dataset = dataset.drop(columns=['CurrentTime', 'RouteID'], axis=0)

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
xscale = scaler.transform(x)
yscale = scaler.transform(y)

# dividing into train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)


def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(12, input_dim=6, kernel_initializer='normal', activation='relu'))
    regressor.add(Dense(8, activation='relu'))
    regressor.add(Dense(1, activation='linear'))

    optimizer = keras.optimizers.adam(amsgrad=True)
    regressor.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'accuracy'])
    return regressor


regressor = KerasRegressor(build_fn=build_regressor, batch_size=20, epochs=100)

print(y_train.shape)
print(X_train.shape)

results = regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
