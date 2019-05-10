import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras
from keras import layers

import matplotlib.pyplot as plt

dataPath = "C:/DEV/hadoop/clean_data.csv"

dataset = pd.read_csv(dataPath, sep=';', header=0)

dataset = dataset.astype(float)
# check the data distribution
sns.pairplot(dataset)

# creating training set
x = dataset.to_numpy()[:, 0:6]
y = dataset.to_numpy()[:, 6]
#y = np.reshape(y, (-1, 1))

# dividing into train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_dim=6),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

model.summary()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  X_train, y_train,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()])

y_pred= regressor.predict(X_test)
