import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from tensorflow.python.util.tf_export import KERAS_API_NAME
housing = fetch_california_housing()

print(housing)

# split data
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

# preprocessing: standardize data
# subtract the mean of each variable from their individual values
# and divide by the variance
# at the end we want all the variables with mean 0 and their variance as 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

np.random.seed(42)
tf.random.set_seed(42)

print(X_train.shape)

# create NN structure
# single neuron in the output layer since it's the regression problem
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])

# SGD = stochastic gradient descent
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3), metrics=["mae"])

print(model.summary())

model_history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

mae_test = model.evaluate(X_test, y_test)

pd.DataFrame(model_history.history).plot(figsize=(8, 5))
plt.grid()
plt.gca().set_ylim(0, 1)
plt.show()

X_new = X_test[:3]
y_pred = model.predict(X_new)
print(y_pred)
print(y_test[:3])
