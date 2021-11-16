import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# import pydot
import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# plt.imshow(X_train_full[1])
# plt.show()

y_train_full[1]
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# print(class_names[y_train_full[10]])

# normalize data
# pixel densities lies in [0, 255]
# we want the output in form of floating numbers between 0 and 1
# so we divide by 255.0, not integer 255
X_train_n = X_train_full / 255.0
X_test_n = X_test / 255.0

# split data into train/validation/test datasets
X_valid, X_train = X_train_n[:5000], X_train_n[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test_n

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.summary()

# keras.utils.plot_model(model)
weights, biases = model.layers[1].get_weights()
print(weights.shape)
print(biases.shape)

# sgd = stochastic gradient descent
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

model_history = model.fit(X_train, y_train, epochs=2, validation_data=(X_valid, y_valid))

pd.DataFrame(model_history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# evaluating performance
model.evaluate(X_test, y_test)

# treat first 3 samples from dataset as new unseen data
X_new = X_test[:3]
y_prob = model.predict(X_new)
y_prob.round(2)

# predict probabilities
y_predict = model.predict(X_new)
print("predict!", "\n")
print(y_predict)

# list 3 predictions
# print(np.array(class_names)[y_predict])

plt.imshow(X_test[0])
plt.show()

plt.imshow(X_test[1])
plt.show()

plt.imshow(X_test[2])
plt.show()
