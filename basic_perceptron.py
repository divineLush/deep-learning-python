import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

iris = load_iris()

# 3 different types of irises: Setosa, Versicolour, Virginica
# columns: sepal length, sepal width, petal length, petal width

# 2-dim array with petal length and width
x = iris.data[:, (2, 3)]

# we wanna know if the flower is Setosa
# iris.target consists of 1, 2, 3 each corresponding to flower type
# gotta convert iris.target so that
# all Setosa flowers are 1 and others are 0
y = (iris.target == 0).astype(int)

# create perceptron classifier using sklearn
per_clf = Perceptron(random_state=42)
per_clf.fit(x, y)

y_pred = per_clf.predict(x)

# 1.0 - all predictions are right
print(accuracy_score(y, y_pred))

