# exp-1:Implementation of MP model

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Load dataset

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Binarize the features
X_bin = X.apply(pd.cut, bins=2, labels=[0, 1]).astype(int)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_bin, y, test_size=0.2, random_state=1, stratify=y)

# Define the MP Neuron model
class MPNeuron:

 def __init__(self):
  self.b = None

 def model(self, x):
  return int(np.sum(x) >= self.b)

 def predict(self, X):
  return np.array([self.model(x) for x in X])

 def fit(self, X, y):
  accuracy = {}

  for b in range(X.shape[1] + 1):
   self.b = b
   y_pred = self.predict(X)
   acc = accuracy_score(y, y_pred)
   accuracy[b] = acc
  self.b = max(accuracy, key=accuracy.get)
  print(f"Optimal Threshold (b): {self.b}")
  print(f"Training Accuracy: {accuracy[self.b]}")

# Train and evaluate the MP Neuron model

mp = MPNeuron()
mp.fit(X_train.values, y_train)
y_pred = mp.predict(X_test.values)
accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", accuracy)
