# week-3

# Implementation of back propagation algorithm using with python

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X = data.data
y = data.target

# One-hot encode labels
y = pd.get_dummies(y).values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Parameters
learning_rate = 0.1
segment_iterations = 5000
segments = 3
N = y_train.shape[0]
input_size = X.shape[1]
hidden_size = 2
output_size = y.shape[1]

# Activation and loss functions

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def mean_squared_error(y_pred, y_true):
  return ((y_pred - y_true) ** 2).sum() / (2 * y_pred.shape[0])

def accuracy(y_pred, y_true):
  return (y_pred.argmax(axis=1) == y_true.argmax(axis=1)).mean()

# For plotting results
all_results = pd.DataFrame(columns=["mse", "accuracy"])

# Fix seed for reproducibility
np.random.seed(10)

# Training loop over multiple segments
for segment in range(segments):
  # Initialize weights
  W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))
  W2 = np.random.normal(scale=0.5, size=(hidden_size, output_size))
  results = pd.DataFrame(columns=["mse", "accuracy"])
  for itr in range(segment_iterations):
    # Feedforward
    Z1 = np.dot(X_train, W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)

    # Compute loss and accuracy
    mse = mean_squared_error(A2, y_train)
    acc = accuracy(A2, y_train)
    results.loc[len(results)] = [mse, acc]
    # Backpropagation

    error_output = A2 - y_train
    delta_output = error_output * A2 * (1 - A2)
    error_hidden = np.dot(delta_output, W2.T)
    delta_hidden = error_hidden * A1 * (1 - A1)
    W2_update = np.dot(A1.T, delta_output) / N
    W1_update = np.dot(X_train.T, delta_hidden) / N

    # Weight update

    W2 -= learning_rate * W2_update
    W1 -= learning_rate * W1_update

  # Adjust index and collect results

  results.index += segment * segment_iterations
  all_results = pd.concat([all_results, results])

# Plotting loss and accuracy

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(all_results["mse"], label="MSE")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(all_results["accuracy"], color='green', label="Accuracy")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# Test evaluation
Z1 = np.dot(X_test, W1)
A1 = sigmoid(Z1)
Z2 = np.dot(A1, W2)
A2 = sigmoid(Z2)
acc = accuracy(A2, y_test)

print("Test Accuracy: {:.2f}".format(acc))

# Show some predictions
predictions = A2.argmax(axis=1)
print("Predictions (first 3):", predictions[:3])
