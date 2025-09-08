# week-2

# exp-2 : Implement of feed forword neural network using python

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(0)
X,y = datasets.make_moons(200,noise = 0.20)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output = False)
Y = encoder.fit_transform(y.reshape(-1,1))
input_dim = 2
hidden_dim = 6
output_dim = 3
W1= np.random.randn(input_dim, hidden_dim)
b1 = np.zeros((1, hidden_dim))
W2= np.random.randn(hidden_dim, hidden_dim)
b2 = np.zeros((1, hidden_dim))
W3= np.random.randn(hidden_dim, output_dim)
b3 = np.zeros((1, output_dim))
z1 = np.dot(X,W1) + b1
a1 = np.tanh(z1)
z2 = np.dot(a1, W2) + b2
a2 = np.tanh(z2)
z3 = np.dot(a2,W3) + b3
exp_scores = np.exp(z3 - np.max(z3, axis = 1,keepdims = True))
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims = True )

print("Predicted probabilitites (first 5 rows):\n", probs[:5])
