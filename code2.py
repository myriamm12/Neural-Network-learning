
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def set(X, Y, hidden_size):
    np.random.seed(3)
    input_size = X.shape[0]
    output_size = Y.shape[0]
    W1 = np.random.randn(2, 2)
    b1 = np.zeros((2, 1))
    W2 = np.random.randn(output_size, 2)
    b2 = np.zeros((output_size, 1))
    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}


def initialize_parameters(layers_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(
            layers_dims[l], layers_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert parameters["W" + str(l)].shape == (
            layers_dims[l], layers_dims[l - 1])
        assert parameters["b" + str(l)].shape == (layers_dims[l], 1)

    #return parameters


def forwardP(X, params):
    Z1 = np.dot(params['W1'], X) + params['b1']
    A1 = sigmoid(Z1)
    Z2 = np.dot(params['W2'], A1) + params['b2']
    dsigmoid(Z2)

    y = sigmoid(Z2)
    return y, {'Z1': Z1, 'Z2': Z2, 'A1': A1, 'y': y}

def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation_fn):
    assert activation_fn == "sigmoid" or activation_fn == "tanh" or \
        activation_fn == "relu"

    if activation_fn == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation_fn == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)

    elif activation_fn == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        #A, activation_cache = relu(Z)

    assert A.shape == (W.shape[0], A_prev.shape[1])

    cache = (linear_cache, activation_cache)
    return A, cache


def dsigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = s * (1 - s)
    return dZ


def cost(predict, actual):
    error_out = (np.power((predict - actual), 2))
    return error_out.sum()


def backP(X, Y, params, cache):
    m = X.shape[1]
    dy = cache['y'] - Y
    dz2 = dy * dsigmoid(cache['Z2'])
    dLoss_A1 = np.dot(params["W2"].T, dz2)

    dW2 = (1 / m) * np.dot(dz2, np.transpose(cache['A1']))
    db2 = (1 / m) * np.sum(dy, axis=1, keepdims=True)
    dZ1 = dLoss_A1 * dsigmoid(cache['Z1'])
    dW1 = (1 / m) * np.dot(dZ1, np.transpose(X))
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update(gradients, params, learning_rate):
    W1 = params['W1'] - learning_rate * gradients['dW1']
    b1 = params['b1'] - learning_rate * gradients['db1']
    W2 = params['W2'] - learning_rate * gradients['dW2']
    b2 = params['b2'] - learning_rate * gradients['db2']
    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}


def fit(X, Y, learning_rate, hidden_size, number_of_iterations=10000):
    params = set(X, Y, hidden_size)
    cost_ = []
    for j in range(number_of_iterations):
        y_pred, cache = forwardP(X, params)
        costit = cost(y_pred, Y)
        gradients = backP(X, Y, params, cache)
        params = update(gradients, params, learning_rate)
        cost_.append(costit)
    return params, cost_

import pandas as pd

data = pd.read_csv('/Users/maryam/Desktop/term6/HooshMohasebati/dataset.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

from sklearn.utils import shuffle

X, Y = shuffle(X, Y)
X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
plt.figure(figsize=(10, 7))
X, Y = X.T, Y.reshape(1, Y.shape[0])
X_test, y_test = X_test.T, y_test.reshape(1, y_test.shape[0])
params, cost_ = fit(X, Y, 8, 2, 8000)

y_pred, cache = forwardP(X_test, params)

for i in range(0, len(y_pred.T)):
    if (y_pred[0][i] >= 0.5):
        y_pred[0][i] = 1
    else:
        y_pred[0][i] = 0

option = 0
for i in range(0, len(y_test.T)):

    if (y_test[0][i] == y_pred[0][i]):
        option = option + 1
print("percentage is :")
print(option / 40 * 100)


plt.plot(cost_)
X_test, y_test = X_test.T, y_test
plt.figure(figsize=(10, 7))
y_pred = y_pred.reshape(36)
y_test = y_test.reshape(36)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
plt.show()

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
plt.show()



