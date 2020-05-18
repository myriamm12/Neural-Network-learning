import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
data = pd.read_csv('/Users/maryam/Desktop/term6/HooshMohasebati/dataset.csv')


X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X, y = shuffle(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
X, X_test, y, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
np.random.seed(0)
plt.figure(figsize=(10, 7))
y = y.reshape(144)


def learn(X, Y, epochs=1):
    # weights be the lenght of columns of dataset
    weights = np.zeros(len(X[0]) + 1)

    # set learning rate
    eta = 1

    # lets monitor errors
    errors_list = []

    for learning_round in range(epochs):
        #print("---- Learning {} -----".format(learning_round))
        # to calculate error at every round/epoch
        total_error = 0

        for x, y in zip(X, Y):

            prediceted_out = predict(x, weights)
            error = eta * (y - prediceted_out)
            weights[1:] = weights[1:] + (x * error)
            weights[0] = weights[0] + error

            total_error += abs(error)

            #print("x= {}, weights= {}, y= {} error= {}  predicted = {}".format(x, weights, y, error, prediceted_out))

        errors_list.append(error)

    plt.plot(errors_list)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')

    #return weights

def perceptron(epochs, eta):

    w = np.random.rand(2, 1)
    b = np.random.randn(1, 1)

    for i in range(0, epochs):

        myZ = np.dot(X, w) + b

        Y = sigmoid(myZ)


        Yt = y.reshape(144, 1)

        error = -(Yt/Y + (Yt-1)/(1-Y))*(1/2.30258)


        dpred_dz = sigmoid_derivative(Y)

        z_delta = error * dpred_dz

        input = X.T

        b = z_delta
        w = w - (eta * np.dot(input, z_delta) * 1 / 144)

    return w


def predict(row, weights):
    # takes a row [x1, x2, x3.....]
    # weight [bias, w1, w2, w3 .....]

    # lets initiate activation with bias
    # its equal to
    # activation = sum(weight_i * x_i) + bias
    # Step activation function ->  prediction = 1.0 if activation >= 0.0 else 0.0
    # Bias is needed to pull the values up
    # y = ax + b(bias)

    activation = weights[0]
    just_weights = weights[1:]

    for x, w in zip(row, just_weights):
        activation += x * w
    #return 1.0 if activation >= 0 else -1.0


# So with the given weights actual value and predicted is correct

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':

    w = perceptron(epochs=2000, eta=0.05)

    XW = np.dot(X_test, w)

    z = sigmoid(XW)

    s = z
    for i in range(0, len(z)):
        if (z[i][0] >= 0.5):
            z[i][0] = 1
        else:
            z[i][0] = 0

    option = 0
    y_test = y_test.reshape(36, 1)
    for i in range(0, len(y_test)):
        if (y_test[i][0] == z[i][0]):
            option = option + 1
    print("percentage is :")
    print(option / 40 * 100)
    print("cost is:")
    plt.figure(figsize=(10, 7))
    z = z.reshape(36)
    y_test = y_test.reshape(36)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    plt.show()
    plt.scatter(X_test[:, 0], X_test[:, 1], c=z)
    plt.show()

    error_out = ((1 / 2) * (np.power((z - y_test), 2)))
    print(np.sum(error_out))







