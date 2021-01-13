import h5py as h5
import numpy as np
from matplotlib import pyplot as plt

# In order to employ corr2 algorithm of MATLAB in python
# We should create two functions having the same results with MATLAB functions
def mean2(x):
    return np.sum(x)/np.size(x)


def corr2(x, y):
    a = x - mean2(x)
    b = y - mean2(y)
    return np.sum(a * b)/ np.sqrt( np.sum(a * a) * np.sum(b * b))


# Load the data
file = h5.File('C:/Users/Sanyu Buke Badrac/Desktop/EEE 443/assignment1/assign1_data1.h5', "r")
train_data = np.array(file["trainims"])
train_data = np.swapaxes(train_data, 2, 1)
test_data = np.array(file["testims"])
test_data = np.swapaxes(test_data, 2, 1)
train_label = np.array(file["trainlbls"])
test_label = np.array(file["testlbls"])

# Examine the dataset in order to observe distribution of classes
(unique, counts) = np.unique(train_label, return_counts=True)
frequencies = np.asarray((unique, counts)).T

# Calculate the correlation matrix by 2D correlation
corr_mat = np.zeros((26, 26))
print(corr2(train_data[200, :, :], train_data[200, :, :]))
for i in range(26):
    for j in range(i+1):
        corr_coef = corr2(train_data[200*i, :, :], train_data[200*j, :, :])
        corr_mat[i, j] = corr_coef
        corr_mat[j, i] = corr_coef


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def one_hot_rep(y):
    x = np.zeros((26, 1))
    x[int(y - 1)] = 1
    return x


def fit(train_data, train_label, learning_rate):
    # Normalize data to accelerate traning process This is because, when normalized GD algorithm is going to make
    # small changes towards a minima rather than wobbling around
    train_data = train_data / 255
    # Initialize weights by given conditions
    np.random.seed(1)
    # W, b = 0.1*np.random.randn(26, 784), 0.1*np.random.randn(26, 1)
    W, b = np.random.normal(0, 0.1, (26, 784)), np.random.normal(0, 0.1, (26, 1))

    iteration = 10000
    se = []
    mse = []
    output = np.zeros((26, 1), dtype=int)
    for i in range(iteration):
        index = np.random.randint(low=0, high=np.size(train_label))
        y = one_hot_rep(train_label[index])
        img = train_data[index, :, :]
        img = img.flatten()
        img = img[:, np.newaxis]
        output = output * 0

        # Forward propagation
        V = W @ img + b
        a = sigmoid(V)
        max_index = np.argmax(a)
        output[max_index] = 1

        # Backward propagation
        C = (a - y) * (1 - a) * a
        dW = C @ img.T
        db = -C

        # Update weights
        W = W - learning_rate * dW
        b = b - learning_rate * db

        error = (1 / 52) * ((y - a).T) @ (y - a)  # obtaining the square error for the instance
        se.append(error[0][0])  # adding it to the total sum of mse
        mse.append(np.mean(se))

    weights = (W, b)
    return weights, mse


def evaluate(test_data, test_label, weights):
    W, b = weights
    test_data = test_data / 255

    accuracy = 0
    se = []
    mse = []
    output = np.zeros((26, 1), dtype=int)
    for i in range(np.size(test_label)):
        y = one_hot_rep(test_label[i])
        img = test_data[i, :, :]
        img = img.flatten()
        img = img[:, np.newaxis]
        output = output * 0

        # Forward pass
        V = W @ img + b
        a = sigmoid(V)
        max_index = np.argmax(a)
        output[max_index] = 1

        error = (1 / 52) * ((y - a).T) @ (y - a)  # obtaining the square error for the instance
        se.append(error[0][0])  # adding it to the total sum of mse
        mse.append(np.mean(se))
        accuracy = accuracy + 1 * np.array_equal(output, y)

    accuracy = accuracy / np.size(test_label)

    return mse, accuracy

weights, mse = fit(train_data, train_label, 0.1)
mse, accuracy = evaluate(test_data, test_label, weights)