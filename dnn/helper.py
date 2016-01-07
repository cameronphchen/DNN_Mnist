import numpy as np


def onehot(labels):
    classes = np.unique(labels)
    num_classes = classes.shape[0]
    onehot_labels = np.zeros(labels.shape + (num_classes,))
    for c in classes:
        onehot_labels[labels == c, c] = 1
    return onehot_labels


def unhot(one_hot_labels):
    return np.argmax(one_hot_labels, axis=-1)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_d(x):
    s = sigmoid(x)
    return s*(1-s)


def relu(x):
    return np.max(0, x)


def relu_d(x):
    dx = np.zeros(x.shape)
    dx[x >= 0] = 1
    return dx


def tanh(x):
    return np.tanh(x)


def tanh_d(x):
    e = np.exp(2*x)
    return (e-1)/(e+1)