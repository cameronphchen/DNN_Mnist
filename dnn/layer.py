import numpy as np
from .helper import sigmoid, sigmoid_d, relu, relu_d, tanh, tanh_d


class Layer(object):
    def _setup(self):
        raise NotImplementedError()

    def forwardprop(self, inp):
        raise NotImplementedError()

    def backprop(self, out_grad):
        raise NotImplementedError()


class LinearLayer(Layer):
    def __init__(self, outsize, Wscale, Wdecay):
        self.outsize = outsize
        self.Wscale = Wscale
        self.Wdecay = Wdecay

    def _setup(self, insize, rnd):
        W_shape = (self.outsize, insize)
        self.W = rnd.normal(W_shape, scale=self.Wscale)
        self.b = np.zeros(self.outsize)

    def forwardprop(self, inp):
        self.last_inp = inp
        return self.W.dot(inp) + self.b

    def backprop(self, out_grad, learning_rate):
        # calculate gradient
        self.dW = np.outer(out_grad, self.last_inp)
        self.db = np.mean(out_grad, axis=1)
        # update parameters
        self.W -= learning_rate*self.dW
        self.b -= learning_rate*self.db

        return self.W.T.dot(out_grad)


class ActivationLayer(Layer):
    def __init__(self, act_type):
        if act_type == 'sigmoid':
            self.func = sigmoid
            self.func_d = sigmoid_d
        elif act_type == 'relu':
            self.func = relu
            self.func_d = relu_d
        elif act_type == 'tanh':
            self.func = tanh
            self.func_d = tanh_d
        else:
            raise ValueError('invalid activation type')

    def forwardprop(self, inp):
        self.last_inp = inp
        return self.func(inp)

    def backprop(self, out_grad):
        return out_grad*self.func_d(self.last_inp)


class SoftMax(Layer):

    def forwardprop(self, inp):
        return np.exp(inp)/np.sum(np.exp(inp))

    def backprop(self):
        raise NotImplementedError('output layer, no back prop')

    def loss(self, y, y_pred):
        loss = -np.sum(np.multiply(y, y_pred))
        return loss/y.shape[0]
