import numpy as np


class Layer(object):
    def _setup(self):
        raise NotImplementedError()

    def forwardprop(self):
        raise NotImplementedError()

    def backprop(self):
        raise NotImplementedError()


class LinearLayer(Layer):
    def __init__(self, outsize, Wscale, Wdecay):
        self.outsize = outsize
        self.Wscale = Wscale
        self.Wdecay = Wdecay

    def _setup(self, insize, rnd):
        W_shape = (self.outsize, insize)
        self.W = rnd.normal(W_shape, scale=Wscale)
        self.b = np.zeros(self.outsize)

    def forwardprop(self, inp):
        return self.W.dot(inp) + self.b

    def backprop(self):
        # TODO


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
