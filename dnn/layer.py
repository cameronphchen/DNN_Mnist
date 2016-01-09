import numpy as np
from .helper import sigmoid, sigmoid_d, relu, relu_d, tanh, tanh_d


class Layer(object):
    def _setup(self, insize, rnd):
        pass

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
        self.W = rnd.normal(size=W_shape, scale=self.Wscale)
        self.b = np.zeros(self.outsize)

    def forwardprop(self, inp):
        self.last_inp = inp
        return self.W.dot(inp) + np.outer(self.b, np.ones(inp.shape[1]))

    def backprop(self, out_grad, learning_rate):
        # calculate gradient
        #self.dW = np.outer(out_grad, self.last_inp)
        self.dW = out_grad.dot(self.last_inp.T)
        #n = out_grad.shape[1]
        #self.dW = out_grad.dot(self.last_inp.T)/n - self.Wdecay*self.W
        
        self.db = np.mean(out_grad, axis=1)
        # update parameters
        bp = self.W.T.dot(out_grad)
        self.W -= learning_rate*self.dW
        self.b -= learning_rate*self.db
        return bp

    def getoutsize(self, insize):
        return self.outsize


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

    def backprop(self, out_grad, learning_rate):
        return out_grad*self.func_d(self.last_inp)

    def getoutsize(self, insize):
        return insize


class SoftMaxLayer(Layer):

    def forwardprop(self, inp):
        e = np.exp(inp - np.amax(inp, axis=0, keepdims=True))
        return e/np.sum(e, axis=0, keepdims=True)
        #return np.exp(inp)/np.sum(np.exp(inp))

    def backprop(self):
        raise NotImplementedError('output layer, no back prop')

    def loss(self, y, y_pred):
        #loss = -np.sum(np.multiply(y, y_pred))

        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        y_pred /= y_pred.sum(axis=0, keepdims=True)
        loss = -np.sum(np.multiply(y, np.log(y_pred)))

        return loss/y.shape[1]

    def getoutsize(self, insize):
        return insize

    def out_grad(self, Y, Y_pred):
        # Assumes one-hot encoding.
        return -(Y - Y_pred)
