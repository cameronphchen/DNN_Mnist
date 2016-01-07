import numpy as np
from .helper import onehot, unhot


class NeuralNetwork(object):
    def __init__(self, layers):
        self.layers = layers

    def train(X_trn, Y_trn, learning_rate=0.1, max_iter=10, batch_size=64):
        num_batches = X_trn.shape[1]//batch_size

        Y_trn_oh = onehot(Y_trn)

        # SGD with mini-batches
        for iter in xrange(max_iter):

            for batch in xrange(num_batches):
                b_start = batch*batch_size
                b_end = b_start+batch_size
                X_batch = X_trn[b_start:b_end]
                Y_batch = Y_trn_oh[b_start:b_end]

                # forprop
                H_l = X_batch
                for layer in self.layers:
                    H_l = layer.forwardprop(H_l)
                Y_pred = H_l

                # backprop and update parameters
                out_grad = self.layers[-1].out_grad(Y_batch, Y_pred)
                for layer in reversed(self.layers[:-1]):
                    out_grad = layer.backprop(out_grad)

            # output training status
            loss = self._loss(X_trn, Y_trn_oh)
            error = self.error(X_trn, Y_trn)

    def _loss(self, X, Y_oh):
        H_l = X
        for layer in self.layers:
            H_l = layer.forwardprop(H_l)
        Y_pred = H_l
        return self.layers[-1].lostt(Y_oh, Y_pred)

    def predict(self, X):
        H_l = X
        for layer in self.layers:
            H_l = layer.forwardprop(H_l)
        Y_pred = H_l
        return unhot(Y_pred)
    
    def error(self, X, Y):
        Y_pred = self.predict(X)
        error = Y_pred != Y
        return np.mean(error)
        
