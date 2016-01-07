import sklearn.datasets
import numpy as np
import dnn

mnist = sklearn.datasets.fetch_mldata('mnist-original', data_home='data/input')
X_trn = mnist.data[:60000].T/255.0
X_tst = mnist.data[60000:].T/255.0
Y_trn = mnist.target[:60000].T
Y_tst = mnist.target[60000:].T
num_classes = np.unique(mnist.target).shape[0]

# Defind network structure
layers = [
    dnn.LinearLayer(outsize=100, Wscale=0.2, Wdecay=0.004),
    dnn.ActivationLayer('relu'),
    dnn.LinearLayer(outsize=50, Wscale=0.2, Wdecay=0.004),
    dnn.ActivationLayer('relu'),
    dnn.LinearLayer(outsize=num_classes, Wscale=0.2, Wdecay=0.004),
    dnn.SoftMaxLayer()
]

# set up MLP and train
network = nn.NeuralNetwork(layers)
network.train(X_trn, Y_trn)
