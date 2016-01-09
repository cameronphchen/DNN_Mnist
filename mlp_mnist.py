import sklearn.datasets
import numpy as np
from dnn import layer, neuralnetwork

mnist = sklearn.datasets.fetch_mldata('mnist-original', data_home='data/input')
X_trn = mnist.data[:60000].T/255.0
X_tst = mnist.data[60000:].T/255.0
Y_trn = mnist.target[:60000].T
Y_tst = mnist.target[60000:].T
num_classes = np.unique(mnist.target).shape[0]


# Downsample training data
num_train_samples = 10000
train_idxs = np.random.random_integers(0, 60000-1, num_train_samples)
X_trn_ds = X_trn[:, train_idxs]
Y_trn_ds = Y_trn[train_idxs]

# Defind network structure
layers = [
    layer.LinearLayer(outsize=100, Wscale=0.2, Wdecay=0.004),
    layer.ActivationLayer('sigmoid'),
    layer.LinearLayer(outsize=50, Wscale=0.2, Wdecay=0.004),
    layer.ActivationLayer('sigmoid'),
    layer.LinearLayer(outsize=num_classes, Wscale=0.2, Wdecay=0.004),
    layer.SoftMaxLayer()
]

# set up MLP and train
network = neuralnetwork.NeuralNetwork(layers)
network.train(X_trn_ds, Y_trn_ds)

test_err = network.error(X_tst, Y_tst)

print 'test error :{}'.format(test_err)
