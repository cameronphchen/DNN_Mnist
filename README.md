# DNN_Mnist
Feedforward Neural Network implementation for MNIST

[![Build Status](https://travis-ci.org/cameronphchen/DNN_Mnist.svg?branch=master)](https://travis-ci.org/cameronphchen/DNN_Mnist)

2016/1/1:
Start implementation of multilayer perceptron (MLP) for MNIST recognition

reference:
- http://deeplearning.net/tutorial/mlp.html
- https://github.com/andersbll/nnet

Modules:
layers.py
    layer module defines different type of layers
    inputs to each layer are (1) layer type (2) parameters
    linear layer, activation layer

neuralnetwork.py
    input to neuralnetwork object is a list of layers


Let's say we want to implement a single-hidden layer multi-layer perceptron (MLP)


x = MNIST dataset with dimension 784
h1 = s(W1*x  + b1)
y  = g(W2*h1 + b2)

linear layer
nonlinearity
output layer

2016/1/3 
set up travis-CI for testing
