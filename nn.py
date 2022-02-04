import math

import numpy as np
from activation_functions import *

def batchnorm_forward(x):
    """
    Forward pass for batch normalization.

    Input:
    - x: Input vector
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    eps = 1e-5
    momentum = 0.1
    gamma = 1
    beta = 0
    sample_mean = np.mean(x, axis=0)
    sample_var = np.var(x, axis=0)

    # Normalization followed by Affine transformation
    x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)
    out = gamma * x_normalized + beta
    return out


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # initializing layers
        self.input_layer = layer_sizes[0]
        self.hidden_layer = layer_sizes[1]
        self.output_layer = layer_sizes[2]

        # initializing wights
        w1 = np.random.randn(self.hidden_layer, self.input_layer)
        w2 = np.random.randn(self.output_layer, self.hidden_layer)
        w = [w1, w2]

        # initializing biases
        b1 = np.zeros((1, self.hidden_layer))
        b2 = np.zeros((1, self.output_layer))
        b = [b1, b2]

        self.biases = b
        self.weights = w

    def activation(self, x, flag):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param flag: Decide between Sigmoid and ReLU activation functions
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        if flag:
            return sigmoid(x)
        else:
            return relu(x)

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        flag = True
        x = batchnorm_forward(x)
        z1 = (self.weights[0] @ x) + self.biases[0]
        a1 = np.asarray(self.activation(z1, flag)).reshape(self.hidden_layer, 1)
        a1 = batchnorm_forward(a1)
        z2 = np.add((self.weights[1] @ a1), self.biases[1])
        a2 = self.activation(z2, flag)
        return a2
