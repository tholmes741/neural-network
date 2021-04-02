#!/usr/bin/env python3

import random
import numpy as np

class Network:

    def __init__(self, sizes):
        self.layers = len(sizes)
        self.sizes = sizes
        
        """Biases is an array of arrays. Each array is instantiated 
        with a bias for every node in the network excluding the input
        nodes.
        """
        self.biases = [np.random.randn(x, 1) for x in self.sizes[1:]]
        """Creates weights for each node connecting to each node in the
        next layer. 
        """
        self.weights = [np.random.randn(y, x)/np.sqrt(x) 
            for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        

        print('init with {}'.format(self.sizes))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, batch_size, epochs, training_rate):
        """we gon be doing a lot of stuff"""
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, training_rate)
            print("Epoch {} completed".format(j + 1))

    def update_batch(self, batch, training_rate):
        new_biases = [np.zeros(b.shape) for b in self.biases]
        new_weights = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            temp_biases, temp_weights = self.backprop(x, y)
            new_biases = [nb +tb for nb, tb in zip(new_biases, temp_biases)]
            new_weights = [nw +tw for nw, tw in zip(new_weights, temp_weights)]
        self.biases = [b - (training_rate/len(batch)*nb) for b, nb in zip(self.biases, new_biases)]
        self.weights = [w - (training_rate/len(batch))*nw for w, nw in zip(self.weights, new_weights)]

    def backprop(self, x, y):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b , w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activations[-2].transpose())

        for layer in range(2, self.layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            delta_b[-layer] = delta
            delta_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return (delta_b ,delta_w)

    def evaluate(self, test_data):
        results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in results)


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x): 
    return sigmoid(x) * (1 - sigmoid(x))