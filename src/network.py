#!/usr/bin/env python3

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
        

        print('init with {}'.format(self.sizes[1:]))
