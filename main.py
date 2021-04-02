#!/usr/bin/env python3

from src.network import Network
import src.mnist_loader as mnist_loader

import numpy as np

def main():
    net = Network((784, 30, 10))
    training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()

    print("Network initial performance: {0}/{1}"
        .format(net.evaluate(test_data), len(test_data)))

    net.SGD(training_data, 30, 10, 3.0)

    print("How does our network perform after training? {0}/{1} right!" 
        .format(net.evaluate(test_data), len(test_data)))

if __name__ == '__main__': main()
