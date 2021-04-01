#!/usr/bin/env python3

from src.network import Network
import src.mnist_loader as mnist_loader

import numpy as np

def main():
    net = Network((784, 30, 10))
    training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()
    net.SGD(training_data, 30, 10, 3.0, test_data)

if __name__ == '__main__': main()
