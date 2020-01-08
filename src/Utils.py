import numpy as np
import h5py

def load_input(path):
    data = {}
    with h5py.File(path, 'r') as file:
        for name in file.keys():
            x = file[name][:]
            data[name] = x
    return data

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def sigmoid_derivative(a):
    return a * (1-a)

def relu_derivative(a):
    d = np.zeros(a.shape)
    d[a > 0] = 1
    return d

def tanh_derivative(a):
    return 1 - np.power(a, 2)