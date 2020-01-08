import numpy as np
from Utils import sigmoid, relu, sigmoid_derivative, relu_derivative, tanh_derivative

class NeuralNetwork:
    def __init__(self):
        """
        Implements a simple deep neural network.
        """
        self.L = 0
        self.learning_rate = 0.05
        self.__parameters = []
        self.__d_act = {'sigmoid': (sigmoid, sigmoid_derivative),
                        'tanh': (np.tanh, tanh_derivative),
                        'relu': (relu, relu_derivative)}

    def add_layer(self, nodes, activation, inputs=None):
        """
        Adds a single hidden layer to the network.

        :param nodes: number of hidden nodes
        :param activation: activation function ('sigmoid', 'tanh', 'relu')
        :param inputs: number of inputs
        """
        if inputs is None:
            inputs = self.__parameters[-1]['W'].shape[0]
        self._add_layer_private(inputs, nodes, activation)

    def fit(self, X, Y, iterations, verbose=False):
        """
        Trains the neural network using train inputs and outputs.

        :param X: train inputs
        :param Y: train outputs
        :param iterations: number of iterations
        :param verbose: makes the process print outputs
        """
        for i in range(iterations):
            A, a_cache = self._forward_propagation(X)
            grads = self._backward_propagation(X, Y, a_cache)
            self._update_parameters(grads, self.learning_rate)

            cost = self._compute_cost(Y, A)
            if verbose and (i == 0 or (i+1) % 100 == 0):
                accuracy = self._compute_accuracy(Y, A)
                print(f"Iteration {i+1:<4} cost: {cost:>7.5f} accuracy: {accuracy:>5.2f}%")

    def predict(self, X):
        """
        Outputs a prediction with the current state of the network.

        :param X: inputs
        :return: predictions
        """
        A, _ = self._forward_propagation(X)
        return np.round(A)

    @staticmethod
    def _initialize_parameters(inputs, nodes):
        """
        Randomly initializes parameters for a single layer.

        :param inputs: number of inputs
        :param nodes: number of hidden nodes
        :return: W and b
        """
        W = np.random.randn(nodes, inputs) / np.sqrt(inputs)
        b = np.zeros((nodes, 1))

        parameters = {'W': W, 'b': b}
        return parameters

    @staticmethod
    def _compute_cost(Y, A):
        """
        Computes the cost of the iteration using binary cross-entropy.

        :param Y: expected results
        :param A: predictions
        :return: cost
        """
        m = Y.shape[1]
        return -1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))

    @staticmethod
    def _compute_accuracy(Y, A):
        """
        Computes the % accuracy after the iteration.

        :param Y: expected output
        :param A: predictions
        :return: accuracy
        """
        return 100 - np.mean(np.abs(Y - np.round(A))) * 100

    def _get_activation_function(self, activation):
        """
        Gets the activation function.

        :param activation: name of the activation
        :return: activation function
        """
        return self.__d_act[activation][0]

    def _get_derivative_function(self, activation):
        """
        Gets the derivative of the activation function.

        :param activation: name of the activation
        :return: derivative of the activation function
        """
        return self.__d_act[activation][1]

    def _add_layer_private(self, inputs, nodes, activation):
        """
        Does the operations to add a new layer to the network.

        :param inputs: number of inputs
        :param nodes: number of nodes
        :param activation: activation function ('sigmoid', 'tanh', 'relu')
        """
        self.L += 1
        parameters = self._initialize_parameters(inputs, nodes)
        parameters['a'] = activation
        self.__parameters.append(parameters)

    def _forward_propagation(self, X):
        """
        Computes the forward propagation steps for the neural network.

        :param X: inputs
        :return: predictions and cache of A
        """
        A = X
        a_cache = []
        for parameters in self.__parameters:
            W = parameters['W']
            b = parameters['b']
            activation = self._get_activation_function(parameters['a'])

            Z = W @ A + b
            A = activation(Z)

            a_cache.append(A)
        return A, a_cache

    def _backward_propagation(self, X, Y, a_cache):
        """
        Computes the backpropagation steps for the neural network.

        :param X: inputs
        :param Y: expected results
        :param a_cache: cache containing the A matrices of every layer
        :return: gradients
        """
        grads = []
        m = X.shape[1]
        A = a_cache.pop()
        dA = (1-Y)/(1-A) - Y/A
        for l in reversed(range(self.L)):
            next_A = a_cache.pop() if l!=0 else X

            parameters = self.__parameters[l]
            W = parameters['W']
            derivative = self._get_derivative_function(parameters['a'])

            dZ = dA * derivative(A)
            dW = 1/m * (dZ @ next_A.T)
            db = 1/m * np.sum(dZ, axis=1, keepdims=True)
            dA = W.T @ dZ

            grads_entry = {'dW': dW, 'db': db}
            grads.append(grads_entry)
            A = next_A
        grads.reverse()
        return grads

    def _update_parameters(self, grads, learning_rate):
        """
        Updates the network parameters after the results produced by backpropagation.

        :param grads: gradients
        :param learning_rate: learning rate
        """
        for l in range(self.L):
            self.__parameters[l]['W'] -= learning_rate * grads[l]['dW']
            self.__parameters[l]['b'] -= learning_rate * grads[l]['db']
