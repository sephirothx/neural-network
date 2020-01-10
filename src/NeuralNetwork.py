import numpy as np
from Utils import sigmoid, relu, sigmoid_derivative, relu_derivative, tanh_derivative

class NeuralNetwork:
    _d_best_init = {'sigmoid': 'xavier',
                    'tanh': 'xavier',
                    'relu': 'he'}
    _d_act = {'sigmoid': (sigmoid, sigmoid_derivative),
              'tanh': (np.tanh, tanh_derivative),
              'relu': (relu, relu_derivative)}

    def __init__(self):
        """
        Implements a simple deep neural network.
        """
        self.L = 0
        self.learning_rate = 0.05
        self._parameters = []
        self._regularization = lambda m: 0
        self._regularization_back = lambda m, W: 0
        self._lambd = 0.01

    def add_layer(self, nodes, activation, *, inputs=None, initialization=None):
        """
        Adds a single hidden layer to the network.

        :param nodes: number of hidden nodes
        :param activation: activation function ('sigmoid', 'tanh', 'relu')
        :param inputs: number of inputs
        :param initialization: method used to initialize the W matrix ('he', 'xavier', 'zeros').
         If no value is provided the supposedly best one will be chosen by default.
        """
        if inputs is None:
            inputs = self._parameters[-1]['W'].shape[0]
        if initialization is None:
            initialization = self._d_best_init[activation]
        self._add_layer_private(inputs, nodes, activation, initialization)

    def compile(self, *, learning_rate=None, regularization=None, lambd=None):
        """
        Completes the initialization of the neural network providing the last details.

        :param learning_rate: learning rate
        :param regularization: regularization method ('L2')
        :param lambd: coefficient for the regularization
        :return:
        """
        if regularization in ('l2', 'L2'):
            self._regularization = self._l2_regularization
            self._regularization_back = self._l2_regularization_back
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if lambd is not None:
            self._lambd = lambd

    def fit(self, X, Y, iterations, verbose=False, step=100):
        """
        Trains the neural network using train inputs and outputs.

        :param X: train inputs
        :param Y: train outputs
        :param iterations: number of iterations
        :param verbose: makes the process print outputs
        :param step: every how many iterations the outputs will be stored
        """
        costs = []
        for i in range(iterations):
            A, a_cache = self._forward_propagation(X)
            grads = self._backward_propagation(X, Y, a_cache)
            self._update_parameters(grads, self.learning_rate)

            cost = self._compute_cost(Y, A)
            if i == 0 or (i+1) % step == 0:
                costs.append(cost)
                if verbose:
                    accuracy = self._compute_accuracy(Y, A)
                    print(f"Iteration {i+1:<4} "
                          f"cost: {cost:>7.5f} "
                          f"accuracy: {accuracy:>5.2f}%")
        return costs

    def predict(self, X):
        """
        Outputs a prediction with the current state of the network.

        :param X: inputs
        :return: predictions
        """
        A, _ = self._forward_propagation(X)
        return np.round(A)

    @classmethod
    def _initialize_parameters(cls, inputs, nodes, initialization):
        """
        Randomly initializes parameters for a single layer.

        :param inputs: number of inputs
        :param nodes: number of hidden nodes
        :param initialization: method used to initialize the W matrix
        :return: W and b
        """
        if initialization == 'he':
            W = cls._init_parameters_he(inputs, nodes)
        elif initialization == 'xavier':
            W = cls._init_parameters_xavier(inputs, nodes)
        else:
            W = cls._init_parameters_zeros(inputs, nodes)

        b = np.zeros((nodes, 1))

        parameters = {'W': W, 'b': b}
        return parameters

    @staticmethod
    def _init_parameters_xavier(inputs, nodes):
        return np.random.randn(nodes, inputs) * np.sqrt(2 / (inputs+nodes))

    @staticmethod
    def _init_parameters_he(inputs, nodes):
        return np.random.randn(nodes, inputs) * np.sqrt(2 / inputs)

    @staticmethod
    def _init_parameters_zeros(inputs, nodes):
        return np.zeros((nodes, inputs))

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
        return self._d_act[activation][0]

    def _get_derivative_function(self, activation):
        """
        Gets the derivative of the activation function.

        :param activation: name of the activation
        :return: derivative of the activation function
        """
        return self._d_act[activation][1]

    def _add_layer_private(self, inputs, nodes, activation, initialization):
        """
        Does the operations to add a new layer to the network.

        :param inputs: number of inputs
        :param nodes: number of nodes
        :param activation: activation function ('sigmoid', 'tanh', 'relu')
        :param initialization: method used to initialize the W matrix ('he', 'xavier', 'zeros')
        """
        self.L += 1
        parameters = self._initialize_parameters(inputs, nodes, initialization)
        parameters['a'] = activation
        self._parameters.append(parameters)

    def _l2_regularization(self, m):
        return self._lambd / (2*m) * np.sum([np.sum(np.square(p['W'])) for p in self._parameters])

    def _l2_regularization_back(self, m, W):
        return self._lambd / m * W

    def _compute_cost(self, Y, A):
        """
        Computes the cost of the iteration using binary cross-entropy.

        :param Y: expected results
        :param A: predictions
        :return: cost
        """
        m = Y.shape[1]
        return -1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) + self._regularization(m)

    def _forward_propagation(self, X):
        """
        Computes the forward propagation steps for the neural network.

        :param X: inputs
        :return: predictions and cache of A
        """
        A = X
        a_cache = []
        for parameters in self._parameters:
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
        for layer in reversed(range(self.L)):
            next_A = a_cache.pop() if layer != 0 else X

            parameters = self._parameters[layer]
            W = parameters['W']
            derivative = self._get_derivative_function(parameters['a'])

            dZ = dA * derivative(A)
            dW = 1/m * (dZ @ next_A.T) + self._regularization_back(m, W)
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
        for layer in range(self.L):
            self._parameters[layer]['W'] -= learning_rate * grads[layer]['dW']
            self._parameters[layer]['b'] -= learning_rate * grads[layer]['db']
