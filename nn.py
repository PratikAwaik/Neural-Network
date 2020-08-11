# Implementation of simple neural network.

import numpy as np
import scipy.optimize 

class NeuralNetwork:
    """
    Artificial Neural Network.
    The Cost Function is minimized using scipy.
    alpha = learning rate, default = 1

    epsilon: float, default = 0.12
        When initializing weights the values of the weights are in the range [-epsilon, epsilon]

    maxiter: int, default = 100

    num_units: int
        Number of neurons in each 'hidden' layer.

    X: array-like, input 
    y: array-like, output
    """
    def __init__(self, *num_units, alpha=1, epsilon=0.12, maxiter=100):
        self.alpha = alpha
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.best_theta_ = None
        self.num_units = []
        for layer in num_units:
            self.num_units.append(layer)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def initializeWeights(self, X, y):
        """Returns a list of randomly initialized weights based on the number of layers."""
        input_layer_size = X.shape[1]  
        output_layer_size = len(np.unique(y)) 

        self.num_units.insert(0, input_layer_size)  # add an input layer 
        self.num_units.append(output_layer_size)   # add an output layer

        weights = []
        for layer in range(len(self.num_units) - 1):
            theta = np.random.rand(self.num_units[layer + 1], self.num_units[layer] + 1) * 2 * self.epsilon - self.epsilon
            weights.append(theta)

        return weights

    def unrollMatrices(self, theta):
        """
        Returns an unrolled version of the matrices. 
        theta: list, rolled version of the matrices
        """
        # turn each weight to 1 dimensional vector
        ravels = list(map(lambda x: x.ravel(), theta))

        # concatenate every weight, returns 1 dimensional vector
        unrolled = np.concatenate(ravels, axis=0)
        return unrolled

    def rollVector(self, theta):
        """
        Returns the rolled version of the vector. In particular, returning all the weights with their correct dimensions.
        theta: unrolled version of the matrices.
        """
        rolled = []
        first, last = 0, 0

        # assumes self.num_units has input_layer_size and output_layer_size
        for i in range(len(self.num_units) - 1):
            last += self.num_units[i + 1] * (self.num_units[i] + 1)
            rolled.append(theta[first:last].reshape((self.num_units[i + 1], self.num_units[i] + 1)))
            first += self.num_units[i + 1] * (self.num_units[i] + 1)

        return rolled

    def costFunction(self, theta, X, y):
        """Returns the cost and gradients(in an unrolled version)."""
        # roll the vector
        theta = self.rollVector(theta)
        m, n = X.shape

        # implement feed forward propogation
        activations = [X]
        for t in theta:
            act = activations[-1]
            activations[-1] = np.c_[np.ones((len(act), 1)), act]
            x = self.sigmoid(activations[-1] @ t.T)
            activations.append(x)

        h = activations[-1]  # m x k (k = num of labels)

        # convert y (vector) to sparse matrix (one-hot encoding)
        # self.num_units[-1] = output_layer_size (assuming self.num_units has input and output layer sizes)
        new_y = np.zeros((m, self.num_units[-1]))  
        for label in range(self.num_units[-1]):
            new_y[:, label: label + 1] = (y == label).astype(np.int)

        # Cost Function
        J = -(1 / m) * np.sum((np.log(h) * new_y) + np.log(1 - h) * (1 - new_y))

        # adding regularization
        t = list(map(lambda x: np.sum(np.sum(x[:, 1:] ** 2)), theta))
        reg = (self.alpha / ( 2 * m)) * np.sum(t)
        J = J + reg

        # implement Backpropogation
        # calculating errors (we don't calculate the error of input layer)
        """
            if 4 layers    50, 25, 25, 10
            theta = [t1, t2, t3]   25x51, 25x26, 10x26
            actiavtions = [a1, a2, a3, a4]   mx51, mx26, mx26, mx10 
            d = [d4, d3, d2]   mx10, mx26, mx25    no d1
            theta_grads = [tg1, tg2, tg3]   25x51, 26x26, 10x26
        """
        deltas = [(h - new_y)]
        for i in range(len(theta) - 1, 0, -1):
            delta = (deltas[-1] @ theta[i]) * (activations[i] * (1 - activations[i]))
            deltas.append(delta[:, 1:])

        # calculating gradients 
        theta_grads = []
        for i in range(len(theta)):
            grad = (1 / m) * deltas[len(theta) - i - 1].T @ activations[i]
            grad[:, 1:] += (self.alpha / m) * theta[i][:, 1:]
            theta_grads.append(grad)

        # unroll the gradients 
        grads = self.unrollMatrices(theta_grads)

        return [J, grads]
    
    def fit(self, X, y):
        """Returns optimal values of the weights using scipy.optimize.minimize"""
        print("Training Neural Network...")

        # initializing weights
        init_theta = self.unrollMatrices(self.initializeWeights(X, y))

        # create a shorthand for the costfunction to be minimized
        cf = lambda p: self.costFunction(p, X, y)

        # apply scipy.optimize.minimize
        options = {"maxiter": self.maxiter}

        print("Minimizing Cost Function... this may take some time...")
        result = scipy.optimize.minimize(
            cf,
            init_theta, 
            jac=True, 
            method="TNC", 
            options=options
        )

        theta = result.x
        self.best_theta_ = self.rollVector(theta)  # list of rolled matrices
        print("Training Done...")
        return 

    def predict(self, X):
        """Returns the predictions."""
        theta = self.best_theta_
        activations = [X]
        for t in theta:
            act = activations[-1]
            activations[-1] = np.c_[np.ones((len(act), 1)), act]
            x = self.sigmoid(activations[-1] @ t.T)
            activations.append(x)
        h = activations[-1]
        return np.argmax(h, axis=1).reshape(-1, 1)
