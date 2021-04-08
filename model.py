import numpy as np
from mymath import logistic_sigmoid, relu

class Layer():

    def __init__(self, input_count, n, activation):
        self.input_count = input_count
        self.n = n
        self.activation = activation
        self.w = np.random.randint(10, size=(n, input_count)) / 10 # each row = all weights per neuron
        self.a = np.zeros(shape=n)
        self.z = np.zeros(shape=n)
        self.b = np.random.randint(10, size=n) / 10

    def process(self, x):
        self.a = np.matmul(self.w, x) + self.b
        self.z = self.activation(self.a)

        return self.z

class NeuralNet():

    def __init__(self, input_count, output_count):
        self.input_count = input_count
        self.output_count = output_count
        self.layers = list()

    def add_layer(self, n, activation):
        input_count = self.input_count
        if (len(self.layers) > 0):
            input_count = self.layers[-1].n
        self.layers.append(Layer(input_count, n, activation))

    def get_squared_error(self, result, target):
        result[target] -= 1
        return np.sum(result ** 2)

    def train(self, error):
        print(error)

    def process(self, input):
        temporary_input = input
        for layer in self.layers:
            temporary_input = layer.process(temporary_input)
        return temporary_input / np.sum(temporary_input)