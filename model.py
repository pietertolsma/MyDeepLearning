import numpy as np
from mymath import logistic_sigmoid

class HiddenLayer():

    def __init__(self, input_count, n):
        self.input_count = input_count
        self.n = n
        self.weights = np.random.randint(10, size=(n, input_count)) / 10 # each row = all weights per neuron
        self.bias = np.random.randint(10, size=n) / 10

    def process(self, input):
        return np.matmul(self.weights, input) + self.bias

class NeuralNet():

    def __init__(self, input_count, output_count):
        self.input_count = input_count
        self.output_count = output_count
        self.hidden_layers = list()

    def addHiddenLayer(self, n):
        input_count = self.input_count
        if (len(self.hidden_layers) > 0):
            input_count = self.hidden_layers[-1].n
        self.hidden_layers.append(HiddenLayer(input_count, n))

    def get_squared_error(self, result, target):
        result[target] -= 1
        return np.sum(result ** 2)

    def train(self, error):
        print("hi")

    def addFinalLayer(self, n):
        self.addHiddenLayer(n)

    def process(self, input):
        temporary_input = input
        for layer in self.hidden_layers:
            temporary_input = logistic_sigmoid(layer.process(temporary_input))
        return temporary_input / np.sum(temporary_input)