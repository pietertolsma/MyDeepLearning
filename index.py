import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import numpy as np

from model import NeuralNet
from mymath import relu, logistic_sigmoid

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# data is comprised of tuples (img, label index) where
# label index corresponds to the names in the labels_map.
# Each image is a 28x28 matrix of floats.
# Hence the first layer will have 28 * 28 = 784 neurons.
# The output layer will be a vector of 10 values (after a logistic sigmoid function)
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

model = NeuralNet(28*28, 10)
model.add_layer(300, relu)
model.add_layer(200, relu)
model.add_layer(10, logistic_sigmoid)

for img, label in training_data:
    input_data = img.numpy().flatten()
    result = model.process(input_data)

    result[label] -= 1
    error = result
    model.train(error)

